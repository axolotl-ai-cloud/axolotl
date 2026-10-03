"""vLLM generation and correctly processed proposal-density scoring."""

import gc
import math
from importlib import import_module
from typing import Any

from ..args import ProjectionSamplingConfig, VLLMBackendOptions, get_seed
from ..backend import SamplingBackend


class VLLMBackend(SamplingBackend):
    """Use raw prompt scores for targets and processed next-token scores for MH."""

    def __init__(
        self,
        engine,
        tokenizer,
        config: ProjectionSamplingConfig,
        eos_token_ids: set[int] | None = None,
        seed: int = 42,
    ):
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config
        self.seed = seed
        self.options = VLLMBackendOptions.model_validate(config.backend_kwargs)
        self.sampling_params = import_module("vllm").SamplingParams
        self.specific_logprobs = "logprob_token_ids" in getattr(
            self.sampling_params, "__struct_fields__", ()
        )
        model_config = engine.llm_engine.model_config
        self.max_model_len = model_config.max_model_len
        eos = model_config.hf_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_token_ids = (
            eos_token_ids
            if eos_token_ids is not None
            else set(eos if isinstance(eos, list) else [eos]) - {None}
        )
        if len(tokenizer) > model_config.get_vocab_size():
            raise ValueError(
                "Projection sampling does not support adding tokens to the base model vocabulary"
            )
        self.request_number = 0

    @classmethod
    def from_config(cls, cfg, config: ProjectionSamplingConfig):
        try:
            vllm = import_module("vllm")
        except ImportError as exc:
            raise ImportError(
                "Install Axolotl's vllm extra (`pip install 'axolotl[vllm]'`) to use backend: vllm"
            ) from exc
        from transformers import GenerationConfig

        from axolotl.loaders import load_tokenizer

        options = VLLMBackendOptions.model_validate(config.backend_kwargs)
        tokenizer = load_tokenizer(cfg)
        engine = vllm.LLM(
            model=cfg.base_model,
            revision=cfg.revision_of_model,
            trust_remote_code=bool(cfg.trust_remote_code),
            dtype=config.dtype,
            seed=get_seed(cfg),
            skip_tokenizer_init=True,
            generation_config="vllm",
            logprobs_mode="processed_logprobs",
            max_logprobs=-1,
            **options.model_dump(exclude={"score_batch_size"}, exclude_none=True),
        )
        try:
            try:
                generation = GenerationConfig.from_pretrained(
                    cfg.base_model, revision=cfg.revision_of_model
                )
            except OSError:
                generation = GenerationConfig.from_model_config(
                    engine.llm_engine.model_config.hf_config
                )
            eos = generation.eos_token_id
            if eos is None:
                eos = tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
            return cls(engine, tokenizer, config, eos_ids, seed=get_seed(cfg))
        except BaseException:
            engine.llm_engine.engine_core.shutdown()
            raise

    def _check_context(self, context: list[int], length: int):
        if not context or length < 1 or len(context) + length > self.max_model_len:
            raise ValueError(
                f"Sampling context ({len(context)} tokens) plus continuation ({length} tokens) exceeds vLLM context length ({self.max_model_len})"
            )

    def _params(self, max_tokens: int, *, proposal: bool, **kwargs):
        return self.sampling_params(
            max_tokens=max_tokens,
            temperature=self.config.temperature if proposal else 1.0,
            repetition_penalty=self.config.repetition_penalty if proposal else 1.0,
            top_k=-1,
            top_p=1.0,
            min_p=0.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            ignore_eos=True,
            detokenize=False,
            **kwargs,
        )

    def sample(self, context: list[int], max_tokens: int) -> list[int]:
        self._check_context(context, max_tokens)
        seed = (self.seed + self.request_number) % (2**32)
        self.request_number += 1
        params = self._params(
            max_tokens,
            proposal=True,
            seed=seed,
            stop_token_ids=sorted(self.eos_token_ids),
        )
        outputs = self.engine.generate(
            [{"prompt_token_ids": context}],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("vLLM returned an unexpected number of completions")
        return list(outputs[0].outputs[0].token_ids)

    @staticmethod
    def _logprob(row, token: int) -> float:
        if row is None or token not in row:
            raise ValueError(
                f"vLLM did not return the requested token log probability for {token}"
            )
        value = float(row[token].logprob)
        if not math.isfinite(value):
            raise ValueError("vLLM returned a non-finite token log probability")
        return value

    def target_logprob(self, context: list[int], tokens: list[int]) -> float:
        if not tokens:
            return 0.0
        self._check_context(context, len(tokens))
        if len(context) + len(tokens) == self.max_model_len:
            # Prompt scoring generates an extra token; score the boundary token separately.
            return self.target_logprob(
                context, tokens[:-1]
            ) + self._next_token_logprobs(
                [context + tokens[:-1]], [tokens[-1]], proposal=False
            )
        params = self._params(1, proposal=False, prompt_logprobs=0, seed=self.seed)
        outputs = self.engine.generate(
            [{"prompt_token_ids": context + tokens}],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise ValueError("vLLM returned an unexpected number of prompt scores")
        rows = outputs[0].prompt_logprobs
        if rows is None or len(rows) != len(context) + len(tokens):
            raise ValueError(
                "vLLM returned missing or misaligned prompt log probabilities"
            )
        return sum(
            self._logprob(row, token)
            for row, token in zip(rows[len(context) :], tokens, strict=True)
        )

    def proposal_logprob(self, context: list[int], tokens: list[int]) -> float:
        if not tokens:
            return 0.0
        self._check_context(context, len(tokens))
        if self.config.temperature == 1.0 and self.config.repetition_penalty == 1.0:
            return self.target_logprob(context, tokens)
        total = 0.0
        for start in range(0, len(tokens), self.options.score_batch_size):
            end = min(start + self.options.score_batch_size, len(tokens))
            prefixes = [context + tokens[:position] for position in range(start, end)]
            total += self._next_token_logprobs(
                prefixes, tokens[start:end], proposal=True
            )
        return total

    def _next_token_logprobs(
        self, contexts: list[list[int]], tokens: list[int], *, proposal: bool
    ) -> float:
        params = []
        for token in tokens:
            score_kwargs: dict[str, Any] = (
                {"logprob_token_ids": [token]}
                if self.specific_logprobs
                else {"logprobs": -1}
            )
            params.append(
                self._params(1, proposal=proposal, seed=self.seed, **score_kwargs)
            )
        outputs = self.engine.generate(
            [{"prompt_token_ids": context} for context in contexts],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != len(tokens):
            raise ValueError("vLLM returned an unexpected number of token scores")
        total = 0.0
        for output, token in zip(outputs, tokens, strict=True):
            if len(output.outputs) != 1:
                raise ValueError("vLLM returned an unexpected number of completions")
            rows = output.outputs[0].logprobs
            if rows is None or len(rows) != 1:
                raise ValueError(
                    "vLLM returned missing or misaligned output log probabilities"
                )
            total += self._logprob(rows[0], token)
        return total

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            try:
                engine.llm_engine.engine_core.shutdown()
            finally:
                del engine
                gc.collect()
