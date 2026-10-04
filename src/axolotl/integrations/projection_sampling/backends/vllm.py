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
        eos = getattr(model_config.hf_config, "eos_token_id", None)
        if eos is None:
            eos = getattr(
                getattr(model_config.hf_config, "text_config", None),
                "eos_token_id",
                None,
            )
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

    def sample_batch(
        self, contexts: list[list[int]], max_tokens: list[int]
    ) -> list[list[int]]:
        if len(contexts) != len(max_tokens):
            raise ValueError("Sampling batch contexts and budgets must align")
        if len(contexts) == 1:
            return [self.sample(contexts[0], max_tokens[0])]
        if not contexts:
            return []
        seeds = [
            (self.seed + self.request_number + index) % (2**32)
            for index in range(len(contexts))
        ]
        return self.sample_batch_seeded(contexts, max_tokens, seeds)

    def sample_batch_seeded(self, contexts, max_tokens, seeds):
        if len(contexts) != len(max_tokens) or len(contexts) != len(seeds):
            raise ValueError("Sampling batch contexts, budgets and seeds must align")
        if not contexts:
            return []
        for context, budget in zip(contexts, max_tokens, strict=True):
            self._check_context(context, budget)
        params = []
        for budget, seed in zip(max_tokens, seeds, strict=True):
            params.append(
                self._params(
                    budget,
                    proposal=True,
                    seed=seed % (2**32),
                    stop_token_ids=sorted(self.eos_token_ids),
                )
            )
        self.request_number += len(contexts)
        outputs = self.engine.generate(
            [{"prompt_token_ids": context} for context in contexts],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != len(contexts) or any(
            len(output.outputs) != 1 for output in outputs
        ):
            raise ValueError("vLLM returned a misaligned completion batch")
        return [list(output.outputs[0].token_ids) for output in outputs]

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

    def target_logprob_batch(
        self, contexts: list[list[int]], tokens: list[list[int]]
    ) -> list[float]:
        if len(contexts) != len(tokens):
            raise ValueError("Scoring batch contexts and continuations must align")
        result = [0.0] * len(contexts)
        pending = []
        for index, (context, continuation) in enumerate(
            zip(contexts, tokens, strict=True)
        ):
            if not continuation:
                continue
            self._check_context(context, len(continuation))
            if len(context) + len(continuation) == self.max_model_len:
                result[index] = self.target_logprob(context, continuation)
            else:
                pending.append(index)
        if not pending:
            return result
        params = self._params(1, proposal=False, prompt_logprobs=0, seed=self.seed)
        outputs = self.engine.generate(
            [
                {"prompt_token_ids": contexts[index] + tokens[index]}
                for index in pending
            ],
            sampling_params=params,
            use_tqdm=False,
        )
        if len(outputs) != len(pending):
            raise ValueError("vLLM returned a misaligned prompt-score batch")
        for index, output in zip(pending, outputs, strict=True):
            rows = output.prompt_logprobs
            context, continuation = contexts[index], tokens[index]
            if rows is None or len(rows) != len(context) + len(continuation):
                raise ValueError(
                    "vLLM returned missing or misaligned prompt log probabilities"
                )
            result[index] = sum(
                self._logprob(row, token)
                for row, token in zip(rows[len(context) :], continuation, strict=True)
            )
        return result

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

    def proposal_logprob_batch(
        self, contexts: list[list[int]], tokens: list[list[int]]
    ) -> list[float]:
        if len(contexts) != len(tokens):
            raise ValueError("Scoring batch contexts and continuations must align")
        if self.config.temperature == 1.0 and self.config.repetition_penalty == 1.0:
            return self.target_logprob_batch(contexts, tokens)
        for context, continuation in zip(contexts, tokens, strict=True):
            if continuation:
                self._check_context(context, len(continuation))
        result = [0.0] * len(contexts)
        prefixes: list[list[int]] = []
        wanted: list[int] = []
        owners: list[int] = []

        def flush():
            scores = self._token_logprobs(prefixes, wanted, proposal=True)
            for owner, score in zip(owners, scores, strict=True):
                result[owner] += score
            prefixes.clear()
            wanted.clear()
            owners.clear()

        for position in range(max((len(row) for row in tokens), default=0)):
            for owner, (context, continuation) in enumerate(
                zip(contexts, tokens, strict=True)
            ):
                if position < len(continuation):
                    prefixes.append(context + continuation[:position])
                    wanted.append(continuation[position])
                    owners.append(owner)
                    if len(prefixes) == self.options.score_batch_size:
                        flush()
        if prefixes:
            flush()
        return result

    def _next_token_logprobs(
        self, contexts: list[list[int]], tokens: list[int], *, proposal: bool
    ) -> float:
        return sum(self._token_logprobs(contexts, tokens, proposal=proposal), 0.0)

    def _token_logprobs(
        self, contexts: list[list[int]], tokens: list[int], *, proposal: bool
    ) -> list[float]:
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
        result = []
        for output, token in zip(outputs, tokens, strict=True):
            if len(output.outputs) != 1:
                raise ValueError("vLLM returned an unexpected number of completions")
            rows = output.outputs[0].logprobs
            if rows is None or len(rows) != 1:
                raise ValueError(
                    "vLLM returned missing or misaligned output log probabilities"
                )
            result.append(self._logprob(rows[0], token))
        return result

    def proposal_kl(self, target_context, proposal_context, tokens, positions):
        if not positions:
            return []
        if any(position < 0 or position >= len(tokens) for position in positions):
            raise ValueError("KL positions must index continuation tokens")
        self._check_context(target_context, max(positions) + 1)
        self._check_context(proposal_context, max(positions) + 1)
        vocab_size = self.engine.llm_engine.model_config.get_vocab_size()
        result = []
        for first in range(0, len(positions), self.options.score_batch_size):
            selected = positions[first : first + self.options.score_batch_size]
            prompts = []
            params = []
            for position in selected:
                for context, proposal in (
                    (target_context, False),
                    (proposal_context, True),
                ):
                    prompts.append({"prompt_token_ids": context + tokens[:position]})
                    params.append(
                        self._params(1, proposal=proposal, seed=self.seed, logprobs=-1)
                    )
            outputs = self.engine.generate(
                prompts, sampling_params=params, use_tqdm=False
            )
            if len(outputs) != len(prompts):
                raise ValueError("vLLM returned a misaligned KL-score batch")
            distributions = []
            for output in outputs:
                if len(output.outputs) != 1:
                    raise ValueError(
                        "vLLM returned an unexpected number of completions"
                    )
                rows = output.outputs[0].logprobs
                if rows is None or len(rows) != 1 or not rows[0]:
                    raise ValueError("vLLM returned missing KL log probabilities")
                row = rows[0]
                if (
                    len(row) != vocab_size
                    or min(row) != 0
                    or max(row) != vocab_size - 1
                ):
                    raise ValueError(
                        "Proposal KL requires full-vocabulary vLLM log probabilities"
                    )
                values = [self._logprob(row, token) for token in range(vocab_size)]
                if not math.isclose(
                    math.fsum(math.exp(value) for value in values),
                    1.0,
                    rel_tol=1e-4,
                    abs_tol=1e-4,
                ):
                    raise ValueError(
                        "vLLM returned unnormalized full-vocabulary KL log probabilities"
                    )
                distributions.append(values)
            for index in range(0, len(distributions), 2):
                base, proposal = distributions[index : index + 2]
                kl = math.fsum(
                    math.exp(logq) * (logq - logp)
                    for logp, logq in zip(base, proposal, strict=True)
                )
                if not math.isfinite(kl):
                    raise ValueError("vLLM returned non-finite proposal KL")
                result.append(max(0.0, kl))
        return result

    def close(self):
        if self.engine is not None:
            engine, self.engine = self.engine, None
            try:
                engine.llm_engine.engine_core.shutdown()
            finally:
                del engine
                gc.collect()
