"""Transformers inference with matching generation and proposal densities."""

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from .args import ProjectionSamplingConfig


class TransformersBackend:
    """Unmodified base model for target and expert-conditioned proposal scoring."""

    def __init__(self, model, tokenizer, config: ProjectionSamplingConfig):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.config = config
        eos = model.generation_config.eos_token_id
        if eos is None:
            eos = tokenizer.eos_token_id
        self.eos_token_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        self.device = model.get_input_embeddings().weight.device

    @classmethod
    def from_config(cls, cfg, config: ProjectionSamplingConfig):
        from axolotl.loaders import load_tokenizer

        tokenizer = load_tokenizer(cfg)
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            revision=cfg.revision_of_model,
            trust_remote_code=bool(cfg.trust_remote_code),
            dtype=config.dtype
            if config.dtype == "auto"
            else getattr(torch, config.dtype),
            attn_implementation="eager",
        ).to(config.device)
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            raise ValueError(
                "Projection sampling does not support adding tokens to the base model vocabulary"
            )
        return cls(model, tokenizer, config)

    def _check_context(self, context: list[int], length: int):
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if not context or (limit and len(context) + length > limit):
            raise ValueError(
                f"Sampling context ({len(context)} tokens) plus continuation "
                f"({length} tokens) exceeds model context length ({limit}). "
                "Reduce max_new_tokens or shorten the expert solution."
            )

    @torch.inference_mode()
    def sample(self, context: list[int], max_tokens: int) -> list[int]:
        self._check_context(context, max_tokens)
        ids = torch.tensor([context], device=self.device)
        generation = GenerationConfig(
            do_sample=True,
            temperature=self.config.temperature,
            repetition_penalty=self.config.repetition_penalty,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_tokens,
            eos_token_id=sorted(self.eos_token_ids) or None,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else next(iter(self.eos_token_ids), None),
            bos_token_id=self.tokenizer.bos_token_id,
        )
        output = self.model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            generation_config=generation,
        )
        return output[0, len(context) :].tolist()

    @torch.inference_mode()
    def score(self, context: list[int], tokens: list[int], *, proposal: bool) -> float:
        if not tokens:
            return 0.0
        self._check_context(context, len(tokens))
        ids = torch.tensor([context + tokens], device=self.device)
        logits = (
            self.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
            )
            .logits[0, len(context) - 1 : -1]
            .float()
        )
        if proposal:
            if self.config.repetition_penalty != 1.0:
                for position, row in enumerate(logits):
                    previous = ids[0, : len(context) + position].unique()
                    values = row[previous]
                    row[previous] = torch.where(
                        values < 0,
                        values * self.config.repetition_penalty,
                        values / self.config.repetition_penalty,
                    )
            logits /= self.config.temperature
        selected = logits.log_softmax(-1).gather(-1, ids[0, len(context) :, None])
        result = selected.sum().item()
        if not torch.isfinite(selected).all():
            raise ValueError("Model returned non-finite token log probabilities")
        return result

    def close(self):
        self.model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
