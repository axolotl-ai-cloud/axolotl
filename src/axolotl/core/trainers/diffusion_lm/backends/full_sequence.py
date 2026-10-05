"""Full-sequence diffusion backend."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion", "diffusion_lm")

import torch


def create_bidirectional_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    sample_packing: bool = False,
) -> torch.Tensor:
    del sample_packing
    batch_size, sequence_length = input_ids.shape
    if attention_mask is None:
        return torch.ones(
            batch_size,
            1,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=input_ids.device,
        )
    mask_i = attention_mask.unsqueeze(2)
    mask_j = attention_mask.unsqueeze(1)
    return ((mask_i == mask_j) & (mask_i > 0)).unsqueeze(1)


def shift_logits_to_input_positions(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) <= 1:
        return logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


def create_packed_document_attention_mask(
    document_ids: torch.Tensor, semantic_validity: torch.Tensor
) -> torch.Tensor:
    """Bidirectional attention constrained to valid positions in one document."""

    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    valid = semantic_validity[:, :, None] & semantic_validity[:, None, :]
    return (same_document & valid & (document_ids[:, :, None] >= 0)).unsqueeze(1)


def reset_position_ids(
    document_ids: torch.Tensor, semantic_validity: torch.Tensor
) -> torch.Tensor:
    """Return RoPE positions reset at each logical document boundary."""

    valid = semantic_validity & (document_ids >= 0)
    starts = torch.ones_like(valid)
    starts[:, 1:] = (
        (document_ids[:, 1:] != document_ids[:, :-1]) | ~valid[:, :-1] | ~valid[:, 1:]
    )
    columns = torch.arange(
        document_ids.shape[1], device=document_ids.device, dtype=document_ids.dtype
    ).expand_as(document_ids)
    start_columns = torch.where(starts, columns, 0).cummax(dim=1).values
    return torch.where(valid, columns - start_columns, 0)


def shift_logits_within_documents(
    logits: torch.Tensor, document_ids: torch.Tensor, semantic_validity: torch.Tensor
) -> torch.Tensor:
    """Dream's duplicate-first shift, reset at every packed document boundary."""

    shifted = logits.clone()
    shifted[:, 1:] = logits[:, :-1]
    boundary = torch.ones_like(semantic_validity, dtype=torch.bool)
    boundary[:, 1:] = document_ids[:, 1:] != document_ids[:, :-1]
    boundary |= ~semantic_validity
    shifted[boundary] = logits[boundary]
    return shifted


def _require_contiguous_document_runs(
    document_ids: torch.Tensor, semantic_validity: torch.Tensor
) -> None:
    valid = semantic_validity & (document_ids >= 0)
    positions = torch.arange(
        document_ids.shape[1], device=document_ids.device
    ).expand_as(document_ids)
    sorted_ids, order = torch.where(valid, document_ids, -1).sort(dim=1, stable=True)
    sorted_valid = valid.gather(1, order)
    sorted_positions = positions.gather(1, order)
    repeated = (
        sorted_valid[:, 1:]
        & sorted_valid[:, :-1]
        & (sorted_ids[:, 1:] == sorted_ids[:, :-1])
    )
    adjacent = (sorted_positions[:, 1:] - sorted_positions[:, :-1]).abs() == 1
    torch._assert_async(
        (~repeated | adjacent).all(),
        "a document ID must occupy one contiguous valid run within each row",
    )


def corrupt_packed_absorbing(
    input_ids: torch.Tensor,
    *,
    document_ids: torch.Tensor,
    semantic_validity: torch.Tensor,
    corruptible_mask: torch.Tensor,
    logical_times: torch.Tensor,
    document_time_indices: torch.Tensor | None = None,
    mask_token_id: int,
    eos_token_id: int | None = None,
    treat_eos_as_one: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-logical-time absorbing corruption for a physically packed row."""

    probabilities = torch.zeros_like(input_ids, dtype=torch.float)
    time_indices = (
        document_ids if document_time_indices is None else document_time_indices
    )
    valid_docs = (document_ids >= 0) & (time_indices >= 0)
    probabilities[valid_docs] = logical_times[time_indices[valid_docs]]
    events = (
        torch.rand(input_ids.shape, device=input_ids.device, generator=generator)
        < probabilities
    )
    events &= semantic_validity & corruptible_mask
    if treat_eos_as_one and eos_token_id is not None:
        _require_contiguous_document_runs(document_ids, semantic_validity)
        valid = semantic_validity & (document_ids >= 0)
        run_starts = valid.clone()
        run_starts[:, 1:] &= (~valid[:, :-1]) | (
            document_ids[:, 1:] != document_ids[:, :-1]
        )
        run_ends = valid.clone()
        run_ends[:, :-1] &= (~valid[:, 1:]) | (
            document_ids[:, :-1] != document_ids[:, 1:]
        )
        tail_eligible = valid & corruptible_mask & (input_ids == eos_token_id)
        reverse_eligible = tail_eligible.flip(1)
        reverse_starts = run_ends.flip(1)
        failures = (~reverse_eligible).to(torch.long)
        cumulative_failures = failures.cumsum(dim=1)
        before_run = (
            torch.where(
                reverse_starts,
                cumulative_failures - failures,
                torch.full_like(cumulative_failures, -1),
            )
            .cummax(dim=1)
            .values
        )
        trailing = (reverse_eligible & ((cumulative_failures - before_run) == 0)).flip(
            1
        )
        trailing_starts = trailing.clone()
        trailing_starts[:, 1:] &= (~trailing[:, :-1]) | run_starts[:, 1:]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(
            input_ids
        )
        sources = (
            torch.where(trailing_starts, positions, torch.full_like(positions, -1))
            .cummax(dim=1)
            .values
        )
        events = torch.where(
            trailing,
            events.gather(1, sources.clamp_min(0)),
            events,
        )
    events &= semantic_validity & corruptible_mask
    return (
        torch.where(events, torch.full_like(input_ids, mask_token_id), input_ids),
        events,
        probabilities,
    )


class FullSequenceBackend:
    """Legacy full-sequence corruption and alignment behavior."""

    def __init__(
        self,
        *,
        mask_token_id: int,
        special_token_ids: set[int] | None = None,
        sample_packing: bool = False,
        attention_backend: str = "dense",
        physical_bucket_size: int | None = None,
    ):
        self.mask_token_id = mask_token_id
        self.special_token_ids = special_token_ids or set()
        self.sample_packing = sample_packing
        if attention_backend not in {"dense", "flex_attention", "varlen"}:
            raise ValueError(
                "attention_backend must be 'dense', 'flex_attention', or 'varlen'"
            )
        self.attention_backend = attention_backend
        if physical_bucket_size is not None and physical_bucket_size <= 0:
            raise ValueError("physical_bucket_size must be positive")
        self.physical_bucket_size = (
            128
            if attention_backend == "flex_attention" and physical_bucket_size is None
            else physical_bucket_size
        )

    def corrupt(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = input_ids.shape
        t = torch.rand(batch_size, device=input_ids.device)
        p_mask = ((1 - eps) * t + eps)[:, None].repeat(1, sequence_length)
        if attention_mask is not None:
            p_mask = p_mask * attention_mask.bool().float()

        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids:
            special_token_mask |= input_ids == token_id

        masked_indices = (
            torch.rand((batch_size, sequence_length), device=input_ids.device) < p_mask
        )
        masked_indices = masked_indices & ~special_token_mask
        if attention_mask is not None:
            masked_indices = masked_indices & attention_mask.bool()
        if labels is not None:
            masked_indices = masked_indices & (labels != -100)

        noisy_batch = torch.where(
            masked_indices, torch.full_like(input_ids, self.mask_token_id), input_ids
        )
        return noisy_batch, masked_indices, p_mask

    def attention_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return create_bidirectional_attention_mask(
            input_ids,
            attention_mask,
            sample_packing=self.sample_packing,
        )

    def align_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return shift_logits_to_input_positions(logits)

    def pack(
        self,
        input_ids: torch.Tensor,
        document_ids: torch.Tensor,
        semantic_validity: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the native full-sequence physical pack without legacy mutation."""

        if input_ids.ndim != 2:
            raise ValueError("native full-sequence input_ids must be [rows, tokens]")
        if (
            document_ids.shape != input_ids.shape
            or semantic_validity.shape != input_ids.shape
        ):
            raise ValueError(
                "native full-sequence document and validity masks must match input_ids"
            )
        if position_ids is None:
            position_ids = reset_position_ids(document_ids, semantic_validity)
        if position_ids.shape != input_ids.shape:
            raise ValueError("native full-sequence position_ids must match input_ids")
        if self.physical_bucket_size is not None:
            length = input_ids.shape[1]
            bucket = (
                (length + self.physical_bucket_size - 1)
                // self.physical_bucket_size
                * self.physical_bucket_size
            )
            if bucket != length:
                input_ids = torch.nn.functional.pad(input_ids, (0, bucket - length))
                document_ids = torch.nn.functional.pad(
                    document_ids, (0, bucket - length), value=-1
                )
                semantic_validity = torch.nn.functional.pad(
                    semantic_validity, (0, bucket - length), value=False
                )
                position_ids = torch.nn.functional.pad(
                    position_ids, (0, bucket - length)
                )
        varlen_metadata = None
        if self.attention_backend == "varlen":
            from ..varlen import build_varlen_metadata

            varlen_metadata = build_varlen_metadata(document_ids, semantic_validity)
            attention_mask = None
        elif self.attention_backend == "flex_attention":
            from ..attention import full_sequence_flex_block_mask

            attention_mask = full_sequence_flex_block_mask(
                document_ids, semantic_validity, position_ids
            )
        else:
            attention_mask = create_packed_document_attention_mask(
                document_ids, semantic_validity
            )
        return {
            "input_ids": input_ids,
            "document_ids": document_ids,
            "semantic_validity": semantic_validity,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "diffusion_varlen": varlen_metadata,
        }

    def corrupt_native(
        self,
        packed: dict[str, torch.Tensor],
        corruptible_mask: torch.Tensor,
        logical_times: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return corrupt_packed_absorbing(
            packed["input_ids"],
            document_ids=packed["document_ids"],
            semantic_validity=packed["semantic_validity"],
            corruptible_mask=corruptible_mask,
            logical_times=logical_times,
            mask_token_id=self.mask_token_id,
            **kwargs,
        )

    def forward(
        self,
        model,
        packed: dict[str, torch.Tensor],
        noisy_input_ids: torch.Tensor,
        *,
        kernel_options: dict | None = None,
        model_kwargs: dict | None = None,
    ):
        kwargs = {
            "input_ids": noisy_input_ids,
            "attention_mask": packed["attention_mask"],
            "position_ids": packed["position_ids"],
            "use_cache": False,
        }
        if kernel_options is not None:
            kwargs["kernel_options"] = kernel_options
        if model_kwargs is not None:
            if not isinstance(model_kwargs, dict):
                raise TypeError("model_kwargs must be a dict")
            reserved = {
                "attention_mask",
                "cache_position",
                "diffusion_varlen",
                "input_ids",
                "kernel_options",
                "labels",
                "past_key_values",
                "position_ids",
                "use_cache",
                "use_causal_mask",
            }
            conflicts = reserved.intersection(model_kwargs)
            if conflicts:
                raise ValueError(
                    "model_kwargs cannot override backend routing keys: "
                    + ", ".join(sorted(conflicts))
                )
            kwargs.update(model_kwargs)
        if self.attention_backend == "varlen":
            capability_model = model
            while hasattr(capability_model, "module"):
                capability_model = capability_model.module
            if not getattr(capability_model, "supports_diffusion_varlen", False):
                raise ValueError(
                    "This model does not support diffusion varlen attention"
                )
            kwargs["diffusion_varlen"] = packed["diffusion_varlen"]
        return model(**kwargs)

    def canvas_logits(
        self, outputs, packed: dict[str, torch.Tensor], *, aligned: bool = False
    ) -> torch.Tensor:
        if aligned:
            return outputs.logits
        return shift_logits_within_documents(
            outputs.logits, packed["document_ids"], packed["semantic_validity"]
        )

    @staticmethod
    def update(
        input_ids: torch.Tensor, logits: torch.Tensor, update_mask: torch.Tensor
    ) -> torch.Tensor:
        """Commit discrete predictions only at positions selected by the loop."""

        if input_ids.shape != update_mask.shape or logits.shape[:2] != input_ids.shape:
            raise ValueError("state, logits, and update_mask must share sequence shape")
        return torch.where(update_mask, logits.detach().argmax(-1), input_ids)
