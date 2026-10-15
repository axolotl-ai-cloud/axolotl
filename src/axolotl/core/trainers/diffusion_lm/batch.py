"""Typed tensors exchanged by diffusion collators and backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _require_bool(name: str, value: torch.Tensor) -> None:
    if value.dtype is not torch.bool:
        raise TypeError(f"{name} must be bool, got {value.dtype}")


def _require_shape(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {shape}")


@dataclass(frozen=True)
class DiffusionBatch:
    """Logical examples before their prompt and canvas streams are packed."""

    encoder_input_ids: torch.Tensor
    encoder_validity: torch.Tensor
    encoder_ar_valid_mask: torch.Tensor
    encoder_document_ids: torch.Tensor
    encoder_position_ids: torch.Tensor
    canvas_clean_ids: torch.Tensor
    canvas_semantic_validity: torch.Tensor
    canvas_loss_mask: torch.Tensor
    canvas_corruptible_mask: torch.Tensor
    canvas_input_pinned_mask: torch.Tensor
    canvas_sc_eligible_mask: torch.Tensor
    canvas_read_only_mask: torch.Tensor
    canvas_update_mask: torch.Tensor
    logical_ids: torch.Tensor
    encoder_lengths: torch.Tensor
    canvas_lengths: torch.Tensor
    decoder_prefix_lengths: torch.Tensor
    selected_block_ids: torch.Tensor

    def __post_init__(self) -> None:
        batch_size, prompt_width = self.encoder_input_ids.shape
        canvas_shape = self.canvas_clean_ids.shape
        if len(canvas_shape) != 2 or canvas_shape[0] != batch_size:
            raise ValueError("canvas_clean_ids must be [batch, canvas]")
        _require_shape(
            "encoder_validity", self.encoder_validity, (batch_size, prompt_width)
        )
        _require_shape(
            "encoder_ar_valid_mask",
            self.encoder_ar_valid_mask,
            (batch_size, prompt_width),
        )
        _require_bool("encoder_ar_valid_mask", self.encoder_ar_valid_mask)
        _require_shape(
            "encoder_document_ids",
            self.encoder_document_ids,
            (batch_size, prompt_width),
        )
        _require_shape(
            "encoder_position_ids",
            self.encoder_position_ids,
            (batch_size, prompt_width),
        )
        for name in (
            "canvas_semantic_validity",
            "canvas_loss_mask",
            "canvas_corruptible_mask",
            "canvas_input_pinned_mask",
            "canvas_sc_eligible_mask",
            "canvas_read_only_mask",
            "canvas_update_mask",
        ):
            value = getattr(self, name)
            _require_shape(name, value, canvas_shape)
            _require_bool(name, value)
        for name in (
            "logical_ids",
            "encoder_lengths",
            "canvas_lengths",
            "decoder_prefix_lengths",
            "selected_block_ids",
        ):
            _require_shape(name, getattr(self, name), (batch_size,))
        if torch.any(self.canvas_loss_mask & ~self.canvas_semantic_validity):
            raise ValueError("canvas_loss_mask must be semantically valid")
        if torch.any(self.canvas_corruptible_mask & ~self.canvas_semantic_validity):
            raise ValueError("canvas_corruptible_mask must be semantically valid")
        if torch.any(self.canvas_corruptible_mask & self.canvas_input_pinned_mask):
            raise ValueError("input-pinned canvas tokens cannot be corrupted")
        if torch.any(self.canvas_update_mask & self.canvas_input_pinned_mask):
            raise ValueError("input-pinned canvas tokens cannot be updated")
        expected_encoder_validity = (
            torch.arange(prompt_width, device=self.device)[None]
            < self.encoder_lengths[:, None]
        )
        expected_canvas_validity = (
            torch.arange(canvas_shape[1], device=self.device)[None]
            < self.canvas_lengths[:, None]
        )
        if not torch.equal(self.encoder_validity, expected_encoder_validity):
            raise ValueError("encoder_validity must be a left-aligned logical document")
        if not torch.equal(self.canvas_semantic_validity, expected_canvas_validity):
            raise ValueError("canvas_semantic_validity must be a left-aligned canvas")
        expected_docs = self.logical_ids[:, None].expand(-1, prompt_width)
        if not torch.equal(
            self.encoder_document_ids[self.encoder_validity],
            expected_docs[self.encoder_validity],
        ):
            raise ValueError("encoder document IDs must match their logical example")

    @property
    def device(self) -> torch.device:
        return self.encoder_input_ids.device

    def to(self, *args, **kwargs) -> DiffusionBatch:
        return DiffusionBatch(
            **{
                name: getattr(self, name).to(*args, **kwargs)
                for name in self.__dataclass_fields__
            }
        )


@dataclass(frozen=True)
class PackedEncoderCanvasBatch:
    """One physical packed row with logical segment boundaries retained."""

    batch: DiffusionBatch
    encoder_input_ids: torch.Tensor
    encoder_validity: torch.Tensor
    encoder_ar_valid_mask: torch.Tensor
    encoder_document_ids: torch.Tensor
    encoder_position_ids: torch.Tensor
    canvas_clean_ids: torch.Tensor
    canvas_semantic_validity: torch.Tensor
    canvas_loss_mask: torch.Tensor
    canvas_corruptible_mask: torch.Tensor
    canvas_input_pinned_mask: torch.Tensor
    canvas_sc_eligible_mask: torch.Tensor
    canvas_read_only_mask: torch.Tensor
    canvas_update_mask: torch.Tensor
    canvas_document_ids: torch.Tensor
    canvas_logical_row_indices: torch.Tensor
    canvas_position_ids: torch.Tensor
    prompt_offsets: torch.Tensor
    canvas_offsets: torch.Tensor
    encoder_attention_mask: dict[str, torch.Tensor]
    decoder_attention_mask: dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if self.encoder_input_ids.ndim != 2 or self.encoder_input_ids.shape[0] != 1:
            raise ValueError("packed encoder_input_ids must be [1, prompt_length]")
        if self.canvas_clean_ids.ndim != 2 or self.canvas_clean_ids.shape[0] != 1:
            raise ValueError("packed canvas_clean_ids must be [1, canvas_length]")
        prompt_shape = self.encoder_input_ids.shape
        canvas_shape = self.canvas_clean_ids.shape
        for name in (
            "encoder_validity",
            "encoder_ar_valid_mask",
            "encoder_document_ids",
            "encoder_position_ids",
        ):
            value = getattr(self, name)
            _require_shape(name, value, prompt_shape)
            if name in {"encoder_validity", "encoder_ar_valid_mask"}:
                _require_bool(name, value)
        for name in (
            "canvas_semantic_validity",
            "canvas_loss_mask",
            "canvas_corruptible_mask",
            "canvas_input_pinned_mask",
            "canvas_sc_eligible_mask",
            "canvas_read_only_mask",
            "canvas_update_mask",
        ):
            value = getattr(self, name)
            _require_shape(name, value, canvas_shape)
            _require_bool(name, value)
        _require_shape("canvas_document_ids", self.canvas_document_ids, canvas_shape)
        _require_shape(
            "canvas_logical_row_indices", self.canvas_logical_row_indices, canvas_shape
        )
        _require_shape("canvas_position_ids", self.canvas_position_ids, canvas_shape)
        count = self.batch.logical_ids.numel()
        _require_shape("prompt_offsets", self.prompt_offsets, (count + 1,))
        _require_shape("canvas_offsets", self.canvas_offsets, (count + 1,))
        for masks, expected in (
            (self.encoder_attention_mask, (1, 1, prompt_shape[1], prompt_shape[1])),
            (
                self.decoder_attention_mask,
                (1, 1, canvas_shape[1], prompt_shape[1] + canvas_shape[1]),
            ),
        ):
            if not masks:
                raise ValueError("attention mask mapping cannot be empty")
            for mask in masks.values():
                if isinstance(mask, torch.Tensor):
                    _require_shape("attention mask", mask, expected)
                    _require_bool("attention mask", mask)

    @property
    def device(self) -> torch.device:
        return self.encoder_input_ids.device

    def to(self, *args, **kwargs) -> PackedEncoderCanvasBatch:
        return PackedEncoderCanvasBatch(
            batch=self.batch.to(*args, **kwargs),
            **{
                name: (
                    {
                        key: value.to(*args, **kwargs)
                        for key, value in getattr(self, name).items()
                    }
                    if name.endswith("attention_mask")
                    else getattr(self, name).to(*args, **kwargs)
                )
                for name in self.__dataclass_fields__
                if name != "batch"
            },
        )


@dataclass(frozen=True)
class CorruptedCanvas:
    """Corruption output, retaining event probabilities independently of IDs."""

    input_ids: torch.Tensor
    replacement_event_mask: torch.Tensor
    replacement_probabilities: torch.Tensor

    def __post_init__(self) -> None:
        _require_shape(
            "replacement_event_mask", self.replacement_event_mask, self.input_ids.shape
        )
        _require_shape(
            "replacement_probabilities",
            self.replacement_probabilities,
            self.input_ids.shape,
        )
        _require_bool("replacement_event_mask", self.replacement_event_mask)
