"""Collate tokenized chat examples into encoder/canvas diffusion tensors."""

from __future__ import annotations

from collections.abc import Sequence, Sized

import torch

from .batch import DiffusionBatch
from .sampling import native_packing_lengths


class DiffusionCollator:
    """Build one selected supervised response block for each logical example."""

    def __init__(
        self,
        pad_token_id: int,
        canvas_width: int | None = None,
        layout: str = "encoder_canvas",
        *,
        logical_sequence_length: int | None = None,
        physical_pack_budget: int | None = None,
        eos_tail: str | None = None,
        eos_token_id: int | None = None,
        overflow_policy: str = "error",
    ):
        self.pad_token_id = pad_token_id
        self.canvas_width = 256 if canvas_width is None else canvas_width
        if layout not in {"encoder_canvas", "full_sequence"}:
            raise ValueError("layout must be encoder_canvas or full_sequence")
        if logical_sequence_length is not None and logical_sequence_length <= 0:
            raise ValueError("logical_sequence_length must be positive")
        if physical_pack_budget is not None and physical_pack_budget <= 0:
            raise ValueError("physical_pack_budget must be positive")
        if eos_tail not in {None, "none", "visible_supervised"}:
            raise ValueError("eos_tail must be none or visible_supervised")
        if overflow_policy not in {"error", "drop"}:
            raise ValueError("overflow_policy must be error or drop")
        self.layout = layout
        self.logical_sequence_length = logical_sequence_length
        self.physical_pack_budget = physical_pack_budget
        self.eos_tail = eos_tail
        self.eos_token_id = pad_token_id if eos_token_id is None else eos_token_id
        self.overflow_policy = overflow_policy

    def __call__(
        self, features: Sequence[dict[str, object] | Sequence[dict[str, object]]]
    ) -> dict[str, torch.Tensor]:
        return self.build_batch(features).__dict__

    def build_batch(
        self, features: Sequence[dict[str, object] | Sequence[dict[str, object]]]
    ) -> DiffusionBatch:
        features = self._flatten_packed_features(features)
        if not features:
            raise ValueError("DiffusionCollator requires at least one feature")
        features = self._apply_budgets(features)
        ids = [
            torch.as_tensor(feature["input_ids"], dtype=torch.long)
            for feature in features
        ]
        labels = [
            torch.as_tensor(feature["labels"], dtype=torch.long) for feature in features
        ]
        if any(item.ndim != 1 for item in ids) or any(
            item.shape != label.shape for item, label in zip(ids, labels, strict=True)
        ):
            raise ValueError("input_ids and labels must be matching rank-one tensors")
        if self.eos_tail == "visible_supervised" and self.logical_sequence_length:
            with_tail = [
                self._append_visible_eos_tail(input_ids, item_labels)
                for input_ids, item_labels in zip(ids, labels, strict=True)
            ]
            ids = [input_ids for input_ids, _ in with_tail]
            labels = [item_labels for _, item_labels in with_tail]
        if self.layout == "full_sequence":
            return self._build_full_sequence(ids, labels)
        prompt_width = max(item.numel() for item in ids)
        batch_size = len(ids)
        encoder_ids = torch.full(
            (batch_size, prompt_width), self.pad_token_id, dtype=torch.long
        )
        encoder_validity = torch.zeros_like(encoder_ids, dtype=torch.bool)
        canvas_ids = torch.full(
            (batch_size, self.canvas_width), self.pad_token_id, dtype=torch.long
        )
        canvas_validity = torch.zeros_like(canvas_ids, dtype=torch.bool)
        canvas_loss_mask = torch.zeros_like(canvas_validity)
        prefix_lengths = torch.zeros(batch_size, dtype=torch.long)
        canvas_lengths = torch.zeros(batch_size, dtype=torch.long)
        selected_block_ids = torch.zeros(batch_size, dtype=torch.long)
        for index, (input_ids, item_labels) in enumerate(zip(ids, labels, strict=True)):
            valid_length = input_ids.numel()
            encoder_ids[index, :valid_length] = input_ids
            encoder_validity[index, :valid_length] = True
            supervised = torch.where(item_labels != -100)[0]
            if not supervised.numel():
                raise ValueError(
                    "encoder/canvas diffusion requires a supervised response span"
                )
            first, last = supervised[0].item(), supervised[-1].item() + 1
            response = input_ids[first:last]
            response_labels = item_labels[first:last]
            block_id = 0
            if response.numel() > self.canvas_width:
                eligible_blocks = ((supervised - first) // self.canvas_width).unique()
                block_id = int(
                    eligible_blocks[torch.randint(eligible_blocks.numel(), (1,))].item()
                )
                block = slice(
                    block_id * self.canvas_width, (block_id + 1) * self.canvas_width
                )
                response = response[block]
                response_labels = response_labels[block]
            canvas_loss_mask[index, : response.numel()] = response_labels != -100
            canvas_ids[index, : response.numel()] = response
            canvas_validity[index, : response.numel()] = True
            prefix_lengths[index] = first + block_id * self.canvas_width
            canvas_lengths[index] = response.numel()
            selected_block_ids[index] = block_id
        logical_ids = torch.arange(batch_size, dtype=torch.long)
        encoder_positions = torch.arange(prompt_width).expand(batch_size, -1).clone()
        encoder_document_ids = torch.full_like(encoder_ids, -1)
        encoder_document_ids[encoder_validity] = logical_ids[:, None].expand(
            -1, prompt_width
        )[encoder_validity]
        return DiffusionBatch(
            encoder_input_ids=encoder_ids,
            encoder_validity=encoder_validity,
            encoder_ar_valid_mask=encoder_validity.clone(),
            encoder_document_ids=encoder_document_ids,
            encoder_position_ids=encoder_positions,
            canvas_clean_ids=canvas_ids,
            canvas_semantic_validity=canvas_validity,
            canvas_loss_mask=canvas_loss_mask,
            canvas_corruptible_mask=canvas_loss_mask.clone(),
            canvas_input_pinned_mask=canvas_validity & ~canvas_loss_mask,
            canvas_sc_eligible_mask=canvas_validity.clone(),
            canvas_read_only_mask=torch.zeros_like(canvas_validity),
            canvas_update_mask=canvas_loss_mask.clone(),
            logical_ids=logical_ids,
            encoder_lengths=encoder_validity.sum(-1),
            canvas_lengths=canvas_lengths,
            decoder_prefix_lengths=prefix_lengths,
            selected_block_ids=selected_block_ids,
        )

    def _flatten_packed_features(
        self, features: Sequence[dict[str, object] | Sequence[dict[str, object]]]
    ) -> list[dict[str, object]]:
        flattened: list[dict[str, object]] = []
        for feature in features:
            if isinstance(feature, dict):
                flattened.append(feature)
            else:
                flattened.extend(feature)
        return flattened

    def _apply_budgets(
        self, features: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        kept: list[dict[str, object]] = []
        used = 0
        for feature in features:
            input_ids = feature["input_ids"]
            if not isinstance(input_ids, Sized):
                raise TypeError("diffusion input_ids must have a length")
            length = len(input_ids)
            exceeds_logical = (
                self.logical_sequence_length is not None
                and length > self.logical_sequence_length
            )
            if exceeds_logical:
                if self.overflow_policy == "drop":
                    continue
                raise ValueError(
                    f"diffusion logical example exceeds packed token budget {self.logical_sequence_length}"
                )
            effective_length = native_packing_lengths(
                [feature],
                eos_tail=self.eos_tail,
                logical_sequence_length=self.logical_sequence_length,
                layout=self.layout,
                canvas_width=self.canvas_width,
            )[0]
            exceeds_physical = (
                self.physical_pack_budget is not None
                and effective_length > self.physical_pack_budget
            )
            if exceeds_logical or exceeds_physical:
                if self.overflow_policy == "error":
                    limit = (
                        self.logical_sequence_length
                        if exceeds_logical
                        else self.physical_pack_budget
                    )
                    raise ValueError(
                        f"diffusion logical example exceeds packed token budget {limit}"
                    )
                continue
            if (
                self.physical_pack_budget is not None
                and used + effective_length > self.physical_pack_budget
            ):
                raise ValueError(
                    "diffusion batch exceeds packed token budget; "
                    "the sampler must use native diffusion packing costs"
                )
            kept.append(feature)
            used += effective_length
        if not kept:
            raise ValueError(
                "all diffusion examples exceeded the configured token budget"
            )
        return kept

    def _append_visible_eos_tail(
        self, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_length = self.logical_sequence_length
        if target_length is None or input_ids.numel() >= target_length:
            return input_ids, labels
        tail_length = target_length - input_ids.numel()
        tail_ids = torch.full((tail_length,), self.pad_token_id, dtype=input_ids.dtype)
        tail_ids[0] = self.eos_token_id
        return (
            torch.cat((input_ids, tail_ids)),
            torch.cat((labels, tail_ids.to(labels.dtype))),
        )

    def _build_full_sequence(
        self, ids: list[torch.Tensor], labels: list[torch.Tensor]
    ) -> DiffusionBatch:
        batch_size = len(ids)
        width = max(item.numel() for item in ids)
        canvas_ids = torch.full(
            (batch_size, width), self.pad_token_id, dtype=torch.long
        )
        semantic_validity = torch.zeros_like(canvas_ids, dtype=torch.bool)
        loss_mask = torch.zeros_like(canvas_ids, dtype=torch.bool)
        for index, (input_ids, item_labels) in enumerate(zip(ids, labels, strict=True)):
            length = input_ids.numel()
            canvas_ids[index, :length] = input_ids
            semantic_validity[index, :length] = True
            loss_mask[index, :length] = item_labels != -100
        pinned = semantic_validity & ~loss_mask
        empty_encoder = torch.empty((batch_size, 0), dtype=torch.long)
        empty_validity = torch.empty((batch_size, 0), dtype=torch.bool)
        logical_ids = torch.arange(batch_size, dtype=torch.long)
        return DiffusionBatch(
            encoder_input_ids=empty_encoder,
            encoder_validity=empty_validity,
            encoder_ar_valid_mask=empty_validity.clone(),
            encoder_document_ids=empty_encoder.clone(),
            encoder_position_ids=empty_encoder.clone(),
            canvas_clean_ids=canvas_ids,
            canvas_semantic_validity=semantic_validity,
            canvas_loss_mask=loss_mask,
            canvas_corruptible_mask=loss_mask.clone(),
            canvas_input_pinned_mask=pinned,
            canvas_sc_eligible_mask=loss_mask.clone(),
            canvas_read_only_mask=pinned.clone(),
            canvas_update_mask=loss_mask.clone(),
            logical_ids=logical_ids,
            encoder_lengths=torch.zeros(batch_size, dtype=torch.long),
            canvas_lengths=semantic_validity.sum(-1),
            decoder_prefix_lengths=torch.zeros(batch_size, dtype=torch.long),
            selected_block_ids=torch.zeros(batch_size, dtype=torch.long),
        )
