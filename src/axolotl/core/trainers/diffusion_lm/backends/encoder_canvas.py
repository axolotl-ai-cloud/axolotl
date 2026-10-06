"""Dense reference backend for DiffusionGemma encoder/canvas batches."""

from __future__ import annotations

import torch

from ..attention import (
    decoder_flex_block_mask,
    decoder_prefix_canvas_mask,
    encoder_causal_mask,
    encoder_flex_block_mask,
)
from ..batch import CorruptedCanvas, DiffusionBatch, PackedEncoderCanvasBatch


class EncoderCanvasBackend:
    """Pack logical streams once and retain document isolation in dense masks."""

    def __init__(
        self,
        vocab_size: int,
        sliding_window: int,
        attention_backend: str = "dense",
        physical_bucket_size: int | None = None,
    ):
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.vocab_size = vocab_size
        self.sliding_window = sliding_window
        if attention_backend not in {"dense", "flex_attention"}:
            raise ValueError("attention_backend must be dense or flex_attention")
        if physical_bucket_size is not None and physical_bucket_size <= 0:
            raise ValueError("physical_bucket_size must be positive")
        self.attention_backend = attention_backend
        self.physical_bucket_size = (
            128
            if attention_backend == "flex_attention" and physical_bucket_size is None
            else physical_bucket_size
        )

    @staticmethod
    def _bucket_pad(value: torch.Tensor, length: int, fill: int | bool) -> torch.Tensor:
        if value.shape[1] >= length:
            return value
        return torch.nn.functional.pad(value, (0, length - value.shape[1]), value=fill)

    def _bucket_length(self, length: int) -> int:
        if self.physical_bucket_size is None:
            return length
        return (
            (length + self.physical_bucket_size - 1) // self.physical_bucket_size
        ) * self.physical_bucket_size

    def pack(self, batch: DiffusionBatch) -> PackedEncoderCanvasBatch:
        prompt_offsets = torch.cat(
            (
                torch.zeros(1, device=batch.device, dtype=torch.long),
                batch.encoder_lengths.cumsum(0),
            )
        )
        canvas_offsets = torch.cat(
            (
                torch.zeros(1, device=batch.device, dtype=torch.long),
                batch.canvas_lengths.cumsum(0),
            )
        )
        encoder_ids = torch.cat(
            [
                batch.encoder_input_ids[i, :length]
                for i, length in enumerate(batch.encoder_lengths.tolist())
            ]
        )[None]
        encoder_ar = torch.cat(
            [
                batch.encoder_ar_valid_mask[i, :length]
                for i, length in enumerate(batch.encoder_lengths.tolist())
            ]
        )[None]
        encoder_docs = torch.cat(
            [
                torch.full(
                    (length,),
                    batch.logical_ids[i],
                    device=batch.device,
                    dtype=torch.long,
                )
                for i, length in enumerate(batch.encoder_lengths.tolist())
            ]
        )[None]
        encoder_pos = torch.cat(
            [
                batch.encoder_position_ids[i, :length]
                for i, length in enumerate(batch.encoder_lengths.tolist())
            ]
        )[None]

        def cat_canvas(name: str) -> torch.Tensor:
            value = getattr(batch, name)
            return torch.cat(
                [
                    value[i, :length]
                    for i, length in enumerate(batch.canvas_lengths.tolist())
                ]
            )[None]

        canvas_docs = torch.cat(
            [
                torch.full(
                    (length,),
                    batch.logical_ids[i],
                    device=batch.device,
                    dtype=torch.long,
                )
                for i, length in enumerate(batch.canvas_lengths.tolist())
            ]
        )[None]
        canvas_rows = torch.cat(
            [
                torch.full((length,), index, device=batch.device, dtype=torch.long)
                for index, length in enumerate(batch.canvas_lengths.tolist())
            ]
        )[None]
        canvas_pos = torch.cat(
            [
                batch.decoder_prefix_lengths[i]
                + torch.arange(length, device=batch.device)
                for i, length in enumerate(batch.canvas_lengths.tolist())
            ]
        )[None]
        encoder_validity = torch.ones_like(encoder_ids, dtype=torch.bool)
        canvas_validity = cat_canvas("canvas_semantic_validity")
        if self.physical_bucket_size is not None:
            encoder_length = self._bucket_length(encoder_ids.shape[1])
            canvas_length = self._bucket_length(canvas_docs.shape[1])
            encoder_ids = self._bucket_pad(encoder_ids, encoder_length, 0)
            encoder_ar = self._bucket_pad(encoder_ar, encoder_length, False)
            encoder_docs = self._bucket_pad(encoder_docs, encoder_length, -1)
            encoder_pos = self._bucket_pad(encoder_pos, encoder_length, 0)
            encoder_validity = self._bucket_pad(encoder_validity, encoder_length, False)
            canvas_docs = self._bucket_pad(canvas_docs, canvas_length, -1)
            canvas_rows = self._bucket_pad(canvas_rows, canvas_length, -1)
            canvas_pos = self._bucket_pad(canvas_pos, canvas_length, 0)
            canvas_validity = self._bucket_pad(canvas_validity, canvas_length, False)

            def bucket_canvas(name: str) -> torch.Tensor:
                return self._bucket_pad(cat_canvas(name), canvas_length, False)

            canvas_clean_ids = self._bucket_pad(
                cat_canvas("canvas_clean_ids"), canvas_length, 0
            )
            canvas_loss_mask = bucket_canvas("canvas_loss_mask")
            canvas_corruptible_mask = bucket_canvas("canvas_corruptible_mask")
            canvas_input_pinned_mask = bucket_canvas("canvas_input_pinned_mask")
            canvas_sc_eligible_mask = bucket_canvas("canvas_sc_eligible_mask")
            canvas_read_only_mask = bucket_canvas("canvas_read_only_mask")
            canvas_update_mask = bucket_canvas("canvas_update_mask")
        else:
            canvas_clean_ids = cat_canvas("canvas_clean_ids")
            canvas_loss_mask = cat_canvas("canvas_loss_mask")
            canvas_corruptible_mask = cat_canvas("canvas_corruptible_mask")
            canvas_input_pinned_mask = cat_canvas("canvas_input_pinned_mask")
            canvas_sc_eligible_mask = cat_canvas("canvas_sc_eligible_mask")
            canvas_read_only_mask = cat_canvas("canvas_read_only_mask")
            canvas_update_mask = cat_canvas("canvas_update_mask")
        if self.attention_backend == "flex_attention":
            encoder_mask = encoder_flex_block_mask(
                encoder_docs, encoder_validity, encoder_pos
            )
            encoder_sliding_mask = encoder_flex_block_mask(
                encoder_docs, encoder_validity, encoder_pos, self.sliding_window
            )
            decoder_mask = decoder_flex_block_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                batch.decoder_prefix_lengths,
                encoder_pos,
                logical_ids=batch.logical_ids,
            )
            decoder_sliding_mask = decoder_flex_block_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                batch.decoder_prefix_lengths,
                encoder_pos,
                self.sliding_window,
                logical_ids=batch.logical_ids,
            )
        else:
            encoder_mask = encoder_causal_mask(
                encoder_docs, encoder_validity, encoder_pos
            )
            encoder_sliding_mask = encoder_causal_mask(
                encoder_docs, encoder_validity, encoder_pos, self.sliding_window
            )
            decoder_mask = decoder_prefix_canvas_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                batch.decoder_prefix_lengths,
                encoder_pos,
                logical_ids=batch.logical_ids,
            )
            decoder_sliding_mask = decoder_prefix_canvas_mask(
                encoder_docs,
                canvas_docs,
                encoder_validity,
                canvas_validity,
                batch.decoder_prefix_lengths,
                encoder_pos,
                self.sliding_window,
                logical_ids=batch.logical_ids,
            )
        masks = {
            "full_attention": encoder_mask,
            "sliding_attention": encoder_sliding_mask,
        }
        decoder_masks = {
            "full_attention": decoder_mask,
            "sliding_attention": decoder_sliding_mask,
        }
        return PackedEncoderCanvasBatch(
            batch,
            encoder_ids,
            encoder_validity,
            encoder_ar,
            encoder_docs,
            encoder_pos,
            canvas_clean_ids,
            canvas_validity,
            canvas_loss_mask,
            canvas_corruptible_mask,
            canvas_input_pinned_mask,
            canvas_sc_eligible_mask,
            canvas_read_only_mask,
            canvas_update_mask,
            canvas_docs,
            canvas_rows,
            canvas_pos,
            prompt_offsets,
            canvas_offsets,
            masks,
            decoder_masks,
        )

    def corrupt(
        self,
        packed: PackedEncoderCanvasBatch,
        times: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> CorruptedCanvas:
        if times.shape != packed.batch.logical_ids.shape:
            raise ValueError("times must contain one value for each logical example")
        probabilities = times[packed.canvas_logical_row_indices].to(torch.float32)
        events = (
            (
                torch.rand(
                    probabilities.shape, device=packed.device, generator=generator
                )
                < probabilities
            )
            & packed.canvas_corruptible_mask
            & packed.canvas_semantic_validity
            & ~packed.canvas_input_pinned_mask
        )
        replacements = torch.randint(
            0,
            self.vocab_size,
            packed.canvas_clean_ids.shape,
            device=packed.device,
            generator=generator,
        )
        return CorruptedCanvas(
            torch.where(events, replacements, packed.canvas_clean_ids),
            events,
            probabilities,
        )

    def forward(
        self,
        model,
        packed: PackedEncoderCanvasBatch,
        input_ids: torch.Tensor,
        *,
        cache=None,
        self_conditioning_logits: torch.Tensor | None = None,
        self_conditioning_token_mask: torch.Tensor | None = None,
        unroll_steps: int = 1,
        pilot_for_single_step: bool = True,
        grad_through_steps: bool = False,
        k1_conditioning_mask: torch.Tensor | None = None,
        recurrent_conditioning_mask: torch.Tensor | None = None,
        update_mask: torch.Tensor | None = None,
        kernel_options: dict | None = None,
    ):
        """Run one packed canvas state through the native model."""
        if input_ids.shape != packed.canvas_clean_ids.shape:
            raise ValueError("packed canvas state must match canvas_clean_ids")
        return model(
            encoder_input_ids=packed.encoder_input_ids,
            encoder_attention_mask=packed.encoder_attention_mask,
            encoder_position_ids=packed.encoder_position_ids,
            decoder_input_ids=input_ids,
            decoder_attention_mask=packed.decoder_attention_mask,
            decoder_position_ids=packed.canvas_position_ids,
            past_key_values=cache,
            self_conditioning_logits=self_conditioning_logits,
            self_conditioning_token_mask=self_conditioning_token_mask,
            unroll_steps=unroll_steps,
            pilot_for_single_step=pilot_for_single_step,
            grad_through_steps=grad_through_steps,
            k1_conditioning_mask=k1_conditioning_mask,
            recurrent_conditioning_mask=recurrent_conditioning_mask,
            update_mask=update_mask,
            kernel_options=kernel_options,
        )

    @staticmethod
    def update(
        input_ids: torch.Tensor, logits: torch.Tensor, update_mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace only update-eligible canvas tokens with detached predictions."""
        if input_ids.shape != update_mask.shape or logits.shape[:2] != input_ids.shape:
            raise ValueError("canvas state, logits, and update mask must align")
        if update_mask.dtype is not torch.bool:
            raise TypeError("update_mask must be bool")
        predictions = logits.detach().argmax(-1).to(input_ids.dtype)
        return torch.where(update_mask, predictions, input_ids)
