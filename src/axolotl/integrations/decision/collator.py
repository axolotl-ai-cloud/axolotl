from __future__ import annotations

import torch

from axolotl.integrations.diffusion.lm.batch import DiffusionBatch

from .records import DecisionCanvas


class DecisionCanvasCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id

    def __call__(self, canvases: list[DecisionCanvas]) -> DiffusionBatch:
        if not canvases:
            raise ValueError("empty decision batch")
        pw = max(len(c.prompt_ids) for c in canvases)
        cw = len(canvases[0].canvas_ids)
        if any(len(c.canvas_ids) != cw or len(c.semantic_mask) != cw for c in canvases):
            raise ValueError("canvases must share width")
        b = len(canvases)
        prompt = torch.full((b, pw), self.pad_token_id, dtype=torch.long)
        canvas = torch.tensor([c.canvas_ids for c in canvases], dtype=torch.long)
        ev = torch.zeros((b, pw), dtype=torch.bool)
        ar = ev.clone()
        sem = torch.tensor([c.semantic_mask for c in canvases], dtype=torch.bool)
        loss = torch.zeros((b, cw), dtype=torch.bool)
        for i, c in enumerate(canvases):
            n = len(c.prompt_ids)
            prompt[i, :n] = torch.tensor(c.prompt_ids)
            ev[i, :n] = True
            ar[i, 1:n] = True
            if c.prompt_slot_mask:
                if len(c.prompt_slot_mask) != n:
                    raise ValueError("prompt_slot_mask must align with prompt_ids")
                ar[i, :n] &= ~torch.tensor(c.prompt_slot_mask, dtype=torch.bool)
            loss[i, torch.tensor(c.label_positions, dtype=torch.long)] = True
        pinned = torch.tensor([c.pinned_mask for c in canvases], dtype=torch.bool)
        return DiffusionBatch(
            encoder_input_ids=prompt,
            encoder_validity=ev,
            encoder_ar_valid_mask=ar,
            encoder_document_ids=torch.arange(b)[:, None].expand(-1, pw),
            encoder_position_ids=torch.arange(pw)[None].expand(b, -1),
            canvas_clean_ids=canvas,
            canvas_semantic_validity=sem,
            canvas_loss_mask=loss,
            canvas_corruptible_mask=loss,
            canvas_input_pinned_mask=pinned,
            canvas_sc_eligible_mask=loss & ~pinned,
            canvas_read_only_mask=~loss,
            canvas_update_mask=loss & ~pinned,
            logical_ids=torch.arange(b),
            encoder_lengths=ev.sum(1),
            canvas_lengths=sem.sum(1),
            decoder_prefix_lengths=ev.sum(1),
            selected_block_ids=torch.zeros(b, dtype=torch.long),
        )
