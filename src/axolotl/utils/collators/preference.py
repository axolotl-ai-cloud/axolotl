"""Preference collator that restricts DPO log-prob sums to per-token loss masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import pad

LOSS_MASK_COLUMNS = ("chosen_loss_mask", "rejected_loss_mask")


@dataclass
class AxolotlDataCollatorForPreference(DataCollatorForPreference):
    """TRL's preference collator, with ``completion_mask`` narrowed by per-token loss masks.

    Examples carrying ``chosen_loss_mask``/``rejected_loss_mask`` (one 0/1 entry per
    completion token) keep only the flagged completion tokens in ``completion_mask``,
    which every TRL DPO loss uses to sum log-probs. The masks are truncated and padded
    exactly like the ids, so they stay aligned under ``max_length``.
    """

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        output = super().torch_call(examples)
        if LOSS_MASK_COLUMNS[0] not in examples[0]:
            return output

        masks = []
        for side, column in zip(("chosen", "rejected"), LOSS_MASK_COLUMNS, strict=True):
            for example in examples:
                loss_mask = list(example[column])
                if len(loss_mask) != len(example[f"{side}_ids"]):
                    raise ValueError(
                        f"`{column}` has {len(loss_mask)} entries but `{side}_ids` has "
                        f"{len(example[f'{side}_ids'])} tokens."
                    )
                masks.append([0] * len(example["prompt_ids"]) + loss_mask)

        if self.max_length is not None:
            if self.truncation_mode == "keep_start":
                sl = slice(None, self.max_length)
            else:
                sl = slice(-self.max_length, None)
            masks = [m[sl] for m in masks]

        loss_mask = pad(
            [torch.tensor(m, dtype=torch.long) for m in masks],
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["completion_mask"] = output["completion_mask"] * loss_mask
        return output
