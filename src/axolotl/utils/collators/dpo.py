"""DPO/ORPO/IPO/KTO data collator with pad_to_multiple_of support.

Extends TRL's DPODataCollatorWithPadding to round padded sequence lengths
up to a fixed multiple. This stabilizes Triton autotune caches for kernels
that key on sequence length (e.g. fla's linear attention kernels used by
Qwen3.5), which otherwise re-autotune on every distinct batch length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from trl.experimental.utils import DPODataCollatorWithPadding
from trl.trainer.utils import pad


def _round_up(length: int, multiple: int) -> int:
    return ((length + multiple - 1) // multiple) * multiple


@dataclass
class AxolotlDPODataCollatorWithPadding(DPODataCollatorWithPadding):
    """DPO data collator that pads to a multiple of ``pad_to_multiple_of``.

    Args:
        pad_token_id: Tokenizer pad token id (inherited).
        is_encoder_decoder: Whether the model is encoder-decoder (inherited).
        pad_to_multiple_of: If set, padded lengths are rounded up to this
            multiple. Helps stabilize Triton autotune caches.
    """

    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        pad_to_mult = self.pad_to_multiple_of

        padded_batch: dict[str, Any] = {}
        for k in features[0].keys():
            if k.endswith(
                ("_input_ids", "_attention_mask", "_labels", "_pixel_values")
            ):
                if self.is_encoder_decoder:
                    to_pad = [torch.LongTensor(ex[k]) for ex in features]

                    if k.startswith("prompt") and k.endswith("input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif (
                        k.startswith(("chosen", "rejected", "completion"))
                        or "decoder" in k
                    ):
                        padding_value = -100
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padded = pad_sequence(
                        to_pad, batch_first=True, padding_value=padding_value
                    )
                    if pad_to_mult:
                        cur = padded.shape[1]
                        target = _round_up(cur, pad_to_mult)
                        if target > cur:
                            extra = target - cur
                            pad_shape = list(padded.shape)
                            pad_shape[1] = extra
                            filler = torch.full(
                                pad_shape,
                                padding_value,
                                dtype=padded.dtype,
                                device=padded.device,
                            )
                            padded = torch.cat([padded, filler], dim=1)
                    padded_batch[k] = padded
                else:
                    if k.endswith("_input_ids"):
                        if self.pad_token_id is None:
                            raise ValueError(
                                "Padding is enabled, but the tokenizer is not configured with a padding token."
                            )
                        padding_value = self.pad_token_id
                    elif k.endswith("_labels"):
                        padding_value = -100
                    elif k.endswith("_attention_mask"):
                        padding_value = 0
                    elif k.endswith("_pixel_values"):
                        padding_value = 0
                    else:
                        raise ValueError(f"Unexpected key in batch '{k}'")

                    padding_side = (
                        "left"
                        if k in ("prompt_input_ids", "prompt_attention_mask")
                        else "right"
                    )

                    dtype = (
                        torch.float32 if k.endswith("_pixel_values") else torch.int64
                    )
                    to_pad = [torch.tensor(ex[k], dtype=dtype) for ex in features]

                    # trl.pad() natively supports pad_to_multiple_of
                    padded_batch[k] = pad(
                        to_pad,
                        padding_value=padding_value,
                        padding_side=padding_side,
                        pad_to_multiple_of=pad_to_mult,
                    )
            elif k.endswith("_logps"):
                padded_batch[k] = torch.tensor([ex[k] for ex in features])
            else:
                padded_batch[k] = [ex[k] for ex in features]

        return padded_batch
