"""Shared fakes for the multimodal sample-packing tests."""

from __future__ import annotations

import torch


class PadTokenizer:
    """Minimal right-padding tokenizer stub for collator tests."""

    padding_side = "right"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2

    def encode(self, *_args, **_kwargs):
        return [self.eos_token_id]

    def pad(
        self,
        features,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors=None,
    ):
        target_len = max(len(feature["input_ids"]) for feature in features)
        if max_length is not None and padding == "max_length":
            target_len = max_length
        if pad_to_multiple_of:
            target_len = (
                (target_len + pad_to_multiple_of - 1)
                // pad_to_multiple_of
                * pad_to_multiple_of
            )

        output = {}
        for key in set().union(*(feature.keys() for feature in features)):
            rows = []
            for feature in features:
                row = list(feature[key])
                pad_value = -100 if key == "labels" else 0
                rows.append(row + [pad_value] * (target_len - len(row)))
            output[key] = torch.tensor(rows)
        return output
