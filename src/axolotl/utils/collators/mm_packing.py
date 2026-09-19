"""Packed multimodal SFT collator."""

from __future__ import annotations

from dataclasses import dataclass

from axolotl.utils.collators.batching import DataCollatorForSeq2Seq
from axolotl.utils.collators.mm_pack_core import (
    PackedSample,
    batch_packed_samples,
    normalize_groups,
    pack_group,
)


@dataclass
class MultiModalBatchSamplerDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if features and isinstance(features[0], PackedSample):
            samples = list(features)
        else:
            samples = [pack_group(group) for group in normalize_groups(features)]

        def _text_collate(text_features, return_tensors=None):
            return DataCollatorForSeq2Seq.__call__(
                self, text_features, return_tensors=return_tensors
            )

        return batch_packed_samples(samples, _text_collate, return_tensors)
