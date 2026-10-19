"""Packed FlashAttention metadata emitted by the sequence collator."""

import pytest
import torch
from transformers.modeling_flash_attention_utils import (
    prepare_fa_kwargs_from_position_ids,
)

from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)


class _Tokenizer:
    padding_side = "right"

    @staticmethod
    def pad(features, **kwargs):
        return {
            key: torch.tensor([feature[key] for feature in features])
            for key in features[0]
        }


def test_fa_varlen_collator_matches_transformers_metadata():
    features = [
        {
            "input_ids": [10, 11, 12, 13, 14],
            "position_ids": [0, 1, 2, 0, 1],
        }
    ]
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True)(
        features
    )

    (cu_q, cu_k), (max_q, max_k) = prepare_fa_kwargs_from_position_ids(
        batch["position_ids"]
    )
    torch.testing.assert_close(batch["cu_seq_lens_q"], cu_q)
    torch.testing.assert_close(batch["cu_seq_lens_k"], cu_k)
    assert batch["cu_seq_lens_k"] is not batch["cu_seq_lens_q"]
    assert batch["max_length_q"] == int(max_q)
    assert batch["max_length_k"] == int(max_k)


def test_fa_varlen_collator_is_opt_in():
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer())(
        [{"input_ids": [10, 11], "position_ids": [0, 1]}]
    )
    assert "cu_seq_lens_q" not in batch


def test_fa_varlen_collator_skips_padded_attention_mask():
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True)(
        [
            {
                "input_ids": [10, 11, 12, 0],
                "position_ids": [0, 1, 2, 3],
                "attention_mask": [1, 1, 1, 0],
            }
        ]
    )
    assert "cu_seq_lens_q" not in batch


def test_fa_varlen_collator_keeps_fully_valid_attention_mask():
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True)(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "position_ids": [0, 1, 0, 1],
                "attention_mask": [1, 1, 2, 2],
            }
        ]
    )
    assert batch["cu_seq_lens_q"].tolist() == [0, 2, 4]


def test_fa_varlen_collator_skips_dense_attention_mask():
    batch = DataCollatorForSeq2Seq(tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True)(
        [
            {
                "input_ids": [10, 11, 12, 13],
                "position_ids": [0, 1, 0, 1],
                "attention_mask": [
                    [
                        [1, 0, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 1],
                    ]
                ],
            }
        ]
    )
    assert batch["attention_mask"].shape == (1, 1, 4, 4)
    assert "cu_seq_lens_q" not in batch


@pytest.mark.parametrize(
    ("collator_cls", "attention_mask"),
    [
        (BatchSamplerDataCollatorForSeq2Seq, [1, 1, 1, 1, 1]),
        (V2BatchSamplerDataCollatorForSeq2Seq, [1, 1, 1, 2, 2]),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_packed_collator_computes_metadata_after_concatenation(
    collator_cls, attention_mask, nested
):
    features = [
        {
            "input_ids": [10, 11, 12],
            "position_ids": [0, 1, 2],
            "attention_mask": [1, 1, 1],
            "labels": [10, 11, 12],
            "length": 3,
        },
        {
            "input_ids": [13, 14],
            "position_ids": [0, 1],
            "attention_mask": [1, 1],
            "labels": [13, 14],
            "length": 2,
        },
    ]
    batch = collator_cls(tokenizer=_Tokenizer(), emit_fa_varlen_kwargs=True)(
        [features] if nested else features
    )

    assert batch["input_ids"].tolist() == [[10, 11, 12, 13, 14]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1]]
    assert batch["attention_mask"].tolist() == [attention_mask]
    assert batch["labels"].tolist() == [[10, 11, 12, 13, 14]]
    assert "length" not in batch
    assert batch["cu_seq_lens_q"].tolist() == [0, 3, 5]
    torch.testing.assert_close(batch["cu_seq_lens_k"], batch["cu_seq_lens_q"])
    assert batch["cu_seq_lens_k"] is not batch["cu_seq_lens_q"]
    assert batch["max_length_q"] == batch["max_length_k"] == 3
