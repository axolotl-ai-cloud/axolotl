"""Packed FlashAttention metadata emitted by the sequence collator."""

import torch
from transformers.modeling_flash_attention_utils import (
    prepare_fa_kwargs_from_position_ids,
)

from axolotl.utils.collators import DataCollatorForSeq2Seq


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
