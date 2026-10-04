"""Nemotron full-sequence collation and physical packing contracts."""

import pytest
import torch

from axolotl.integrations.diffusion.lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.integrations.diffusion.lm.collator import DiffusionCollator
from axolotl.integrations.diffusion.lm.sampling import native_packing_lengths


def test_full_sequence_retains_disjoint_supervision_and_context():
    out = DiffusionCollator(0)(
        [{"input_ids": [2, 3, 4, 5, 6], "labels": [-100, 3, -100, 5, 6]}]
    )
    torch.testing.assert_close(out["canvas_clean_ids"], torch.tensor([[2, 3, 4, 5, 6]]))
    torch.testing.assert_close(
        out["canvas_loss_mask"], torch.tensor([[False, True, False, True, True]])
    )
    torch.testing.assert_close(
        out["canvas_input_pinned_mask"],
        torch.tensor([[True, False, True, False, False]]),
    )
    assert out["canvas_lengths"].item() == 5
    assert out["encoder_lengths"].item() == 0
    assert out["decoder_prefix_lengths"].item() == 0
    assert out["selected_block_ids"].item() == 0


def test_nested_sample_packing_keeps_all_logical_examples():
    batch = DiffusionCollator(0, physical_pack_budget=8).build_batch(
        [
            [
                {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
                {"input_ids": [5, 6], "labels": [-100, 6]},
            ],
            [{"input_ids": [7, 8, 9], "labels": [-100, 8, 9]}],
        ]
    )
    assert batch.logical_ids.tolist() == [0, 1, 2]
    assert batch.canvas_lengths.tolist() == [3, 2, 3]
    assert batch.encoder_lengths.tolist() == [0, 0, 0]


def test_packed_budget_rejects_or_drops_whole_logical_examples():
    features = [
        {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
    ]
    with pytest.raises(ValueError, match="sampler must use native"):
        DiffusionCollator(0, physical_pack_budget=5).build_batch(features)
    oversized = {"input_ids": list(range(8)), "labels": list(range(8))}
    batch = DiffusionCollator(
        0, physical_pack_budget=5, overflow_policy="drop"
    ).build_batch([oversized, features[0]])
    assert batch.logical_ids.tolist() == [0]
    with pytest.raises(ValueError, match="packed token budget"):
        DiffusionCollator(0, logical_sequence_length=2).build_batch(features[:1])


def test_visible_eos_tail_preserves_disjoint_labels():
    batch = DiffusionCollator(
        0,
        eos_tail="visible_supervised",
        eos_token_id=1,
        logical_sequence_length=6,
    ).build_batch([{"input_ids": [2, 3, 4, 5], "labels": [-100, 3, -100, 5]}])
    torch.testing.assert_close(
        batch.canvas_loss_mask,
        torch.tensor([[False, True, False, True, True, True]]),
    )
    torch.testing.assert_close(
        batch.canvas_clean_ids, torch.tensor([[2, 3, 4, 5, 1, 0]])
    )
    assert native_packing_lengths(
        [{"input_ids": [2, 3, 4, 5]}],
        eos_tail="visible_supervised",
        logical_sequence_length=6,
    ) == [6]


def test_unsupported_encoder_canvas_arguments_are_rejected():
    with pytest.raises(ValueError, match="full_sequence"):
        DiffusionCollator(0, layout="encoder_canvas")
    with pytest.raises(ValueError, match="canvas_width"):
        DiffusionCollator(0, canvas_width=8)


def test_flex_full_sequence_payload_rounds_once_within_total_budget():
    total_budget = 384
    batch = DiffusionCollator(0, physical_pack_budget=total_budget).build_batch(
        [{"input_ids": list(range(257)), "labels": list(range(257))}]
    )
    packed = FullSequenceBackend(
        mask_token_id=99, attention_backend="flex_attention"
    ).pack(
        batch.canvas_clean_ids,
        torch.where(
            batch.canvas_semantic_validity,
            torch.zeros_like(batch.canvas_clean_ids),
            torch.full_like(batch.canvas_clean_ids, -1),
        ),
        batch.canvas_semantic_validity,
    )
    assert packed["input_ids"].shape[1] == total_budget
