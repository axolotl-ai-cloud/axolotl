import pytest
import torch

from axolotl.core.trainers.diffusion_lm.backends.encoder_canvas import (
    EncoderCanvasBackend,
)
from axolotl.core.trainers.diffusion_lm.backends.full_sequence import (
    FullSequenceBackend,
)
from axolotl.core.trainers.diffusion_lm.collator import DiffusionCollator
from axolotl.core.trainers.diffusion_lm.sampling import native_packing_lengths


def test_long_response_selects_canvas_block_and_returns_tensors():
    torch.manual_seed(1)
    collator = DiffusionCollator(0, 3)
    out = collator(
        [
            {
                "input_ids": [2, 3, 10, 11, 12, 13, 14, 15],
                "labels": [-100, -100, 10, 11, 12, 13, 14, 15],
            }
        ]
    )
    assert isinstance(out, dict)
    assert all(isinstance(value, torch.Tensor) for value in out.values())
    assert out["canvas_loss_mask"].sum().item() == 3
    assert out["canvas_clean_ids"].shape == (1, 3)
    assert out["decoder_prefix_lengths"].item() == 5
    assert out["encoder_lengths"].item() == 8


def test_long_response_can_select_final_partial_block_and_advances_prefix(monkeypatch):
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([2]))
    out = DiffusionCollator(0, 3)(
        [{"input_ids": list(range(10)), "labels": [-100, -100] + list(range(2, 10))}]
    )
    assert out["selected_block_ids"].item() == 2
    assert out["decoder_prefix_lengths"].item() == 8
    torch.testing.assert_close(out["canvas_clean_ids"], torch.tensor([[8, 9, 0]]))
    torch.testing.assert_close(
        out["canvas_semantic_validity"], torch.tensor([[True, True, False]])
    )


def test_full_sequence_layout_retains_full_logical_stream_and_disjoint_supervision():
    out = DiffusionCollator(0, 3, layout="full_sequence")(
        [
            {
                "input_ids": [2, 3, 4, 5, 6],
                "labels": [-100, 3, -100, 5, 6],
            }
        ]
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


def test_nested_sample_packing_flattens_logical_examples_into_one_physical_row():
    batch = DiffusionCollator(0, 3, physical_pack_budget=13).build_batch(
        [
            [
                {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
                {"input_ids": [5, 6], "labels": [-100, 6]},
            ],
            [{"input_ids": [7, 8, 9], "labels": [-100, 8, 9]}],
        ]
    )
    assert batch.logical_ids.tolist() == [0, 1, 2]
    assert batch.encoder_lengths.tolist() == [3, 2, 3]


def test_packed_budget_rejects_or_drops_whole_logical_examples():
    features = [
        {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
        {"input_ids": [5, 6, 7], "labels": [-100, 6, 7]},
    ]
    with pytest.raises(ValueError, match="packed token budget"):
        DiffusionCollator(0, 3, physical_pack_budget=5).build_batch(features)
    with pytest.raises(ValueError, match="sampler must use native"):
        DiffusionCollator(
            0, 3, physical_pack_budget=5, overflow_policy="drop"
        ).build_batch(features)
    oversized = {"input_ids": list(range(8)), "labels": list(range(8))}
    batch = DiffusionCollator(
        0, 3, physical_pack_budget=5, overflow_policy="drop"
    ).build_batch([oversized, features[0]])
    assert batch.logical_ids.tolist() == [0]
    with pytest.raises(ValueError, match="packed token budget"):
        DiffusionCollator(0, 3, logical_sequence_length=2).build_batch(features[:1])


def test_visible_supervised_eos_tail_appends_to_logical_length_only():
    batch = DiffusionCollator(
        0,
        5,
        eos_tail="visible_supervised",
        eos_token_id=1,
        logical_sequence_length=5,
    ).build_batch([{"input_ids": [2, 3, 4], "labels": [-100, 3, 4]}])
    torch.testing.assert_close(
        batch.canvas_loss_mask, torch.tensor([[True, True, True, True, False]])
    )
    torch.testing.assert_close(batch.canvas_clean_ids, torch.tensor([[3, 4, 1, 0, 0]]))


def test_visible_supervised_eos_tail_preserves_existing_disjoint_labels():
    batch = DiffusionCollator(
        0,
        6,
        layout="full_sequence",
        eos_tail="visible_supervised",
        logical_sequence_length=6,
    ).build_batch([{"input_ids": [2, 3, 4, 5], "labels": [-100, 3, -100, 5]}])
    torch.testing.assert_close(
        batch.canvas_loss_mask,
        torch.tensor([[False, True, False, True, True, True]]),
    )
    torch.testing.assert_close(
        batch.canvas_clean_ids, torch.tensor([[2, 3, 4, 5, 0, 0]])
    )


def test_native_packing_lengths_charge_encoder_and_selected_canvas():
    records = [
        {"input_ids": [2, 3, 4, 5], "labels": [-100, -100, 4, 5]},
        {"input_ids": [2, 3, 4], "labels": [-100, 3, 4]},
    ]
    assert native_packing_lengths(
        records,
        eos_tail=None,
        logical_sequence_length=None,
        layout="encoder_canvas",
        canvas_width=1,
    ) == [5, 4]
    assert native_packing_lengths(
        records,
        eos_tail="visible_supervised",
        logical_sequence_length=8,
        layout="encoder_canvas",
        canvas_width=2,
    ) == [10, 10]


def test_encoder_canvas_disjoint_supervision_preserves_context_and_cost():
    feature = {
        "input_ids": list(range(8)),
        "labels": [-100, 1, -100, -100, 4, -100, 6, -100],
    }
    cost = native_packing_lengths(
        [feature],
        eos_tail=None,
        logical_sequence_length=None,
        layout="encoder_canvas",
        canvas_width=8,
    )
    assert cost == [14]
    batch = DiffusionCollator(0, 8, physical_pack_budget=14).build_batch([feature])
    assert batch.canvas_lengths.tolist() == [6]
    assert batch.canvas_loss_mask[0, :6].tolist() == [
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    assert batch.canvas_input_pinned_mask[0, :6].tolist() == [
        False,
        True,
        True,
        False,
        True,
        False,
    ]
    assert (batch.encoder_lengths + batch.canvas_lengths).tolist() == cost


def test_encoder_canvas_cost_includes_appended_tail_and_every_selected_block(
    monkeypatch,
):
    feature = {"input_ids": list(range(6)), "labels": [-100, 1, -100, -100, -100, -100]}
    cost = native_packing_lengths(
        [feature],
        eos_tail="visible_supervised",
        logical_sequence_length=12,
        layout="encoder_canvas",
        canvas_width=4,
    )
    assert cost == [16]
    for selection in range(3):
        monkeypatch.setattr(
            torch,
            "randint",
            lambda *args, selection=selection, **kwargs: torch.tensor([selection]),
        )
        batch = DiffusionCollator(
            0,
            4,
            logical_sequence_length=12,
            physical_pack_budget=16,
            eos_tail="visible_supervised",
        ).build_batch([feature])
        assert int((batch.encoder_lengths + batch.canvas_lengths).sum()) <= cost[0]
        assert batch.canvas_loss_mask.any()


def test_flex_encoder_canvas_payload_never_exceeds_total_bucket_allocation():
    total_budget = 384
    payload_capacity = 256
    feature = {
        "input_ids": list(range(129)),
        "labels": [-100, -100] + list(range(2, 129)),
    }
    batch = DiffusionCollator(
        0,
        128,
        physical_pack_budget=payload_capacity,
    ).build_batch([feature])
    packed = EncoderCanvasBackend(256, 32, attention_backend="flex_attention").pack(
        batch
    )
    assert (
        int(batch.encoder_lengths.sum() + batch.canvas_lengths.sum())
        == payload_capacity
    )
    assert (
        packed.encoder_input_ids.shape[1] + packed.canvas_clean_ids.shape[1]
        == total_budget
    )


def test_flex_full_sequence_payload_rounds_once_within_total_budget():
    total_budget = 384
    batch = DiffusionCollator(
        0,
        layout="full_sequence",
        physical_pack_budget=total_budget,
    ).build_batch([{"input_ids": list(range(257)), "labels": list(range(257))}])
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
