"""Guard against ``batch_ring`` corrupting pre-packed multi-document sequences.

``batch_ring`` uses the ring-flash-attn *batch* API, which has no ``cu_seqlens``
channel, so it rotates K/V across document boundaries inside a packed sequence and
silently inflates the loss (axolotl-ai-cloud/axolotl#3608). ``varlen_llama3`` derives
``cu_seqlens`` from ``position_ids`` and is unaffected. These tests pin the config
that reproduces the bug to a loud error rather than a wrong number.
"""

import sys
import types

import pytest
import torch

from axolotl.utils.ctx_managers.sequence_parallel import (
    _has_multiple_documents,
    apply_sequence_parallelism,
)
from axolotl.utils.schemas.enums import RingAttnFunc


def _batch(
    position_ids: list[int], attention_mask: list[int] | None = None
) -> dict[str, torch.Tensor]:
    pos = torch.tensor([position_ids], dtype=torch.long)
    seq_len = pos.size(1)
    batch = {
        "input_ids": torch.ones(1, seq_len, dtype=torch.long),
        "labels": torch.ones(1, seq_len, dtype=torch.long),
        "position_ids": pos,
    }
    if attention_mask is not None:
        batch["attention_mask"] = torch.tensor([attention_mask], dtype=torch.long)
    return batch


class TestHasMultipleDocuments:
    """``_has_multiple_documents`` mirrors get_cu_seqlens boundary semantics."""

    @pytest.mark.parametrize(
        "position_ids, expected",
        [
            ([0, 1, 2, 3, 4, 5, 6, 7], False),  # single contiguous document
            ([0, 1, 2, 3, 0, 1, 2, 3], True),  # two packed documents
            ([0, 1, 2, 0, 1, 0, 1, 2], True),  # three packed documents
            ([0, 1, 2, 3, 0, 0, 0, 0], False),  # single doc + right zero-padding
            ([0, 1, 0, 1, 0, 0, 0, 0], True),  # two docs + right zero-padding
        ],
    )
    def test_detects_resets(self, position_ids, expected):
        assert _has_multiple_documents(_batch(position_ids)["position_ids"]) is expected

    def test_single_doc_with_range_padding_is_not_a_false_positive(self):
        # DataCollatorForSeq2Seq pads position_ids with range(pad_len), not zeros, so
        # a padded single document also resets to 0 (e.g. [0,1,2,3] padded by 3 becomes
        # [0,1,2,3,0,1,2]). Without attention_mask this must fall back to the trailing
        # zero-padding heuristic and would false-positive; attention_mask disambiguates.
        batch = _batch([0, 1, 2, 3, 0, 1, 2], attention_mask=[1, 1, 1, 1, 0, 0, 0])
        assert (
            _has_multiple_documents(batch["position_ids"], batch["attention_mask"])
            is False
        )

    def test_two_docs_with_range_padding_still_detected(self):
        # Two real documents ([0,1,2] then [0,1]) followed by range(3) padding.
        batch = _batch(
            [0, 1, 2, 0, 1, 0, 1, 2], attention_mask=[1, 1, 1, 1, 1, 0, 0, 0]
        )
        assert (
            _has_multiple_documents(batch["position_ids"], batch["attention_mask"])
            is True
        )


class TestBatchRingRejectsPackedDocuments:
    """apply_sequence_parallelism must reject batch_ring on multi-doc sequences."""

    @pytest.fixture(autouse=True)
    def _mock_ring_flash_attn(self, monkeypatch):
        # Present so the varlen control can reach update_ring_attn_params on CPU;
        # the batch_ring guard raises before ever importing it.
        if "ring_flash_attn" not in sys.modules:
            monkeypatch.setitem(
                sys.modules, "ring_flash_attn", types.ModuleType("ring_flash_attn")
            )

    def test_batch_ring_multi_doc_raises(self):
        with pytest.raises(ValueError, match="varlen_llama3"):
            apply_sequence_parallelism(
                batch=_batch([0, 1, 2, 3, 0, 1, 2, 3]),
                local_rank=0,
                local_world_size=1,
                gradient_accumulation_steps=1,
                ring_attn_func=RingAttnFunc.BATCH_RING,
            )

    def test_batch_ring_single_doc_ok(self):
        # A single document per sequence is the legitimate batch_ring case; it must
        # not trip the guard (the bug's own single-doc control is bitwise-clean).
        batch, orig_len, _ = apply_sequence_parallelism(
            batch=_batch([0, 1, 2, 3, 4, 5, 6, 7]),
            local_rank=0,
            local_world_size=1,
            gradient_accumulation_steps=1,
            ring_attn_func=RingAttnFunc.BATCH_RING,
        )
        assert orig_len == 8
        assert batch["input_ids"].shape[1] == 8

    def test_batch_ring_multi_doc_raises_with_micro_batch_size_gt_1(self):
        # Packed multi-doc rows can also show up with micro_batch_size > 1 (multiple
        # rows per batch); the guard must not be skipped just because batch_size != 1.
        position_ids = torch.tensor(
            [[0, 1, 2, 3, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long
        )
        batch = {
            "input_ids": torch.ones(2, 8, dtype=torch.long),
            "labels": torch.ones(2, 8, dtype=torch.long),
            "position_ids": position_ids,
        }
        with pytest.raises(ValueError, match="varlen_llama3"):
            apply_sequence_parallelism(
                batch=batch,
                local_rank=0,
                local_world_size=1,
                gradient_accumulation_steps=1,
                ring_attn_func=RingAttnFunc.BATCH_RING,
            )
