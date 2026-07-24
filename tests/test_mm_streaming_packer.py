"""Unit tests for the buffered multimodal sample packer."""

from __future__ import annotations

import numpy as np

from axolotl.utils.collators.mm_pack_core import PackedSample
from axolotl.utils.data.mm_streaming import (
    BufferedMultimodalPacker,
    iter_tokenized_mm_rows,
)


def _rows(lengths, hw=8):
    """Tokenized llava-style rows; input_ids are globally unique per row."""
    rows = []
    cursor = 1
    for length in lengths:
        ids = list(range(cursor, cursor + length))
        cursor += length
        rows.append(
            {
                "input_ids": ids,
                "labels": list(ids),
                "attention_mask": [1] * length,
                "position_ids": list(range(length)),
                "length": length,
                "pixel_values": np.ones((1, 3, hw, hw), np.float32),
            }
        )
    return rows


def _all_ids(rows):
    return {tid for row in rows for tid in row["input_ids"]}


def _assert_pack_invariants(sample: PackedSample, batch_max_len: int):
    row = sample.as_row()
    input_ids = np.asarray(row["input_ids"])
    attn = np.asarray(row["attention_mask"])
    pos = np.asarray(row["position_ids"])

    assert input_ids.shape[0] <= batch_max_len
    assert attn.shape == input_ids.shape == pos.shape

    # attention_mask carries 1-based, contiguous segment ids (1,1,2,2,3,...).
    seg_ids = list(dict.fromkeys(attn.tolist()))
    assert seg_ids == list(range(1, len(seg_ids) + 1))

    n_images = 0
    for seg in seg_ids:
        seg_len = int((attn == seg).sum())
        # position_ids reset to 0..seg_len-1 for each packed subsequence.
        seg_pos = pos[attn == seg]
        assert seg_pos.tolist() == list(range(seg_len))
        n_images += 1

    # one (1, 3, H, W) image per packed row -> (n_rows, 3, H, W).
    assert row["pixel_values"].shape[0] == n_images


def test_pack_invariants_and_exact_coverage():
    batch_max_len = 12
    rows = _rows([5, 4, 3, 7, 2, 6, 1, 8])
    packer = BufferedMultimodalPacker(
        rows, batch_max_len=batch_max_len, bin_size=200, buffer_size=100
    )
    packs = list(packer)

    seen: list[int] = []
    for pack in packs:
        _assert_pack_invariants(pack, batch_max_len)
        seen.extend(pack.as_row()["input_ids"].tolist())

    # every source token appears exactly once across all packs
    assert sorted(seen) == sorted(_all_ids(rows))
    assert len(seen) == len(set(seen))


def test_multiple_buffer_refills():
    batch_max_len = 10
    lengths = [3, 4, 2, 5, 3, 4, 2, 6, 3, 3, 4, 2]
    rows = _rows(lengths)
    # buffer smaller than dataset -> several independent pack/evict cycles
    packer = BufferedMultimodalPacker(
        rows, batch_max_len=batch_max_len, bin_size=200, buffer_size=4
    )
    packs = list(packer)

    seen: list[int] = []
    for pack in packs:
        _assert_pack_invariants(pack, batch_max_len)
        seen.extend(pack.as_row()["input_ids"].tolist())
    assert sorted(seen) == sorted(_all_ids(rows))


def test_source_exhaustion_drains_partial_buffer():
    batch_max_len = 20
    rows = _rows([4, 4, 4])  # never fills a buffer_size=100 window
    packer = BufferedMultimodalPacker(
        rows, batch_max_len=batch_max_len, bin_size=200, buffer_size=100
    )
    packs = list(packer)
    assert packs, "drain-on-exhaustion must still emit the buffered rows"
    seen = [tid for p in packs for tid in p.as_row()["input_ids"].tolist()]
    assert sorted(seen) == sorted(_all_ids(rows))


def test_length_key_fallback_to_input_ids():
    rows = _rows([5, 6])
    for row in rows:
        del row["length"]
    packer = BufferedMultimodalPacker(rows, batch_max_len=20, buffer_size=100)
    packs = list(packer)
    seen = [tid for p in packs for tid in p.as_row()["input_ids"].tolist()]
    assert sorted(seen) == sorted(_all_ids(rows))


def test_callable_source_reiterable():
    lengths = [4, 4, 4]
    packer = BufferedMultimodalPacker(
        lambda: iter(_rows(lengths)), batch_max_len=20, buffer_size=100
    )
    first = list(packer)
    second = list(packer)
    assert [p.as_row()["input_ids"].shape[0] for p in first] == [
        p.as_row()["input_ids"].shape[0] for p in second
    ]


def test_ragged_pixtral_media_merges_via_shared_helper():
    # differing per-image H/W within a pack -> pad-and-stack (shared helper path)
    rows = [
        {
            "input_ids": [1, 2, 3],
            "labels": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "position_ids": [0, 1, 2],
            "length": 3,
            "pixel_values": np.ones((1, 3, 96, 64), np.float32),
        },
        {
            "input_ids": [4, 5],
            "labels": [4, 5],
            "attention_mask": [1, 1],
            "position_ids": [0, 1],
            "length": 2,
            "pixel_values": np.full((1, 3, 64, 128), 2.0, np.float32),
        },
    ]
    packer = BufferedMultimodalPacker(rows, batch_max_len=16, buffer_size=100)
    packs = list(packer)
    assert len(packs) == 1
    px = packs[0].as_row()["pixel_values"]
    assert tuple(px.shape) == (2, 3, 96, 128)


def test_packs_batch_through_shared_collator():
    from axolotl.utils.collators import MultiModalBatchSamplerDataCollatorForSeq2Seq

    from tests.mm_packing_utils import PadTokenizer

    rows = _rows([5, 4, 3, 6])
    packs = list(
        BufferedMultimodalPacker(rows, batch_max_len=12, bin_size=200, buffer_size=100)
    )
    collator = MultiModalBatchSamplerDataCollatorForSeq2Seq(
        tokenizer=PadTokenizer(), padding=True, return_tensors="pt"
    )
    # PackedSample list flows straight into the collator (streaming glue).
    batch = collator(packs)
    assert batch["input_ids"].shape[0] == len(packs)
    total_images = sum(int(p.as_row()["pixel_values"].shape[0]) for p in packs)
    assert batch["pixel_values"].shape[0] == total_images


def test_drop_attention_mask_omits_mask_and_keeps_position_ids():
    rows = _rows([5, 4, 3])
    packer = BufferedMultimodalPacker(
        rows, batch_max_len=12, buffer_size=100, drop_attention_mask=True
    )
    packs = list(packer)
    assert packs
    for pack in packs:
        row = pack.as_row()
        # position_ids restarts drive packed-sequence isolation downstream.
        assert "attention_mask" not in row
        assert "position_ids" in row


def test_build_buffered_mm_packer_wires_drop_attention_mask_from_cfg():
    from axolotl.utils.data.mm_streaming import build_buffered_mm_packer
    from axolotl.utils.dict import DictDefault

    cfg = DictDefault(sequence_len=128, attn_decontaminates_packing=True)
    assert build_buffered_mm_packer([], cfg).drop_attention_mask is True

    cfg = DictDefault(sequence_len=128)
    assert build_buffered_mm_packer([], cfg).drop_attention_mask is False


class _StubProcessor:
    def __init__(self, lengths):
        self._lengths = iter(lengths)

    def apply_chat_template(self, messages, **_kwargs):
        import torch

        length = next(self._lengths)
        return {
            "input_ids": torch.arange(1, length + 1).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
            "pixel_values": torch.ones(1, 3, 8, 8),
        }


class _StubStrategy:
    chat_template = "tmpl"

    def __init__(self, lengths):
        self.processor = _StubProcessor(lengths)

    def __call__(self, examples):
        return [{"messages": ex.get("messages", [])} for ex in examples]

    def process_labels(self, input_ids):
        return input_ids.clone()


def test_iter_tokenized_mm_rows_shapes_and_fields():
    import torch

    lengths = [5, 3]
    strategy = _StubStrategy(lengths)
    raw = [{"messages": []}, {"messages": []}]
    rows = list(iter_tokenized_mm_rows(raw, strategy))

    assert len(rows) == 2
    for row, length in zip(rows, lengths, strict=True):
        assert int(row["length"]) == length
        assert row["input_ids"].shape[0] == length
        assert torch.equal(row["position_ids"], torch.arange(length))
        assert row["attention_mask"].shape[0] == length
        assert tuple(row["pixel_values"].shape) == (1, 3, 8, 8)


def test_overlong_row_dropped_from_packs():
    batch_max_len = 12
    rows = _rows([5, 4])
    # a row longer than batch_max_len can never fit a bin -> dropped.
    # ids are disjoint from the fitting rows to detect any leakage.
    overlong = {
        "input_ids": list(range(1000, 1020)),
        "labels": list(range(1000, 1020)),
        "attention_mask": [1] * 20,
        "position_ids": list(range(20)),
        "length": 20,
        "pixel_values": np.ones((1, 3, 8, 8), np.float32),
    }
    overlong_ids = set(overlong["input_ids"])
    packer = BufferedMultimodalPacker(
        [rows[0], overlong, rows[1]],
        batch_max_len=batch_max_len,
        bin_size=200,
        buffer_size=100,
    )
    packs = list(packer)
    seen = [tid for p in packs for tid in p.as_row()["input_ids"].tolist()]
    assert not (overlong_ids & set(seen)), "over-long row must not appear in any pack"
    assert sorted(seen) == sorted(_all_ids(rows))


def test_token_type_ids_carried_into_pack():
    rows = [
        {
            "input_ids": [1, 2, 3],
            "labels": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "position_ids": [0, 1, 2],
            "token_type_ids": [0, 0, 1],
            "length": 3,
            "pixel_values": np.ones((1, 3, 8, 8), np.float32),
        },
        {
            "input_ids": [4, 5],
            "labels": [4, 5],
            "attention_mask": [1, 1],
            "position_ids": [0, 1],
            "token_type_ids": [1, 0],
            "length": 2,
            "pixel_values": np.ones((1, 3, 8, 8), np.float32),
        },
    ]
    packer = BufferedMultimodalPacker(rows, batch_max_len=16, buffer_size=100)
    packs = list(packer)
    assert len(packs) == 1
    row = packs[0].as_row()
    assert "token_type_ids" in row
    assert row["token_type_ids"].tolist() == [0, 0, 1, 1, 0]


class _TokenTypeProcessor(_StubProcessor):
    def apply_chat_template(self, messages, **kwargs):
        import torch

        batch = super().apply_chat_template(messages, **kwargs)
        length = int(batch["input_ids"].shape[1])
        batch["token_type_ids"] = torch.zeros(1, length, dtype=torch.long)
        return batch


class _TokenTypeStrategy(_StubStrategy):
    def __init__(self, lengths):
        super().__init__(lengths)
        self.processor = _TokenTypeProcessor(lengths)


def test_iter_tokenized_mm_rows_passes_token_type_ids():
    strategy = _TokenTypeStrategy([5, 3])
    raw = [{"messages": []}, {"messages": []}]
    rows = list(iter_tokenized_mm_rows(raw, strategy))
    for row, length in zip(rows, [5, 3], strict=True):
        assert "token_type_ids" in row
        assert row["token_type_ids"].shape[0] == length


class _UnknownMediaProcessor(_StubProcessor):
    def apply_chat_template(self, messages, **kwargs):
        import torch

        batch = super().apply_chat_template(messages, **kwargs)
        # an unsupported modality key (e.g. Voxtral audio) the packer can't handle
        batch["input_features"] = torch.ones(1, 80, 3000)
        return batch


class _UnknownMediaStrategy(_StubStrategy):
    def __init__(self, lengths):
        super().__init__(lengths)
        self.processor = _UnknownMediaProcessor(lengths)


def test_iter_tokenized_mm_rows_raises_on_unknown_media_key():
    import pytest

    strategy = _UnknownMediaStrategy([5])
    raw = [{"messages": []}]
    with pytest.raises(ValueError, match="input_features"):
        list(iter_tokenized_mm_rows(raw, strategy))


def test_iter_tokenized_mm_rows_ok_with_only_supported_keys():
    strategy = _StubStrategy([5, 3])
    raw = [{"messages": []}, {"messages": []}]
    rows = list(iter_tokenized_mm_rows(raw, strategy))
    assert len(rows) == 2
    for row in rows:
        assert "pixel_values" in row


def test_producer_feeds_buffered_packer_end_to_end():
    strategy = _StubStrategy([5, 4, 6, 3])
    raw = [{"messages": []} for _ in range(4)]
    packer = BufferedMultimodalPacker(
        lambda: iter_tokenized_mm_rows(raw, strategy),
        batch_max_len=12,
        bin_size=200,
        buffer_size=100,
    )
    packs = list(packer)
    total = sum(int(p.as_row()["input_ids"].shape[0]) for p in packs)
    assert total == 5 + 4 + 6 + 3
    for pack in packs:
        assert pack.as_row()["input_ids"].shape[0] <= 12
