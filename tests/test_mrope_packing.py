"""mRoPE packed position ids: exactness vs per-sample get_rope_index and sample isolation."""

from __future__ import annotations

import pytest
import torch
from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration

from axolotl.monkeypatch.mrope_packing import (
    build_packed_mrope_position_ids,
    find_mrope_model,
    patch_mrope_packing,
    restart_positions_per_segment,
)

VISION_START, VISION_END, IMAGE, VIDEO = 3, 4, 5, 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen2VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        vision_config={
            "depth": 1,
            "embed_dim": 16,
            "hidden_size": 32,
            "num_heads": 2,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_start_token_id=VISION_START,
        vision_end_token_id=VISION_END,
    )
    config._attn_implementation = "sdpa"
    return Qwen2VLForConditionalGeneration(config).eval()


def _row(text_before: int, grid: tuple[int, int, int] | None, text_after: int):
    """One sample: text tokens, optional image block (merged 2x2), text tokens."""
    ids = list(range(10, 10 + text_before))
    types = [0] * text_before
    grids = []
    if grid is not None:
        num_tokens = grid[0] * (grid[1] // 2) * (grid[2] // 2)
        ids += [VISION_START] + [IMAGE] * num_tokens + [VISION_END]
        types += [0] + [1] * num_tokens + [0]
        grids.append(list(grid))
    ids += list(range(20, 20 + text_after))
    types += [0] * text_after
    return {
        "input_ids": torch.tensor(ids),
        "mm_token_type_ids": torch.tensor(types),
        "image_grid_thw": torch.tensor(grids) if grids else None,
        "position_ids": torch.arange(len(ids)),
    }


def _pack(rows, pad_to: int | None = None):
    ids = torch.cat([r["input_ids"] for r in rows])
    types = torch.cat([r["mm_token_type_ids"] for r in rows])
    pos = torch.cat([r["position_ids"] for r in rows])
    grids = [r["image_grid_thw"] for r in rows if r["image_grid_thw"] is not None]
    if pad_to is not None and pad_to > len(ids):
        pad = pad_to - len(ids)
        ids = torch.cat([ids, torch.zeros(pad, dtype=ids.dtype)])
        types = torch.cat([types, torch.zeros(pad, dtype=types.dtype)])
        pos = torch.cat(
            [pos, torch.arange(pad)]
        )  # DataCollatorForSeq2Seq pads with range()
    return ids, types, pos, torch.cat(grids) if grids else None


def _per_sample(inner, rows):
    outs = []
    for r in rows:
        pos, _ = inner.get_rope_index(
            r["input_ids"][None],
            mm_token_type_ids=r["mm_token_type_ids"][None],
            image_grid_thw=r["image_grid_thw"],
            attention_mask=None,
        )
        outs.append(pos[:, 0])
    return torch.cat(outs, dim=1)


def test_restart_positions_per_segment_subtracts_segment_start():
    text = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2, 3]])
    pos = (
        torch.arange(9)[None, None].expand(3, 1, 9)
        + torch.tensor([0, 100, 200])[:, None, None]
    )
    out = restart_positions_per_segment(pos, text)
    assert torch.equal(out[0, 0], torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3]))
    assert torch.equal(out[1], out[0]) and torch.equal(out[2], out[0])


def test_packed_positions_match_per_sample_get_rope_index(model):
    inner = find_mrope_model(model)
    rows = [_row(3, (1, 4, 4), 2), _row(2, (1, 6, 4), 4), _row(1, (1, 4, 6), 1)]
    ids, types, text_pos, grids = _pack(rows)
    pos4 = build_packed_mrope_position_ids(
        inner,
        ids[None],
        text_pos[None],
        image_grid_thw=grids,
        mm_token_type_ids=types[None],
    )
    assert pos4.shape == (4, 1, len(ids))
    assert torch.equal(pos4[0, 0], text_pos)
    assert torch.equal(pos4[1:, 0], _per_sample(inner, rows))


def test_text_only_pack_expands_text_row(model):
    inner = find_mrope_model(model)
    rows = [_row(4, None, 3), _row(2, None, 5)]
    ids, types, text_pos, _ = _pack(rows)
    pos4 = build_packed_mrope_position_ids(
        inner, ids[None], text_pos[None], mm_token_type_ids=types[None]
    )
    assert pos4.shape == (4, 1, len(ids))
    assert all(torch.equal(pos4[i, 0], text_pos) for i in range(4))
    assert torch.equal(pos4[1:, 0], _per_sample(inner, rows))


def test_padded_multi_row_batch(model):
    inner = find_mrope_model(model)
    packs = [[_row(3, (1, 4, 4), 2), _row(2, (1, 4, 4), 1)], [_row(1, (1, 6, 4), 2)]]
    packed = [_pack(p) for p in packs]
    target = max(len(x[0]) for x in packed)
    packed = [_pack(p, pad_to=target) for p in packs]
    ids = torch.stack([x[0] for x in packed])
    types = torch.stack([x[1] for x in packed])
    text_pos = torch.stack([x[2] for x in packed])
    grids = torch.cat([x[3] for x in packed])
    pos4 = build_packed_mrope_position_ids(
        inner, ids, text_pos, image_grid_thw=grids, mm_token_type_ids=types
    )
    assert pos4.shape == (4, 2, target)
    for b, pack in enumerate(packs):
        ref = _per_sample(inner, pack)
        n = ref.shape[1]
        assert torch.equal(pos4[1:, b, :n], ref)
        # padding is one text-only segment: every grid row follows the text row
        assert torch.equal(pos4[1:, b, n:], pos4[0, b, n:].expand(3, -1))


def test_missing_mm_token_type_ids_falls_back_to_token_ids(model):
    inner = find_mrope_model(model)
    rows = [_row(3, (1, 4, 4), 2), _row(2, (1, 4, 4), 1)]
    ids, types, text_pos, grids = _pack(rows)
    with_types = build_packed_mrope_position_ids(
        inner,
        ids[None],
        text_pos[None],
        image_grid_thw=grids,
        mm_token_type_ids=types[None],
    )
    without = build_packed_mrope_position_ids(
        inner, ids[None], text_pos[None], image_grid_thw=grids
    )
    assert torch.equal(with_types, without)


def _pixel_values(grids):
    patches = int(sum(g[0] * g[1] * g[2] for g in grids.tolist()))
    return torch.randn(patches, 3 * 2 * 14 * 14)


@torch.no_grad()
def test_hook_isolates_packed_samples(model):
    torch.manual_seed(1)
    rows = [_row(3, (1, 4, 4), 2), _row(2, (1, 6, 4), 3)]
    ids, types, text_pos, grids = _pack(rows)
    pixels = _pixel_values(grids)
    batch = dict(
        input_ids=ids[None],
        position_ids=text_pos[None],
        mm_token_type_ids=types[None],
        image_grid_thw=grids,
        pixel_values=pixels,
        use_cache=False,
    )
    n0 = len(rows[0]["input_ids"])
    leaked = model(**batch).logits[0, n0:]

    seen = {}
    spy = model.model.language_model.register_forward_pre_hook(
        lambda m, a, k: seen.update(shape=tuple(k["position_ids"].shape)),
        with_kwargs=True,
    )
    assert patch_mrope_packing(model)
    assert patch_mrope_packing(model)  # idempotent
    isolated = model(**batch).logits[0, n0:]
    spy.remove()
    assert seen["shape"] == (4, 1, len(ids))

    r = rows[1]
    solo = model(
        input_ids=r["input_ids"][None],
        mm_token_type_ids=r["mm_token_type_ids"][None],
        image_grid_thw=r["image_grid_thw"],
        pixel_values=pixels[grids[0].prod() :],
        use_cache=False,
    ).logits[0]
    assert torch.allclose(isolated, solo, atol=1e-5)
    assert not torch.allclose(leaked, solo, atol=1e-3)


def test_patch_skips_models_without_mrope():
    assert find_mrope_model(torch.nn.Linear(2, 2)) is None
    assert patch_mrope_packing(torch.nn.Linear(2, 2)) is False
