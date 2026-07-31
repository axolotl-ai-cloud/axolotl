"""IQ1_S block packing against llama.cpp's exact decode arithmetic."""

import pytest
import torch

from axolotl.integrations.ternary.export.gguf_tq import (
    IQ1S_BLOCK_BYTES,
    IQ1S_DELTA,
    QK_K,
    decode_iq1s,
    pack_iq1s,
)
from axolotl.integrations.ternary.iq1s_grid import GRID_DIM, GRID_ENTRIES, grid_tensor


def _grid_codes(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    picks = torch.randint(0, GRID_ENTRIES, (rows, cols // GRID_DIM))
    return grid_tensor()[picks].reshape(rows, cols).to(torch.int8)


def test_block_size_arithmetic():
    assert IQ1S_BLOCK_BYTES == 50
    codes = _grid_codes(2, QK_K)
    packed = pack_iq1s(codes, torch.tensor(0.05))
    assert packed.numel() == 2 * IQ1S_BLOCK_BYTES
    assert packed.dtype == torch.uint8


def test_roundtrip_matches_shifted_master():
    codes = _grid_codes(4, 2 * QK_K, seed=1)
    scale = torch.tensor(0.037)
    packed = pack_iq1s(codes, scale)
    decoded = decode_iq1s(packed, (4, 2 * QK_K))

    # d survives an f16 round-trip of scale/7; recover the exact dl the decoder saw
    d_f16 = (scale.float() / 7).to(torch.float16).float()
    dl = d_f16 * 7

    groups = codes.reshape(4, -1, 32)
    sigma = torch.where(groups.sum(-1, keepdim=True) > 0, -1.0, 1.0)
    expected = dl * (groups.float() + sigma * IQ1S_DELTA)
    assert torch.allclose(decoded, expected.reshape(4, -1), atol=1e-7)


def test_shift_is_uniform_per_group_and_bounded():
    codes = _grid_codes(2, QK_K, seed=2)
    scale = torch.tensor(0.05)
    decoded = decode_iq1s(pack_iq1s(codes, scale), (2, QK_K))

    d_f16 = (scale.float() / 7).to(torch.float16).float()
    error = (decoded - d_f16 * 7 * codes.float()).reshape(2, -1, 32)
    per_group = error / (d_f16 * 7)
    assert torch.allclose(per_group.abs(), torch.full_like(per_group, IQ1S_DELTA))
    assert torch.all(per_group.amax(-1) == per_group.amin(-1))


def test_non_grid_pattern_refused():
    from axolotl.integrations.ternary.iq1s_grid import pattern_index_table

    table = pattern_index_table()
    key = int((table < 0).nonzero()[0])
    digits = [(key // 3**i) % 3 - 1 for i in range(GRID_DIM)]

    codes = _grid_codes(1, QK_K, seed=3)
    codes[0, :GRID_DIM] = torch.tensor(digits, dtype=torch.int8)
    with pytest.raises(ValueError, match="grid patterns"):
        pack_iq1s(codes, torch.tensor(0.05))


def test_ragged_width_refused():
    with pytest.raises(ValueError, match="in_features"):
        pack_iq1s(_grid_codes(1, QK_K)[:, : QK_K - 8], torch.tensor(0.05))
