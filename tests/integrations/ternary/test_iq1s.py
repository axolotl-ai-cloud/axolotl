"""The IQ1_S codebook: healing into a stock llama.cpp grid (task #15)."""

import pytest
import torch
from torch import nn

from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.iq1s import (
    fake_quant_weight_iq1s,
    fake_quant_weight_iq1s_ste,
    project_codes,
)
from axolotl.integrations.ternary.iq1s_grid import (
    GRID_DIM,
    GRID_ENTRIES,
    grid_tensor,
    pattern_index_table,
)
from axolotl.integrations.ternary.modules import TernaryLinear


def test_the_grid_decodes_to_the_published_shape():
    grid = grid_tensor()
    assert grid.shape == (GRID_ENTRIES, GRID_DIM)
    assert set(grid.unique().tolist()) == {-1, 0, 1}
    zero_fraction = float((grid == 0).float().mean())
    assert abs(zero_fraction - 0.406) < 0.01


def test_the_index_table_round_trips_every_grid_pattern():
    grid = grid_tensor().long() + 1
    powers = torch.tensor([3**i for i in range(GRID_DIM)], dtype=torch.long)
    keys = (grid * powers).sum(dim=1)
    assert torch.equal(pattern_index_table()[keys], torch.arange(GRID_ENTRIES))


def test_projection_matches_brute_force():
    torch.manual_seed(0)
    weight = torch.randn(16, 32)
    scale = weight.abs().mean()
    codes = project_codes(weight, scale)

    grid = grid_tensor().float()
    normalized = (weight / scale).reshape(-1, GRID_DIM)
    distances = torch.cdist(normalized, grid)
    brute = grid[distances.argmin(dim=1)].reshape(weight.shape)
    assert torch.equal(codes.float(), brute)


def test_every_projected_group_is_a_grid_pattern():
    torch.manual_seed(1)
    weight = torch.randn(8, 64)
    codes = project_codes(weight, weight.abs().mean()).reshape(-1, GRID_DIM)
    table = pattern_index_table()
    powers = torch.tensor([3**i for i in range(GRID_DIM)], dtype=torch.long)
    keys = ((codes.long() + 1) * powers).sum(dim=1)
    assert int((table[keys] < 0).sum()) == 0


def test_ste_passes_weight_grad_and_scales_get_code_sums():
    torch.manual_seed(2)
    weight = nn.Parameter(torch.randn(4, 16))
    scale = nn.Parameter(torch.ones(1))
    out = fake_quant_weight_iq1s_ste(weight, 1.0, scale)
    out.sum().backward()
    assert weight.grad is not None and torch.equal(weight.grad, torch.ones_like(weight))
    codes = project_codes(weight.detach(), scale.detach())
    assert torch.allclose(scale.grad, codes.sum().reshape(1))


def test_module_forward_and_bake_are_the_projection():
    torch.manual_seed(3)
    linear = nn.Linear(32, 8, bias=False)
    module = TernaryLinear.from_linear(
        linear, weight_scale="learnable", codebook="iq1s", activation_bits=None
    )
    module.lambda_ = 1.0
    x = torch.eye(32)
    scale = module._scale()
    expected = (project_codes(module.weight.detach(), scale.detach()) * scale).to(
        module.weight.dtype
    )
    assert torch.allclose(module(x).T, expected, atol=1e-6)

    baked = module.baked_weight()
    assert torch.allclose(baked, expected.detach(), atol=1e-6)
    with torch.no_grad():
        module.weight.copy_(baked)
    assert torch.equal(module.baked_weight(), baked)


def test_snapshot_packs_projected_codes():
    torch.manual_seed(4)
    linear = nn.Linear(16, 4, bias=False)
    module = TernaryLinear.from_linear(
        linear, weight_scale="absmean", codebook="iq1s", activation_bits=None
    )
    snapshot = module.code_snapshot()
    assert snapshot.numel() == (16 * 4 + 3) // 4


def test_config_validates_iq1s_pairings():
    TernaryConfig(codebook="iq1s", weight_scale="learnable")
    with pytest.raises(ValueError, match="per-tensor"):
        TernaryConfig(codebook="iq1s", weight_scale="dual")
    with pytest.raises(ValueError, match="group_size"):
        TernaryConfig(codebook="iq1s", weight_scale="absmean", group_size=64)


def test_indivisible_width_refuses():
    with pytest.raises(ValueError, match="divisible"):
        project_codes(torch.randn(4, 12), torch.tensor(1.0))


def test_partial_lambda_blends():
    torch.manual_seed(5)
    weight = torch.randn(4, 16)
    scale = weight.abs().mean()
    full = fake_quant_weight_iq1s(weight, 1.0, scale)
    half = fake_quant_weight_iq1s(weight, 0.5, scale)
    assert torch.allclose(half, (weight + full) / 2, atol=1e-6)
