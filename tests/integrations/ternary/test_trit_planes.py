"""The PTQTP-style free-sum grid: `weight_scale: trit_planes`.

Two trit planes summed per weight, `w = t1·s1 + t2·s2` with `s1 >= s2 >= 0`, so a row
spans up to nine values `{0, ±s2, ±(s1-s2), ±s1, ±(s1+s2)}`. `dual` is the constrained
sibling — the same two planes with at most one of them non-zero.
"""

import pydantic
import pytest
import torch
from torch import nn

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.args import TernaryConfig
from axolotl.integrations.ternary.export import bake
from axolotl.integrations.ternary.modules import TernaryLinear
from axolotl.integrations.ternary.ptq.ternary_fit import dequantize_fit, fit_ternary

DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _grid(scale_1: float, scale_2: float) -> set[float]:
    return {
        round(t1 * scale_1 + t2 * scale_2, 6) for t1 in (-1, 0, 1) for t2 in (-1, 0, 1)
    }


def _brute_force(weight: torch.Tensor, first: torch.Tensor, second: torch.Tensor):
    """Nearest of the nine values, found by materializing every candidate."""
    candidates = torch.stack(
        [t1 * first + t2 * second for t1 in (-1, 0, 1) for t2 in (-1, 0, 1)]
    ).expand(9, *weight.shape)
    nearest = (candidates - weight).abs().argmin(0, keepdim=True)
    return candidates.gather(0, nearest)[0]


# ------------------------------------------------------------------- codebook


def test_the_grid_spans_nine_values():
    first = torch.full((1, 1), 3.0)
    second = torch.full((1, 1), 1.0)
    row = torch.tensor([[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]])

    planes = quant.trit_plane_codes(row, first, second)
    values = quant.dequantize_trit_plane_codes(planes, first, second, torch.float32)

    assert torch.equal(values, row)
    assert _grid(3.0, 1.0) == {0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0}
    assert set(torch.unique(planes).tolist()) <= {-1, 0, 1}


@pytest.mark.parametrize("shape", [(4, 32), (8, 64), (3, 17), (1, 5)])
def test_the_assignment_is_the_nearest_of_the_nine(shape):
    torch.manual_seed(shape[0] * shape[1])
    weight = torch.randn(*shape)
    first, second = quant.trit_plane_absmean_scales(weight)

    planes = quant.trit_plane_codes(weight, first, second)
    values = quant.dequantize_trit_plane_codes(planes, first, second, torch.float32)

    assert torch.equal(values, _brute_force(weight, first, second))


def test_ties_take_the_smaller_magnitude():
    """The discipline `round_half_even` gives the single-plane grid at `0.5·s`."""
    first = torch.full((1, 1), 4.0)
    second = torch.full((1, 1), 1.0)
    # 0.5 sits between 0 and s2=1; 3.5 between s1-s2=3 and s1=4
    row = torch.tensor([[0.5, 3.5, -0.5, -3.5]])

    values = quant.dequantize_trit_plane_codes(
        quant.trit_plane_codes(row, first, second), first, second, torch.float32
    )

    assert values.tolist() == [[0.0, 3.0, -0.0, -3.0]]


def test_the_scales_are_ordered_and_floored():
    weight = torch.randn(4, 16)
    small = torch.full((4, 1), 0.25)
    large = torch.full((4, 1), 2.0)

    forward = quant.trit_plane_codes(weight, large, small)
    swapped = quant.trit_plane_codes(weight, small, large)

    assert torch.equal(forward, swapped)


def test_the_free_sum_beats_the_five_value_grid():
    """`dual` is this grid with the same-sign and opposite-sign states removed."""
    torch.manual_seed(0)
    weight = torch.randn(16, 128)

    free_planes, free = fit_ternary(weight, "trit_planes")
    five_planes, five = fit_ternary(weight, "dual")

    free_error = float(
        (
            (weight - dequantize_fit(free_planes, free, scale_mode="trit_planes")) ** 2
        ).sum()
    )
    five_error = float(((weight - dequantize_fit(five_planes, five)) ** 2).sum())
    assert free_error < five_error


# ------------------------------------------------------------- bake and reload


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(8, 64), (5, 33)])
def test_a_baked_master_is_a_fixed_point(dtype, shape):
    torch.manual_seed(shape[1])
    weight = torch.randn(*shape).to(dtype)

    baked = quant.fake_quant_weight_trit_planes(weight, 1.0)
    recovered = quant.baked_trit_plane_codes_and_scales(baked)

    assert recovered is not None
    planes, first, second = recovered
    assert torch.equal(
        quant.dequantize_trit_plane_codes(planes, first, second, dtype), baked
    )
    # and re-baking with the recovered pair changes nothing
    assert torch.equal(
        quant.fake_quant_weight_trit_planes(baked, 1.0, first, second), baked
    )


def test_a_latent_weight_is_not_mistaken_for_a_master():
    torch.manual_seed(0)

    assert quant.baked_trit_plane_codes_and_scales(torch.randn(8, 64)) is None


@pytest.mark.parametrize(
    "name,row",
    [
        ("all zeros", [0.0, 0.0, 0.0, 0.0]),
        ("s2 unused", [0.0, 2.0, -2.0, 2.0]),
        ("s1 == s2", [0.0, 1.0, 2.0, -1.0, -2.0]),
        ("every state", [0.0, 1.0, 2.0, 3.0, 4.0, -4.0, -1.0]),
        ("same-sign state unused", [0.0, 1.0, 3.0, -3.0, 2.0]),
        ("only sum and difference", [0.0, 4.0, 2.0, -4.0]),
        ("one magnitude", [0.0, 5.0, -5.0]),
    ],
)
def test_degenerate_rows_recover(name, row):
    """The pair is over-determined, so a row that skips states still has to resolve."""
    values = torch.tensor([row], dtype=torch.float32)

    recovered = quant.baked_trit_plane_codes_and_scales(values)

    assert recovered is not None, name
    planes, first, second = recovered
    assert torch.equal(
        quant.dequantize_trit_plane_codes(planes, first, second, torch.float32), values
    )
    assert float(first.detach()) >= float(second.detach()) >= 0.0


def test_rows_off_every_grid_are_refused():
    """No `(s1, s2)` reproduces these, so the master is unbakeable rather than wrong."""
    values = torch.tensor([[0.0, 1.0, 2.5, 7.3, 0.11]])

    assert quant.baked_trit_plane_codes_and_scales(values) is None


def test_mixed_rows_are_refused_as_a_whole():
    good = torch.tensor([0.0, 1.0, 3.0, 4.0, 2.0])
    bad = torch.tensor([0.0, 1.0, 2.5, 7.3, 0.11])

    assert quant.baked_trit_plane_codes_and_scales(torch.stack([good, bad])) is None
    assert (
        quant.baked_trit_plane_codes_and_scales(torch.stack([good, good])) is not None
    )


# --------------------------------------------------------------- module wiring


def _module(rows: int = 8, cols: int = 32, **kwargs) -> TernaryLinear:
    torch.manual_seed(0)
    return TernaryLinear.from_linear(
        nn.Linear(cols, rows, bias=False), weight_scale="trit_planes", **kwargs
    )


def test_the_module_carries_two_per_row_scale_parameters():
    module = _module()

    assert module.scale.shape == (8, 1)
    assert module.scale_lo.shape == (8, 1)
    assert set(module.state_dict()) == {"weight", "scale", "scale_lo"}


def test_the_straight_through_estimator_reaches_both_planes():
    module = _module()
    x = torch.randn(4, 32)

    module(x).sum().backward()

    assert module.weight.grad is not None
    assert float(module.scale.grad.abs().sum()) > 0.0
    assert float(module.scale_lo.grad.abs().sum()) > 0.0


def test_the_snapshot_packs_two_ternary_planes():
    module = _module()

    packed = module.code_snapshot()

    assert module.code_count() == module.weight.numel() * quant.DUAL_PLANES
    assert packed.numel() == (module.code_count() + 3) // 4


def test_the_snapshot_zero_fraction_counts_zeroed_weights():
    """Both lanes of a weight are zero exactly when the weight is."""
    module = _module()
    planes = quant.trit_plane_codes(
        module.weight.detach(), *module._trit_plane_scales(module.weight.detach())
    )
    unpacked = quant.unpack_codes(module.code_snapshot(), tuple(planes.shape))

    assert torch.equal(unpacked, planes)
    both_zero = (unpacked[0] == 0) & (unpacked[1] == 0)
    assert torch.equal(both_zero, (planes[0] == 0) & (planes[1] == 0))


def test_bake_folds_the_scales_and_drops_them():
    module = _module()
    x = torch.randn(4, 32)
    with torch.no_grad():
        before = module(x).clone()

    module._post_training(None, "proj")

    assert module.scale is None and module.scale_lo is None
    with torch.no_grad():
        assert torch.equal(module(x), before)
    assert set(module.state_dict()) == {"weight"}


# ------------------------------------------------------------- export plumbing


@pytest.mark.parametrize("dtype", DTYPES)
def test_the_export_derive_path_round_trips(dtype):
    torch.manual_seed(0)
    baked = quant.fake_quant_weight_trit_planes(torch.randn(8, 64).to(dtype), 1.0)

    codes, scale = bake.derive_codes_and_scale(baked, None, "trit_planes")

    assert codes.shape == (quant.DUAL_PLANES, 8, 64)
    assert scale.shape == (8, 2)
    assert torch.equal(
        bake.dequantize_derived(codes, scale, "trit_planes", dtype), baked
    )


def test_baking_a_baked_master_is_a_no_op():
    torch.manual_seed(0)
    baked = quant.fake_quant_weight_trit_planes(torch.randn(8, 64), 1.0)

    assert torch.equal(bake.bake_weight(baked, None, "trit_planes"), baked)


def test_the_export_baker_refuses_a_latent():
    with pytest.raises(ValueError, match="reached the export baker as a latent"):
        bake.bake_weight(torch.randn(8, 64), None, "trit_planes")


def test_deriving_from_a_latent_raises():
    with pytest.raises(ValueError, match="not baked"):
        bake.derive_codes_and_scale(torch.randn(8, 64), None, "trit_planes")


# ------------------------------------------------------------------- the schema


def test_the_schema_accepts_a_master_only_export():
    config = TernaryConfig(
        weight_scale="trit_planes", export={"formats": ["master_bf16"]}
    )

    assert config.weight_scale == "trit_planes"


@pytest.mark.parametrize("fmt", ["hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s"])
def test_the_schema_rejects_every_packed_format(fmt):
    with pytest.raises(pydantic.ValidationError, match="cannot be represented"):
        TernaryConfig(
            weight_scale="trit_planes", export={"formats": ["master_bf16", fmt]}
        )


def test_the_schema_rejects_a_group_size():
    with pytest.raises(pydantic.ValidationError, match="only valid with"):
        TernaryConfig(
            weight_scale="trit_planes",
            group_size=128,
            export={"formats": ["master_bf16"]},
        )


@pytest.mark.parametrize("init", ["ternary_fit", "ternary_fit_calibrated"])
def test_the_schema_allows_the_fitting_inits(init):
    config = TernaryConfig(
        weight_scale="trit_planes",
        init=init,
        lambda_schedule="none",
        export={"formats": ["master_bf16"]},
    )

    assert config.init == init


# ----------------------------------------------------------------- the PTQ fit


def test_the_fit_returns_two_planes_and_a_pair_per_row():
    torch.manual_seed(0)
    weight = torch.randn(16, 128)

    planes, scales = fit_ternary(weight, "trit_planes")

    assert planes.shape == (quant.DUAL_PLANES, 16, 128)
    assert scales.shape == (16, 2)
    assert set(torch.unique(planes).tolist()) <= {-1, 0, 1}


def test_the_fit_beats_the_absmean_seed():
    torch.manual_seed(0)
    weight = torch.randn(16, 128)
    seed_1, seed_2 = quant.trit_plane_absmean_scales(weight)

    planes, scales = fit_ternary(weight, "trit_planes")

    fitted = float(
        ((weight - dequantize_fit(planes, scales, scale_mode="trit_planes")) ** 2).sum()
    )
    seeded = float(
        (
            (weight - quant.fake_quant_weight_trit_planes(weight, 1.0, seed_1, seed_2))
            ** 2
        ).sum()
    )
    assert fitted < seeded


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_fit_writes_a_recoverable_master(dtype):
    """`write_latent` puts the reconstruction in the latent; the module reads it back."""
    from axolotl.integrations.ternary.ptq import write_latent

    torch.manual_seed(0)
    module = TernaryLinear.from_linear(
        nn.Linear(128, 16, bias=False).to(dtype), weight_scale="trit_planes"
    )
    planes, scales = fit_ternary(module.weight.detach().float(), "trit_planes")

    write_latent(module, planes, scales)

    assert module.baked
    latent = module.weight.detach()
    first, second = module._trit_plane_scales(latent)
    # the seeded parameters assign every weight the state the fit gave it
    assert torch.equal(
        quant.trit_plane_codes(latent, first, second),
        quant.baked_trit_plane_codes_and_scales(latent)[0],
    )
    rebaked = module._bake(latent, gathered=False)
    if dtype is torch.bfloat16:
        # the master's own dtype: the grid is closed, so the bake is the identity
        assert torch.equal(rebaked, latent)
    else:
        # in fp32 the `log`/`exp` parametrization moves a summed value by an ulp
        assert float((rebaked - latent).detach().abs().max()) < 1e-6


def test_a_heavy_tailed_row_uses_the_combination_states():
    """The states `dual` lacks are the point: `±(s1 ± s2)` covers the tail."""
    torch.manual_seed(3)
    weight = torch.randn(4, 256)
    weight[:, :8] *= 6.0

    planes, _ = fit_ternary(weight, "trit_planes")

    both_nonzero = ((planes[0] != 0) & (planes[1] != 0)).sum()
    assert int(both_nonzero) > 0
