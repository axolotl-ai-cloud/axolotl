"""CPU-only tests for the data-free ternary fit and the `ternary.init` pass."""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.modules import TernaryLinear, iter_ternary_modules
from axolotl.integrations.ternary.ptq import (
    initialize_model_latents,
    write_latent,
)
from axolotl.integrations.ternary.ptq.ternary_fit import (
    MAX_ITERS,
    RIDGE_KAPPA_TRIGGER,
    RIDGE_MAX,
    _adaptive_ridge,
    dequantize_fit,
    fit_ternary,
    solve_2x2_ridge,
    solve_scale,
)
from axolotl.integrations.ternary.swap import convert_model
from axolotl.utils.dict import DictDefault

SINGLE_PLANE_MODES = [
    ("absmean", None),
    ("learnable", None),
    ("group", 32),
    ("learnable_row", None),
]


def _normal(rows: int = 48, cols: int = 96, seed: int = 0) -> torch.Tensor:
    return torch.randn(rows, cols, generator=torch.Generator().manual_seed(seed))


def _heavy_tailed(rows: int = 48, cols: int = 96, seed: int = 0) -> torch.Tensor:
    """Gaussian scale mixture — the outlier-heavy shape real projections have."""
    gen = torch.Generator().manual_seed(seed)
    spread = torch.rand(rows, 1, generator=gen).clamp_min(0.02).reciprocal()
    return torch.randn(rows, cols, generator=gen) * spread


def _reconstruction_error(
    weight: torch.Tensor, codes: torch.Tensor, scale: torch.Tensor
) -> float:
    return float((weight - dequantize_fit(codes, scale)).norm())


def _absmean_error(weight: torch.Tensor, group_size: int | None = None) -> float:
    scale = quant.absmean_scale(weight, group_size)
    codes = quant.ternary_codes(weight, scale)
    baseline = quant.dequantize_codes(
        codes, quant.f16_round_scale(scale), torch.float32
    )
    return float((weight - baseline).norm())


def _fit_cfg(init: str = "ternary_fit") -> DictDefault:
    """A fitting init needs a scale mode that keeps the scale it solved."""
    return DictDefault({"ternary": {"init": init, "weight_scale": "learnable"}})


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)


def _linear_module(
    weight: torch.Tensor,
    weight_scale: str = "absmean",
    group_size: int | None = None,
    activation_bits: int | None = 8,
    dtype: torch.dtype = torch.float32,
) -> TernaryLinear:
    module = TernaryLinear(
        weight.shape[1],
        weight.shape[0],
        weight_scale=weight_scale,
        group_size=group_size,
        activation_bits=activation_bits,
        fused=False,
        dtype=dtype,
    )
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


# ------------------------------------------------------------------ fit layout


@pytest.mark.parametrize("scale_mode,group_size", SINGLE_PLANE_MODES)
def test_single_plane_fit_returns_the_canonical_layout(scale_mode, group_size):
    weight = _normal()

    codes, scale = fit_ternary(weight, scale_mode, group_size)

    assert codes.shape == weight.shape
    assert codes.dtype == torch.int8
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scale.dtype == torch.float32
    expected = {
        "absmean": (),
        "learnable": (),
        "learnable_row": (weight.shape[0], 1),
        "group": (weight.shape[0], weight.shape[1] // (group_size or 1)),
    }[scale_mode]
    assert tuple(scale.shape) == expected
    assert bool((scale >= quant.SCALE_EPS).all())


def test_dual_fit_returns_two_disjoint_planes_and_ordered_scales():
    weight = _normal()

    codes, scales = fit_ternary(weight, "dual")

    assert codes.shape == (2, *weight.shape)
    assert codes.dtype == torch.int8
    assert scales.shape == (weight.shape[0], 2)
    assert not bool(((codes[0] != 0) & (codes[1] != 0)).any())
    assert bool((scales[:, 0] <= scales[:, 1]).all())


# ------------------------------------------------------------- fixed-point law


@pytest.mark.parametrize("scale_mode,group_size", SINGLE_PLANE_MODES)
def test_fit_terminates_at_a_fixed_point_of_the_canonical_quantizer(
    scale_mode, group_size
):
    """Re-rounding the original weight with the fitted scale must not move a code."""
    weight = _heavy_tailed()

    codes, scale = fit_ternary(weight, scale_mode, group_size)

    canonical = quant.ternary_codes(weight, quant.f16_round_scale(scale))
    assert torch.equal(canonical, codes)


def test_dual_fit_terminates_on_the_nearest_of_five_states():
    weight = _heavy_tailed()

    codes, scales = fit_ternary(weight, "dual")

    levels = quant.f16_round_scale(scales)
    magnitude = weight.abs()
    nearest = torch.stack(
        [
            magnitude,
            (magnitude - levels[:, :1]).abs(),
            (magnitude - levels[:, 1:]).abs(),
        ]
    ).amin(0)

    achieved = (weight - dequantize_fit(codes, scales)).abs()
    assert torch.allclose(achieved, nearest, atol=1e-6)


def test_fitted_scale_survives_the_f16_dequant_scale():
    weight = _normal()

    codes, scale = fit_ternary(weight, "absmean")

    latent = dequantize_fit(codes, scale, torch.float32)
    recovered = quant.baked_codes_and_scale(latent)
    assert recovered is not None
    assert torch.equal(recovered[0], codes)
    assert torch.allclose(recovered[1], quant.f16_round_scale(scale))


# ------------------------------------------------------------------- quality


@pytest.mark.parametrize("scale_mode,group_size", SINGLE_PLANE_MODES)
@pytest.mark.parametrize("sampler", [_normal, _heavy_tailed])
def test_fit_beats_absmean_rounding_on_frobenius_error(scale_mode, group_size, sampler):
    weight = sampler(seed=3)

    codes, scale = fit_ternary(weight, scale_mode, group_size)

    assert _reconstruction_error(weight, codes, scale) < _absmean_error(
        weight, group_size
    )


@pytest.mark.parametrize("sampler", [_normal, _heavy_tailed])
def test_dual_beats_the_single_plane_fit(sampler):
    weight = sampler(seed=5)

    single = fit_ternary(weight, "learnable_row")
    dual = fit_ternary(weight, "dual")

    assert _reconstruction_error(weight, *dual) < _reconstruction_error(weight, *single)


@pytest.mark.parametrize("scale_mode", ["absmean", "learnable_row", "group", "dual"])
def test_the_alternation_never_increases_the_objective(scale_mode):
    weight = _heavy_tailed(seed=7)
    group_size = 32 if scale_mode == "group" else None

    errors = [
        _reconstruction_error(
            weight, *fit_ternary(weight, scale_mode, group_size, max_iters=iterations)
        )
        for iterations in range(1, 9)
    ]

    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < errors[0]


def test_the_fit_converges_well_inside_the_iteration_cap():
    weight = _normal(seed=11)

    early = fit_ternary(weight, "absmean", max_iters=MAX_ITERS)
    late = fit_ternary(weight, "absmean", max_iters=4 * MAX_ITERS)

    assert torch.equal(early[0], late[0])
    assert torch.equal(early[1], late[1])


# ------------------------------------------------------------------ degenerate


@pytest.mark.parametrize("scale_mode,group_size", SINGLE_PLANE_MODES)
def test_an_all_zero_weight_fits_to_zero_codes(scale_mode, group_size):
    weight = torch.zeros(16, 64)

    codes, scale = fit_ternary(weight, scale_mode, group_size)

    assert not bool(codes.any())
    assert bool((scale == quant.SCALE_EPS).all())


def test_an_all_zero_weight_fits_to_zero_codes_in_dual_mode():
    codes, scales = fit_ternary(torch.zeros(16, 64), "dual")

    assert not bool(codes.any())
    assert bool(torch.isfinite(scales).all())


def test_a_constant_magnitude_weight_keeps_every_code():
    weight = torch.full((8, 32), 0.5)
    weight[:, ::2] *= -1

    codes, scale = fit_ternary(weight, "absmean")

    assert bool((codes.abs() == 1).all())
    assert _reconstruction_error(weight, codes, scale) < 1e-2


def test_the_ridge_collapses_an_empty_plane_instead_of_dividing_by_zero():
    counts = torch.tensor([64.0])
    empty = torch.zeros(1)

    solution = solve_2x2_ridge(counts, empty, empty, torch.tensor([32.0]), empty, 1e-8)

    assert bool(torch.isfinite(solution).all())
    assert float(solution[0, 1]) == 0.0
    # a singular system caps the ridge at RIDGE_MAX of the largest eigenvalue, so the
    # surviving plane is biased by exactly that fraction and nothing more
    assert float(solution[0, 0]) == pytest.approx(32.0 / (64.0 * (1 + RIDGE_MAX)))


def test_the_ridge_is_only_boosted_once_the_condition_number_trips_it():
    ones = torch.ones(4)
    partners = torch.tensor([1.0, 10.0 / RIDGE_KAPPA_TRIGGER, 1e-10, 0.0])

    ridge = _adaptive_ridge(ones, torch.zeros(4), partners, 1e-8)

    assert float(ridge[0]) == pytest.approx(1e-8)
    assert float(ridge[1]) == pytest.approx(1e-8)
    assert float(ridge[2]) == pytest.approx(1e-6, rel=1e-2)
    assert float(ridge[3]) == pytest.approx(RIDGE_MAX)


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_the_ridge_is_relative_so_the_solve_is_scale_equivariant(factor):
    """The same solver takes code counts and activation-weighted Grams; only a ridge
    relative to the system is inert in neither."""
    system = [torch.tensor(value) for value in ([9.0], [3.0], [1.0], [6.0], [2.0])]

    plain = solve_2x2_ridge(*system, 1e-8)
    scaled = solve_2x2_ridge(*(term * factor for term in system), 1e-8)

    assert torch.allclose(plain, scaled, rtol=1e-4)


def test_a_well_conditioned_system_is_solved_exactly():
    solution = solve_2x2_ridge(
        torch.tensor([4.0]),
        torch.tensor([0.0]),
        torch.tensor([4.0]),
        torch.tensor([2.0]),
        torch.tensor([6.0]),
        1e-8,
    )

    assert float(solution[0, 0]) == pytest.approx(0.5, rel=1e-6)
    assert float(solution[0, 1]) == pytest.approx(1.5, rel=1e-6)


# ------------------------------------------------------------------ validation


def test_fit_rejects_a_group_size_the_mode_does_not_use():
    with pytest.raises(ValueError, match="only valid with"):
        fit_ternary(_normal(), "absmean", 32)


def test_fit_rejects_a_group_mode_without_a_group_size():
    with pytest.raises(ValueError, match="requires a group_size"):
        fit_ternary(_normal(), "group")


def test_fit_rejects_a_group_size_that_does_not_divide_the_row():
    with pytest.raises(ValueError, match="does not divide"):
        fit_ternary(_normal(), "group", 7)


def test_fit_rejects_an_unknown_scale_mode():
    with pytest.raises(ValueError, match="unknown ternary weight_scale mode"):
        fit_ternary(_normal(), "magic")


def test_fit_rejects_a_non_matrix_weight():
    with pytest.raises(ValueError, match="in_features"):
        fit_ternary(torch.randn(8), "absmean")


# ----------------------------------------------------------------- solve_scale


@pytest.mark.parametrize("scale_mode,group_size", SINGLE_PLANE_MODES)
def test_solve_scale_is_the_least_squares_scale_for_fixed_codes(scale_mode, group_size):
    weight = _normal()
    codes, scale = fit_ternary(weight, scale_mode, group_size)

    resolved = solve_scale(weight, codes, scale_mode, group_size)

    assert resolved.shape == scale.shape
    assert torch.allclose(resolved, scale, rtol=1e-3)


def test_solve_scale_falls_back_to_absmean_when_no_code_is_set():
    weight = _normal(rows=4, cols=16)

    resolved = solve_scale(weight, torch.zeros_like(weight, dtype=torch.int8))

    assert torch.allclose(resolved, quant.absmean_scale(weight))


def test_solve_scale_pairs_the_dual_scales_with_the_planes_not_with_each_other():
    """Sorting the pair without permuting the planes reconstructs something else."""
    weight = torch.tensor([[0.9, 1.1, 0.1, 0.3]])
    planes = torch.tensor(
        [[[0, 0, 1, 1]], [[1, 1, 0, 0]]], dtype=torch.int8
    )  # plane 0 covers the small weights' partner, plane 1 the large ones

    resolved = solve_scale(weight, planes, "dual")

    assert float(resolved[0, 0]) == pytest.approx(0.2, rel=1e-4)
    assert float(resolved[0, 1]) == pytest.approx(1.0, rel=1e-4)
    error = float(((weight - dequantize_fit(planes, resolved)) ** 2).sum())
    swapped = float(((weight - dequantize_fit(planes, resolved.flip(-1))) ** 2).sum())
    assert error == pytest.approx(0.04, abs=1e-3)
    assert swapped > 50 * error


# ------------------------------------------------------------------ init pass


def test_absmean_init_leaves_every_latent_alone():
    model = _tiny_llama()
    manifest = convert_model(model, DictDefault({"ternary": {}}))
    before = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(model)
    }

    initialize_model_latents(model, manifest, DictDefault({"ternary": {}}))

    for name, module in iter_ternary_modules(model):
        assert torch.equal(module.weight.detach(), before[name])


def test_an_unimplemented_init_mode_is_rejected():
    model = _tiny_llama()
    manifest = convert_model(model, DictDefault({"ternary": {}}))

    with pytest.raises(NotImplementedError, match="svid"):
        initialize_model_latents(
            model,
            manifest,
            DictDefault({"ternary": {"init": "svid", "weight_scale": "learnable"}}),
        )


def test_the_fit_init_writes_ternary_latents_and_flags_them_baked():
    model = _tiny_llama()

    manifest = convert_model(model, _fit_cfg())

    assert manifest.init == "ternary_fit"
    for _, module in iter_ternary_modules(model):
        assert module.is_baked()
        assert quant.baked_codes_and_scale(module.weight.detach()) is not None


def test_the_module_quantizer_reproduces_the_codes_the_fit_ended_on():
    weight = _heavy_tailed(rows=32, cols=64, seed=13)
    module = _linear_module(weight)
    codes, scale = fit_ternary(weight, "absmean")

    write_latent(module, codes, scale)

    latent = module.weight.detach()
    rederived = quant.ternary_codes(latent, quant.absmean_scale(latent))
    assert torch.equal(rederived, codes)


def test_a_learnable_scale_is_initialized_to_the_fitted_scale():
    weight = _normal(rows=16, cols=64, seed=17)
    module = _linear_module(weight, weight_scale="learnable")
    codes, scale = fit_ternary(weight, "learnable")

    write_latent(module, codes, scale)

    assert module.scale is not None
    # the module re-reads its scale off the latent, which holds the f16-rounded one;
    # that is the value the quantizer uses either way, so this is exact, not approx
    stored = module.scale.detach().exp().reshape(())
    assert torch.equal(
        quant.f16_round_scale(stored), quant.f16_round_scale(scale.reshape(()))
    )
    assert float(stored) == pytest.approx(float(scale), rel=2.0**-11)
    effective = quant.fake_quant_weight(module.weight.detach(), 1.0, None, stored)
    assert torch.equal(effective, module.weight.detach())


def test_the_init_pass_skips_modules_that_are_already_baked():
    model = _tiny_llama()
    manifest = convert_model(model, _fit_cfg())
    fitted = {
        name: module.weight.detach().clone()
        for name, module in iter_ternary_modules(model)
    }

    initialize_model_latents(model, manifest, _fit_cfg())

    for name, module in iter_ternary_modules(model):
        assert torch.equal(module.weight.detach(), fitted[name])


def test_the_fit_init_beats_absmean_rounding_end_to_end():
    reference = _tiny_llama()
    tokens = torch.randint(0, 128, (4, 24), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        target = reference(input_ids=tokens).logits

    errors = {}
    for init in ("absmean", "ternary_fit"):
        model = _tiny_llama()
        convert_model(model, _fit_cfg(init))
        with torch.no_grad():
            logits = model(input_ids=tokens).logits
        assert bool(torch.isfinite(logits).all())
        errors[init] = float((logits - target).norm())

    assert errors["ternary_fit"] < errors["absmean"]


def test_the_first_forward_runs_the_weight_the_fit_ended_on():
    """λ warmup has nothing to walk to: the init already is the quantized function."""
    weight = _heavy_tailed(rows=8, cols=32, seed=19)
    module = _linear_module(weight, activation_bits=None)
    codes, scale = fit_ternary(weight, "absmean")
    write_latent(module, codes, scale)
    activations = torch.randn(3, 32, generator=torch.Generator().manual_seed(20))

    output = module(activations)

    assert torch.allclose(
        output, activations @ dequantize_fit(codes, scale).T, atol=1e-5
    )


def test_the_fit_survives_a_bf16_latent():
    weight = _heavy_tailed(rows=8, cols=64, seed=21)
    module = _linear_module(weight, dtype=torch.bfloat16)
    codes, scale = fit_ternary(module.weight.detach(), "absmean")

    write_latent(module, codes, scale)

    latent = module.weight.detach()
    assert latent.dtype == torch.bfloat16
    assert module.is_baked()
    assert torch.equal(quant.ternary_codes(latent, quant.absmean_scale(latent)), codes)


def test_write_latent_keeps_the_parameter_identity_for_a_sharded_swap():
    weight = _normal(rows=8, cols=32)
    module = _linear_module(weight)
    parameter = module.weight

    write_latent(module, *fit_ternary(weight, "absmean"))

    assert module.weight is parameter


# ------------------------------------------------------- the fit has to survive step 1


SCALE_KEEPING_MODES = ["learnable", "learnable_row", "dual"]


@pytest.mark.parametrize("weight_scale", SCALE_KEEPING_MODES)
def test_the_fitted_scale_outlives_the_first_write_to_the_latent(weight_scale):
    """`baked` lapses on the first optimizer step; the solved scale must not go with it."""
    weight = _heavy_tailed(rows=32, cols=64, seed=29)
    module = _linear_module(weight, weight_scale=weight_scale)
    codes, scale = fit_ternary(weight, weight_scale)
    write_latent(module, codes, scale)
    fitted = module.weight.detach().clone()

    with torch.no_grad():
        module.weight.add_(1e-8)

    assert not module.is_baked()
    effective = module._quant_weight(1.0).detach()
    nonzero = fitted != 0
    ratio = effective[nonzero] / fitted[nonzero]
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3)


@pytest.mark.parametrize("weight_scale", ["absmean", "group"])
def test_a_statistic_scale_mode_cannot_carry_a_fit_and_the_schema_says_so(weight_scale):
    """The pathology the validator exists for: absmean(codes*s) is s times the density."""
    weight = _heavy_tailed(rows=32, cols=64, seed=29)
    group_size = 32 if weight_scale == "group" else None
    module = _linear_module(weight, weight_scale=weight_scale, group_size=group_size)
    write_latent(module, *fit_ternary(weight, weight_scale, group_size))
    fitted = module.weight.detach().clone()

    with torch.no_grad():
        module.weight.add_(1e-8)
    effective = module._quant_weight(1.0).detach()

    nonzero = fitted != 0
    shrunk = (effective[nonzero] / fitted[nonzero]).max()
    assert float(shrunk) < 0.99
    with pytest.raises(ValueError, match="keeps no scale"):
        convert_model(
            _tiny_llama(),
            DictDefault(
                {
                    "ternary": {
                        "init": "ternary_fit",
                        "weight_scale": weight_scale,
                        "group_size": group_size,
                        "export": {"formats": ["master_bf16"]},
                    }
                }
            ),
        )


def test_the_fitting_init_is_refused_on_a_statistic_scale_mode():
    from axolotl.integrations.ternary.args import TernaryConfig

    with pytest.raises(ValueError, match="keeps no scale"):
        TernaryConfig(init="ternary_fit", weight_scale="absmean")
    with pytest.raises(ValueError, match="keeps no scale"):
        TernaryConfig(
            init="ternary_fit_calibrated", weight_scale="group", group_size=64
        )


def test_the_fitting_init_warns_that_it_makes_the_lambda_warmup_a_no_op(caplog):
    from axolotl.integrations.ternary.args import TernaryConfig

    with caplog.at_level("WARNING"):
        TernaryConfig(init="ternary_fit", weight_scale="learnable")

    assert "lambda_schedule" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING"):
        TernaryConfig(
            init="ternary_fit", weight_scale="learnable", lambda_schedule="none"
        )
    assert "lambda_schedule" not in caplog.text


# ------------------------------------------------------------------ meta device


def test_a_meta_device_rank_skips_the_fit_instead_of_crashing(caplog):
    """FSDP2's cpu_ram_efficient_loading leaves every rank but 0 holding meta weights."""
    config = _tiny_llama().config
    with torch.device("meta"):
        model = LlamaForCausalLM(config)

    with caplog.at_level("WARNING"):
        convert_model(model, _fit_cfg())

    assert "meta device" in caplog.text
    for _, module in iter_ternary_modules(model):
        assert module.weight.device.type == "meta"


def test_a_meta_device_rank_skips_the_calibrated_fit_before_it_loads_a_corpus():
    config = _tiny_llama().config
    with torch.device("meta"):
        model = LlamaForCausalLM(config)
    cfg = DictDefault(
        {"ternary": {"init": "ternary_fit_calibrated", "weight_scale": "learnable"}}
    )

    manifest = convert_model(model, cfg)

    assert manifest.init == "ternary_fit_calibrated"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_the_fit_agrees_between_cpu_and_cuda():
    weight = _normal(seed=23)

    cpu_codes, cpu_scale = fit_ternary(weight, "absmean")
    cuda_codes, cuda_scale = fit_ternary(weight.cuda(), "absmean")

    assert torch.equal(cpu_codes, cuda_codes.cpu())
    assert torch.allclose(cpu_scale, cuda_scale.cpu(), rtol=1e-6)
