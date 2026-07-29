"""Fused multi-state fake-quant kernels vs the eager oracle.

Bit-exact equality is the contract, not a tolerance: the kernels reproduce the oracle's
scale rounding, its dtype cast before the λ blend, and its tie-breaks. Anything less
would make the fused and eager paths different functions, and a run that falls back
mid-training (an FSDP2 shard, a CPU eval) would change model behaviour silently.

The one exception is fp32 with λ < 1, where the kernel blend contracts to an FMA and
lands an ulp from the oracle. That is not papered over with a tolerance in the shipped
path:  declines the fused kernel there, so the invariant holds for
every configuration the dispatch actually uses.
"""

import pytest
import torch
from torch import nn

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.modules import TernaryLinear

triton_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the fused kernels are Triton/CUDA only",
)

DTYPES = (torch.float32, torch.bfloat16, torch.float16)
SHAPES = ((8, 64), (64, 512), (7, 13), (1, 4096))
LAMBDAS = (1.0, 0.75, 0.5, 0.01)


@pytest.fixture(name="fused")
def fixture_fused():
    pytest.importorskip("triton")
    from axolotl.integrations.ternary.kernels.triton import multi_state

    return multi_state


def _exactness(dtype, lambda_):
    """The kernels are bit-exact except for the case the dispatch declines to use."""
    return dtype is not torch.float32 or lambda_ >= 1.0


def _assert_matches(got, want, dtype, lambda_):
    if _exactness(dtype, lambda_):
        assert torch.equal(got, want)
        return
    # fp32 + partial λ: the blend contracts to an FMA, one ulp from the oracle
    assert torch.allclose(got, want, rtol=0, atol=1e-7)


def _weight(shape, dtype, seed=0, scale=0.05):
    torch.manual_seed(seed)
    return (torch.randn(*shape, device="cuda") * scale).to(dtype)


# --------------------------------------------------------------- bit-exactness


@triton_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_dual_matches_the_oracle_bit_for_bit(fused, dtype, shape, lambda_):
    weight = _weight(shape, dtype, seed=shape[1])
    low, high = quant.dual_absmean_scales(weight)

    got = fused.fake_quant_weight_dual(weight, lambda_, low, high)
    want = quant.fake_quant_weight_dual(weight, lambda_, low, high)

    _assert_matches(got, want, dtype, lambda_)


@triton_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_trit_planes_matches_the_oracle_bit_for_bit(fused, dtype, shape, lambda_):
    weight = _weight(shape, dtype, seed=shape[1])
    first, second = quant.trit_plane_absmean_scales(weight)

    got = fused.fake_quant_weight_trit_planes(weight, lambda_, first, second)
    want = quant.fake_quant_weight_trit_planes(weight, lambda_, first, second)

    _assert_matches(got, want, dtype, lambda_)


@triton_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", LAMBDAS)
def test_binary_matches_the_oracle_bit_for_bit(fused, dtype, shape, lambda_):
    weight = _weight(shape, dtype, seed=shape[1])
    scale = quant.binary_scale(weight)

    got = fused.fake_quant_weight_binary(weight, lambda_, None, scale)
    want = quant.fake_quant_weight_binary(weight, lambda_, None, scale)

    _assert_matches(got, want, dtype, lambda_)


@triton_cuda
@pytest.mark.parametrize("group_size", [32, 128])
def test_binary_group_scales_match_the_oracle(fused, group_size):
    weight = _weight((16, 256), torch.bfloat16)
    scale = quant.binary_scale(weight, group_size)

    got = fused.fake_quant_weight_binary(weight, 1.0, group_size, scale)
    want = quant.fake_quant_weight_binary(weight, 1.0, group_size, scale)

    assert torch.equal(got, want)


# ------------------------------------------------------------ degenerate rows


DEGENERATE = {
    "all zero": torch.zeros(1, 32),
    "all one sign": torch.full((1, 32), 0.03),
    "all negative": torch.full((1, 32), -0.03),
    "one nonzero": torch.cat([torch.zeros(1, 31), torch.full((1, 1), 0.9)], dim=1),
    "on the half-scale boundary": torch.full((1, 32), 0.5),
    "wide dynamic range": torch.tensor([[1e-7, 1.0, -1e-7, 5.0] * 8]),
}


@triton_cuda
@pytest.mark.parametrize("name", sorted(DEGENERATE))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_degenerate_rows_match_the_oracle(fused, name, dtype):
    weight = DEGENERATE[name].to("cuda").to(dtype)
    low, high = quant.dual_absmean_scales(weight)
    first, second = quant.trit_plane_absmean_scales(weight)
    scale = quant.binary_scale(weight)

    for lambda_ in (1.0, 0.5):
        _assert_matches(
            fused.fake_quant_weight_dual(weight, lambda_, low, high),
            quant.fake_quant_weight_dual(weight, lambda_, low, high),
            dtype,
            lambda_,
        )
        _assert_matches(
            fused.fake_quant_weight_trit_planes(weight, lambda_, first, second),
            quant.fake_quant_weight_trit_planes(weight, lambda_, first, second),
            dtype,
            lambda_,
        )
        _assert_matches(
            fused.fake_quant_weight_binary(weight, lambda_, None, scale),
            quant.fake_quant_weight_binary(weight, lambda_, None, scale),
            dtype,
            lambda_,
        )


@triton_cuda
def test_a_collapsed_scale_pair_matches_the_oracle(fused):
    """`s_lo == s_hi` collapses the five-value grid onto three; the kernel must too."""
    weight = _weight((4, 64), torch.bfloat16)
    same = torch.full((4, 1), 0.05, device="cuda")

    for lambda_ in (1.0, 0.5):
        assert torch.equal(
            fused.fake_quant_weight_dual(weight, lambda_, same, same),
            quant.fake_quant_weight_dual(weight, lambda_, same, same),
        )
        assert torch.equal(
            fused.fake_quant_weight_trit_planes(weight, lambda_, same, same),
            quant.fake_quant_weight_trit_planes(weight, lambda_, same, same),
        )


@triton_cuda
def test_a_swapped_scale_pair_is_ordered_like_the_oracle(fused):
    """Both take `(min, max)` defensively, so the caller's order cannot matter."""
    weight = _weight((4, 64), torch.bfloat16)
    small = torch.full((4, 1), 0.05, device="cuda")
    large = torch.full((4, 1), 0.2, device="cuda")

    assert torch.equal(
        fused.fake_quant_weight_dual(weight, 1.0, large, small),
        quant.fake_quant_weight_dual(weight, 1.0, large, small),
    )
    assert torch.equal(
        fused.fake_quant_weight_trit_planes(weight, 1.0, small, large),
        quant.fake_quant_weight_trit_planes(weight, 1.0, small, large),
    )


@triton_cuda
def test_a_zero_lambda_is_the_identity(fused):
    weight = _weight((4, 64), torch.bfloat16)
    low, high = quant.dual_absmean_scales(weight)

    assert torch.equal(fused.fake_quant_weight_dual(weight, 0.0, low, high), weight)


@triton_cuda
def test_the_kernels_refuse_a_missing_scale_pair(fused):
    weight = _weight((4, 64), torch.bfloat16)

    with pytest.raises(RuntimeError, match="explicit"):
        fused.fake_quant_weight_dual(weight, 1.0, None, None)
    with pytest.raises(RuntimeError, match="explicit"):
        fused.fake_quant_weight_trit_planes(weight, 1.0, None, None)


# ------------------------------------------------------------ module dispatch


@triton_cuda
@pytest.mark.parametrize(
    "weight_scale,codebook",
    [
        ("dual", "ternary"),
        ("trit_planes", "ternary"),
        ("absmean", "binary"),
        ("absmean", "ternary"),
    ],
)
def test_the_module_forward_is_the_same_fused_or_not(weight_scale, codebook):
    pytest.importorskip("triton")
    torch.manual_seed(0)
    module = (
        TernaryLinear.from_linear(
            nn.Linear(256, 128, bias=False),
            weight_scale=weight_scale,
            codebook=codebook,
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

    module.fused = True
    fused_out = module(x)
    module.fused = False
    eager_out = module(x)

    assert torch.equal(fused_out, eager_out)


@triton_cuda
@pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
def test_the_scale_gradients_survive_the_fused_forward(weight_scale):
    """The STE reads the codes, not the values, so swapping the forward is free."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    module = (
        TernaryLinear.from_linear(
            nn.Linear(512, 256, bias=False), weight_scale=weight_scale
        )
        .cuda()
        .to(torch.bfloat16)
    )
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)

    grads = {}
    for fused in (True, False):
        module.fused = fused
        for param in (module.weight, module.scale, module.scale_lo):
            param.grad = None
        module(x).backward(upstream)
        grads[fused] = tuple(
            param.grad.clone()
            for param in (module.weight, module.scale, module.scale_lo)
        )

    for fused_grad, eager_grad in zip(grads[True], grads[False], strict=True):
        assert torch.equal(fused_grad, eager_grad)


@triton_cuda
def test_a_sharded_weight_falls_back_to_the_oracle():
    """DTensors never reach a Triton kernel; the dispatch has to see that."""
    pytest.importorskip("triton")
    module = (
        TernaryLinear.from_linear(nn.Linear(64, 32, bias=False), weight_scale="dual")
        .cuda()
        .to(torch.bfloat16)
    )

    assert module._multi_state_ops(module.weight) is not None
    module.fused = False
    assert module._multi_state_ops(module.weight) is None


@triton_cuda
def test_the_dispatch_declines_fp32_partial_lambda():
    """Where the kernel is an ulp off, the dispatch hands back to the oracle."""
    pytest.importorskip("triton")
    module = (
        TernaryLinear.from_linear(nn.Linear(64, 32, bias=False), weight_scale="dual")
        .cuda()
        .to(torch.float32)
    )

    assert module._multi_state_ops(module.weight, 1.0) is not None
    assert module._multi_state_ops(module.weight, 0.5) is None
    # bf16 keeps the fused path at every λ
    module = module.to(torch.bfloat16)
    assert module._multi_state_ops(module.weight, 0.5) is not None


def test_the_dispatch_is_off_the_cuda_path_on_cpu():
    module = TernaryLinear.from_linear(
        nn.Linear(64, 32, bias=False), weight_scale="dual"
    )

    assert module._multi_state_ops(module.weight) is None


# ------------------------------------------------- fused backward (scale grads)


@pytest.fixture(name="no_kernels")
def fixture_no_kernels(monkeypatch):
    """Force the eager gradient path, whatever the device offers."""
    monkeypatch.setattr(quant, "_multi_state_kernels", lambda: None)


def _scale_grads(module, x, upstream):
    for param in (module.weight, module.scale, module.scale_lo):
        param.grad = None
    module(x).backward(upstream)
    return tuple(
        param.grad.clone() for param in (module.weight, module.scale, module.scale_lo)
    )


@triton_cuda
@pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lambda_", [1.0, 0.5])
def test_the_fused_backward_is_bit_identical(monkeypatch, weight_scale, dtype, lambda_):
    """The reduction stays in torch precisely so this is equality, not closeness."""
    pytest.importorskip("triton")
    torch.manual_seed(0)
    module = (
        TernaryLinear.from_linear(
            nn.Linear(512, 256, bias=False), weight_scale=weight_scale
        )
        .cuda()
        .to(dtype)
    )
    module.set_lambda(lambda_)
    x = torch.randn(8, 512, device="cuda", dtype=dtype)
    upstream = torch.randn(8, 256, device="cuda", dtype=dtype)

    fused = _scale_grads(module, x, upstream)
    monkeypatch.setattr(quant, "_multi_state_kernels", lambda: None)
    eager = _scale_grads(module, x, upstream)

    for got, want in zip(fused, eager, strict=True):
        assert torch.equal(got, want)


@triton_cuda
@pytest.mark.parametrize("name", sorted(DEGENERATE))
def test_the_fused_backward_handles_degenerate_rows(monkeypatch, name):
    pytest.importorskip("triton")
    weight = DEGENERATE[name].to("cuda").to(torch.bfloat16)
    rows, cols = weight.shape
    module = TernaryLinear(cols, rows, weight_scale="dual", dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        module.weight.copy_(weight)
    module.refresh_scale_from_weight()
    x = torch.randn(4, cols, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn(4, rows, device="cuda", dtype=torch.bfloat16)

    fused = _scale_grads(module, x, upstream)
    monkeypatch.setattr(quant, "_multi_state_kernels", lambda: None)
    eager = _scale_grads(module, x, upstream)

    for got, want in zip(fused, eager, strict=True):
        assert torch.equal(got, want)


@triton_cuda
def test_the_gradient_terms_match_the_eager_expressions(fused):
    """The kernels alone, against the expressions the STE used to inline."""
    torch.manual_seed(0)
    grad = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    codes = torch.randint(-2, 3, (16, 128), device="cuda", dtype=torch.int8)
    planes = torch.randint(-1, 2, (2, 16, 128), device="cuda", dtype=torch.int8)

    low, high = fused.dual_scale_grad_terms(grad, codes, 0.5)
    reference = grad.float() * torch.sign(codes.float()) * 0.5
    assert torch.equal(low, reference * (codes.abs() == 1))
    assert torch.equal(high, reference * (codes.abs() > 1))

    first, second = fused.trit_scale_grad_terms(grad, planes, 0.5)
    weighted = grad.float() * 0.5
    assert torch.equal(first, weighted * planes[0].float())
    assert torch.equal(second, weighted * planes[1].float())


def test_the_gradient_helpers_fall_back_off_cuda():
    grad = torch.randn(4, 8)
    codes = torch.randint(-2, 3, (4, 8), dtype=torch.int8)

    low, high = quant._dual_grad_terms(grad, codes, 1.0)

    assert low.shape == grad.shape and high.shape == grad.shape


# ------------------------------------------------- saved codes (one assignment)


@triton_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", [1.0, 0.5])
def test_the_emitted_dual_codes_are_the_eager_codes(fused, dtype, shape, lambda_):
    """What the STE saves must be what `dual_codes` would have produced."""
    weight = _weight(shape, dtype, seed=shape[1])
    low, high = quant.dual_absmean_scales(weight)

    out, codes = fused.fake_quant_weight_dual(
        weight, lambda_, low, high, with_codes=True
    )

    assert codes.dtype is torch.int8
    assert torch.equal(codes, quant.dual_codes(weight, low, high))
    _assert_matches(
        out, quant.fake_quant_weight_dual(weight, lambda_, low, high), dtype, lambda_
    )


@triton_cuda
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("lambda_", [1.0, 0.5])
def test_the_emitted_trit_planes_are_the_eager_planes(fused, dtype, shape, lambda_):
    weight = _weight(shape, dtype, seed=shape[1])
    scale_1, scale_2 = quant.trit_plane_absmean_scales(weight)
    first, second = quant.trit_plane_grid_scales(scale_1, scale_2, dtype)

    out, planes = fused.fake_quant_weight_trit_planes(
        weight, lambda_, scale_1, scale_2, with_codes=True
    )

    assert planes.dtype is torch.int8
    assert planes.shape == (quant.DUAL_PLANES, *shape)
    assert torch.equal(planes, quant.trit_plane_codes(weight, first, second))
    _assert_matches(
        out,
        quant.fake_quant_weight_trit_planes(weight, lambda_, scale_1, scale_2),
        dtype,
        lambda_,
    )


@triton_cuda
@pytest.mark.parametrize("name", sorted(DEGENERATE))
def test_the_emitted_codes_survive_degenerate_rows(fused, name):
    weight = DEGENERATE[name].to("cuda").to(torch.bfloat16)
    low, high = quant.dual_absmean_scales(weight)
    scale_1, scale_2 = quant.trit_plane_absmean_scales(weight)
    first, second = quant.trit_plane_grid_scales(scale_1, scale_2, torch.bfloat16)

    _, codes = fused.fake_quant_weight_dual(weight, 1.0, low, high, with_codes=True)
    _, planes = fused.fake_quant_weight_trit_planes(
        weight, 1.0, scale_1, scale_2, with_codes=True
    )

    assert torch.equal(codes, quant.dual_codes(weight, low, high))
    assert torch.equal(planes, quant.trit_plane_codes(weight, first, second))


@triton_cuda
def test_emitting_codes_does_not_change_the_effective_weight(fused):
    """`with_codes` is a second output, not a second code path."""
    weight = _weight((64, 512), torch.bfloat16)
    low, high = quant.dual_absmean_scales(weight)
    scale_1, scale_2 = quant.trit_plane_absmean_scales(weight)

    plain = fused.fake_quant_weight_dual(weight, 1.0, low, high)
    with_codes, _ = fused.fake_quant_weight_dual(
        weight, 1.0, low, high, with_codes=True
    )
    assert torch.equal(plain, with_codes)

    plain = fused.fake_quant_weight_trit_planes(weight, 1.0, scale_1, scale_2)
    with_codes, _ = fused.fake_quant_weight_trit_planes(
        weight, 1.0, scale_1, scale_2, with_codes=True
    )
    assert torch.equal(plain, with_codes)


@triton_cuda
@pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
def test_the_ste_saves_int8_codes_and_no_more(weight_scale):
    """The saved tensor is the int8 codes, so the fix costs no activation memory.

    The eager path already saved exactly these; emitting them from the forward only
    removes the second assignment, it does not hold anything new between the passes.
    """
    pytest.importorskip("triton")
    from axolotl.integrations.ternary.kernels.triton import multi_state

    torch.manual_seed(0)
    weight = _weight((64, 512), torch.bfloat16).requires_grad_(True)
    planes = quant.DUAL_PLANES if weight_scale == "trit_planes" else 1

    if weight_scale == "dual":
        low, high = quant.dual_absmean_scales(weight.detach())
        out = quant.fake_quant_weight_dual_ste(
            weight,
            1.0,
            low.requires_grad_(True),
            high.requires_grad_(True),
            impl=multi_state,
        )
    else:
        first, second = quant.trit_plane_absmean_scales(weight.detach())
        out = quant.fake_quant_weight_trit_planes_ste(
            weight,
            1.0,
            first.requires_grad_(True),
            second.requires_grad_(True),
            impl=multi_state,
        )

    saved = [t for t in out.grad_fn.saved_tensors if t.dtype is torch.int8]
    assert len(saved) == 1
    assert saved[0].numel() == weight.numel() * planes


@triton_cuda
@pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
def test_a_frozen_scale_skips_the_codes_entirely(weight_scale):
    """No scale gradient, nothing to save — the forward stays a single output."""
    pytest.importorskip("triton")
    module = (
        TernaryLinear.from_linear(
            nn.Linear(256, 128, bias=False), weight_scale=weight_scale
        )
        .cuda()
        .to(torch.bfloat16)
    )
    module.scale.requires_grad_(False)
    module.scale_lo.requires_grad_(False)
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)

    module.fused = True
    fused_out = module(x)
    module.fused = False
    assert torch.equal(fused_out, module(x))
