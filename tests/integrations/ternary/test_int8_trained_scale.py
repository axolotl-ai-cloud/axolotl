"""The int8 W2A8 forward for the trained-scale modes, `learnable` and `learnable_row`.

One int8 GEMM plus a scalar or per-output-feature epilogue. Both derive their codes
from the *trained* scale rather than an absmean statistic re-derived from the latent;
those are different quantizers once training has moved the scale, so the tests below
pin the trained one. `dual` and `trit_planes` are deliberately *not* on this path —
their two-GEMM split regressed the widest projection shape — so these tests also pin
that they keep falling back to the fake-quant forward.
"""

import pytest
import torch
from torch import nn

from axolotl.integrations.ternary import quant
from axolotl.integrations.ternary.kernels import int8_gemm as int8
from axolotl.integrations.ternary.modules import INT8_FORWARD_SCALE_MODES, TernaryLinear

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the int8 forward needs a CUDA device"
)

# the established gate: int accumulation is exact, only the fp32 rescale order differs
REL_TOLERANCE = 1e-2

TRAINED_SCALE_MODES = ["learnable", "learnable_row"]


def _module(
    weight_scale="learnable_row", in_features=2048, out_features=2048, **kwargs
):
    torch.manual_seed(0)
    return (
        TernaryLinear.from_linear(
            nn.Linear(in_features, out_features, bias=False),
            weight_scale=weight_scale,
            int8_forward=True,
            **kwargs,
        )
        .cuda()
        .to(torch.bfloat16)
        .eval()
    )


def _relative(got, want):
    return float((got.float() - want.float()).abs().max()) / float(
        want.float().abs().max()
    )


# ------------------------------------------------------------------- numerics


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
@pytest.mark.parametrize("tokens", [1, 64, 512])
@pytest.mark.parametrize("out_features", [2048, 8192])
def test_the_int8_forward_matches_the_fake_quant_path(
    tokens, out_features, weight_scale
):
    module = _module(weight_scale=weight_scale, out_features=out_features)
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        module.int8_forward = True
        got = module(x)
        module.int8_forward = False
        want = module(x)

    assert _relative(got, want) < REL_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
def test_the_advertised_modes_are_the_dispatched_ones(weight_scale):
    """`INT8_FORWARD_SCALE_MODES` once advertised a mode the gate silently refused."""
    module = _module(weight_scale=weight_scale)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)

    assert module._int8_linear(x) is not None
    assert weight_scale in INT8_FORWARD_SCALE_MODES


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
def test_the_codes_come_from_the_trained_scale_not_the_absmean_statistic(weight_scale):
    """Training moves the scale off its absmean seed; the two then disagree.

    A quarter of the codes differ at this shift, so a re-derived statistic is not a
    near-enough approximation of the grid the module was actually trained on.
    """
    module = _module(weight_scale=weight_scale, in_features=512, out_features=512)
    with torch.no_grad():
        module.scale.add_(0.7)  # log-space: the grid roughly doubles
    weight = module.weight.detach()

    from_statistic = int8.ternary_codes(weight, quant.absmean_scale(weight))
    codes, _ = int8._derive_weight_codes(module)

    assert not torch.equal(codes, from_statistic)
    x = torch.randn(64, 512, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        module.int8_forward = True
        got = module(x)
        module.int8_forward = False
        want = module(x)
    assert _relative(got, want) < REL_TOLERANCE


@requires_cuda
def test_the_row_scales_reach_the_epilogue():
    """A per-row grid must not be collapsed to one number by the rescale."""
    module = _module(out_features=2048)
    with torch.no_grad():
        # make the rows differ sharply, so a per-tensor epilogue cannot pass
        module.scale.copy_(
            torch.linspace(-4.0, -1.0, 2048, device="cuda").reshape(-1, 1)
        )
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        module.int8_forward = True
        got = module(x)
        module.int8_forward = False
        want = module(x)

    assert _relative(got, want) < REL_TOLERANCE


@requires_cuda
def test_int8_gemm_rejects_a_wrongly_sized_row_scale():
    codes = torch.randint(-1, 2, (128, 64), device="cuda", dtype=torch.int8)
    x_codes = torch.randint(-8, 8, (4, 64), device="cuda", dtype=torch.int8)
    x_scale = torch.rand(4, 1, device="cuda")

    with pytest.raises(ValueError, match="per output feature"):
        int8.int8_gemm(x_codes, codes, x_scale, torch.rand(7, device="cuda"))


# ------------------------------------------------------- the eval code cache


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
def test_a_moved_scale_invalidates_the_cached_codes(weight_scale):
    """The grid lives in a parameter; a stale cache would serve the old one."""
    module = _module(weight_scale=weight_scale)
    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        before = module(x)
        module.scale.mul_(2.0)
        after = module(x)
        module.int8_forward = False
        expected = module(x)

    assert not torch.equal(before, after), "the cache served the pre-update grid"
    assert _relative(after, expected) < REL_TOLERANCE


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
def test_a_fused_optimizer_step_invalidates_the_cached_codes(weight_scale):
    """`torch._fused_adamw_` moves parameters without bumping their version counter.

    That blind spot once let a fit-initialized module claim to be baked for a whole
    run; the code cache must not repeat it.
    """
    module = _module(weight_scale=weight_scale)
    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW([module.scale], lr=0.5, fused=True)

    with torch.no_grad():
        before = module(x)
    version = module.scale._version

    module.scale.grad = torch.ones_like(module.scale)
    optimizer.step()
    assert module.scale._version == version, "premise changed: fused step bumped it"

    with torch.no_grad():
        after = module(x)
        module.int8_forward = False
        expected = module(x)

    assert not torch.equal(before, after)
    assert _relative(after, expected) < REL_TOLERANCE


# --------------------------------------------------- the training path is safe


@requires_cuda
@pytest.mark.parametrize("weight_scale", ["dual", "trit_planes"])
def test_the_rejected_modes_stay_on_the_fake_quant_path(weight_scale):
    module = _module(weight_scale=weight_scale)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)

    assert module._int8_linear(x) is None
    assert weight_scale not in INT8_FORWARD_SCALE_MODES


@requires_cuda
@pytest.mark.parametrize("weight_scale", TRAINED_SCALE_MODES)
def test_a_partial_lambda_stays_on_the_fake_quant_path(weight_scale):
    module = _module(weight_scale=weight_scale)
    module.set_lambda(0.5)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)

    assert module._int8_linear(x) is None


@requires_cuda
def test_training_mode_does_not_cache_codes():
    module = _module().train()
    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)

    module(x)

    assert not hasattr(module, int8._EVAL_CACHE_ATTR)


@requires_cuda
def test_the_gradient_still_reaches_the_latent_and_the_scale():
    """The int8 forward is a λ=1 fast path, not a different function to train."""
    module = _module().train()
    module.int8_forward = True
    x = torch.randn(16, 2048, device="cuda", dtype=torch.bfloat16)

    module(x).sum().backward()

    assert module.weight.grad is not None
    assert float(module.weight.grad.abs().sum()) > 0.0
