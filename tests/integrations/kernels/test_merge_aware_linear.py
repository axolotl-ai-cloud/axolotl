"""CPU tests for merge-aware LoRA over dequantized non-expert NVFP4 linears.

Deliberately does NOT import axolotl.cli.utils.lora_merge (it reconfigures
axolotl logging and breaks caplog for later tests in the same session); the
writer-side bitwise identity for the non-expert path lives in
tests/utils/lora/test_merge_lora.py.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)

torchao = pytest.importorskip("torchao")
peft = pytest.importorskip("peft")

from peft import LoraConfig  # noqa: E402
from peft.tuners.lora.layer import Linear as LoraLinear  # noqa: E402


def _lora_linear(base, dropout=0.0):
    cfg = LoraConfig(r=R, lora_alpha=2 * R, lora_dropout=dropout)
    return LoraLinear(
        base,
        adapter_name="default",
        config=cfg,
        r=R,
        lora_alpha=2 * R,
        lora_dropout=dropout,
    )


from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_lora import (  # noqa: E402
    set_merge_aware_enabled,
)
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (  # noqa: E402
    fake_quant_nvfp4,
)
from axolotl.integrations.kernels.merge_aware_linear import (  # noqa: E402
    install_merge_aware_lora_linears,
)

OUT, IN, R = 32, 64, 8


@pytest.fixture(autouse=True)
def _toggle_off_after():
    yield
    set_merge_aware_enabled(False)


def _nvfp4_base_linear():
    """A dense bf16 linear whose weight sits exactly on an NVFP4 grid, plus its pts."""
    w0 = torch.randn(OUT, IN, dtype=torch.bfloat16)
    pts = (w0.float().abs().amax() / (6.0 * 448.0)).to(torch.float32)
    w_dense = fake_quant_nvfp4(w0, pts)
    base = nn.Linear(IN, OUT, bias=False, dtype=torch.bfloat16)
    base.weight = nn.Parameter(w_dense, requires_grad=False)
    base._nvfp4_pts = pts.reshape(())
    return base


def _wrapped_model(with_pts=True):
    base = _nvfp4_base_linear()
    if not with_pts:
        del base._nvfp4_pts
    lora = _lora_linear(base).to(torch.bfloat16)
    with torch.no_grad():
        lora.lora_B["default"].weight.copy_(
            0.02 * torch.randn(OUT, R, dtype=torch.bfloat16)
        )
    model = nn.Sequential(lora)
    return model, lora


def test_install_wraps_only_nvfp4_origin():
    model, lora = _wrapped_model()
    assert install_merge_aware_lora_linears(model) == 1
    assert hasattr(lora, "_ma_orig_forward")
    # idempotent
    assert install_merge_aware_lora_linears(model) == 1

    plain_model, plain_lora = _wrapped_model(with_pts=False)
    assert install_merge_aware_lora_linears(plain_model) == 0
    assert not hasattr(plain_lora, "_ma_orig_forward")


def test_toggle_off_is_plain_peft_forward():
    model, lora = _wrapped_model()
    x = torch.randn(4, IN, dtype=torch.bfloat16)
    before = lora(x)
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(False)
    assert torch.equal(lora(x), before)


def test_forward_matches_snapped_oracle():
    model, lora = _wrapped_model()
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(True)

    x = torch.randn(4, IN, dtype=torch.bfloat16)
    out = lora(x)

    base = lora.get_base_layer()
    w_eff = (
        base.weight.float()
        + (
            lora.lora_B["default"].weight.float()
            @ lora.lora_A["default"].weight.float()
        )
        * lora.scaling["default"]
    ).to(base.weight.dtype)
    oracle = F.linear(x, fake_quant_nvfp4(w_eff, base._nvfp4_pts))
    assert torch.equal(out, oracle)
    # the snap must actually change the weight (delta is off-grid)
    assert not torch.equal(out, lora._ma_orig_forward(x))


def test_ste_gradients_match_oracle():
    model, lora = _wrapped_model()
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(True)

    x = torch.randn(4, IN, dtype=torch.bfloat16, requires_grad=True)
    lora(x).float().square().sum().backward()

    base = lora.get_base_layer()
    A_o = lora.lora_A["default"].weight.detach().clone().requires_grad_()
    B_o = lora.lora_B["default"].weight.detach().clone().requires_grad_()
    x_o = x.detach().clone().requires_grad_()
    w_eff = (
        base.weight.float() + (B_o.float() @ A_o.float()) * lora.scaling["default"]
    ).to(base.weight.dtype)
    w_fq = fake_quant_nvfp4(w_eff.detach(), base._nvfp4_pts) + (w_eff - w_eff.detach())
    F.linear(x_o, w_fq).float().square().sum().backward()

    assert torch.equal(lora.lora_A["default"].weight.grad, A_o.grad)
    assert torch.equal(lora.lora_B["default"].weight.grad, B_o.grad)
    assert torch.equal(x.grad, x_o.grad)
    # dx flows through the SNAPPED operand, not W_eff
    with torch.no_grad():
        out_grad = 2 * F.linear(x_o, w_fq).float()
    dx_snapped = (out_grad.to(torch.bfloat16) @ w_fq).to(x.grad.dtype)
    assert torch.allclose(x.grad.float(), dx_snapped.float(), rtol=1e-2, atol=1e-2)


def test_dropout_residual_vanishes_at_eval():
    base = _nvfp4_base_linear()
    lora = _lora_linear(base, dropout=0.5).to(torch.bfloat16)
    with torch.no_grad():
        lora.lora_B["default"].weight.copy_(
            0.02 * torch.randn(OUT, R, dtype=torch.bfloat16)
        )
    model = nn.Sequential(lora)
    assert install_merge_aware_lora_linears(model) == 1
    set_merge_aware_enabled(True)

    x = torch.randn(4, IN, dtype=torch.bfloat16)
    lora.eval()
    out = lora(x)
    w_eff = (
        base.weight.float()
        + (
            lora.lora_B["default"].weight.float()
            @ lora.lora_A["default"].weight.float()
        )
        * lora.scaling["default"]
    ).to(base.weight.dtype)
    oracle = F.linear(x, fake_quant_nvfp4(w_eff, base._nvfp4_pts))
    assert torch.equal(out, oracle)

    # train mode: the dropout residual perturbs the output around the snapped term
    lora.train()
    torch.manual_seed(1)
    t1 = lora(x)
    torch.manual_seed(2)
    t2 = lora(x)
    assert not torch.equal(t1, t2)
    assert not torch.equal(t1, oracle)


def test_disable_adapters_falls_back():
    model, lora = _wrapped_model()
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(True)
    x = torch.randn(4, IN, dtype=torch.bfloat16)
    lora.enable_adapters(False)
    base_out = F.linear(x, lora.get_base_layer().weight)
    assert torch.equal(lora(x), base_out)
    lora.enable_adapters(True)


@pytest.mark.parametrize("kind", ["linear", "o", "qkv", "qk", "swiglu", "geglu", "gdn"])
def test_lora_kernel_dispatch_preserves_fake_quant_and_gradients(kind):
    from axolotl.kernels import lora as kernels

    model = nn.Module()
    names = {
        "linear": ["proj"],
        "o": ["o_proj"],
        "qkv": ["q_proj", "k_proj", "v_proj"],
        "qk": ["q_proj", "k_proj"],
        "swiglu": ["gate_proj", "up_proj", "down_proj"],
        "geglu": ["gate_proj", "up_proj", "down_proj"],
        "gdn": ["in_proj_qkv", "in_proj_z"],
    }[kind]
    for name in names:
        base = nn.Linear(IN, IN, bias=False, dtype=torch.bfloat16)
        base.weight.requires_grad_(False)
        base._nvfp4_pts = base.weight.float().abs().amax() / (6 * 448)
        layer = _lora_linear(base).to(torch.bfloat16)
        nn.init.normal_(layer.lora_B["default"].weight, std=0.02)
        setattr(model, name, layer)
    model.act_fn = (
        nn.SiLU() if kind == "swiglu" else lambda x: F.gelu(x.float()).to(x.dtype)
    )
    install_merge_aware_lora_linears(model)

    def forward(x, fused):
        if kind == "linear":
            return kernels.apply_lora_linear(model.proj, x) if fused else model.proj(x)
        if kind == "o":
            return kernels.apply_lora_o(model, x) if fused else model.o_proj(x)
        if kind in ("qkv", "qk"):
            if fused:
                return getattr(kernels, "apply_lora_" + kind)(model, x)
            q, k = model.q_proj(x), model.k_proj(x)
            return q, k, model.v_proj(x) if kind == "qkv" else k
        if kind == "gdn":
            result = (
                kernels.apply_lora_gdn_in_proj(model, x, tuple(names))
                if fused
                else {n: getattr(model, n)(x) for n in names}
            )
            return tuple(result.values())
        if fused:
            return getattr(kernels, "apply_lora_mlp_" + kind)(model, x)
        return model.down_proj(model.act_fn(model.gate_proj(x)) * model.up_proj(x))

    for enabled in [False, True]:
        set_merge_aware_enabled(enabled)
        x = torch.randn(2, 3, IN, dtype=torch.bfloat16)
        results = []
        for fused in [False, True]:
            model.zero_grad(set_to_none=True)
            leaf = x.clone().requires_grad_()
            result = forward(leaf, fused)
            values = result if isinstance(result, tuple) else (result,)
            sum(v.float().square().sum() for v in values).backward()
            results.append(
                (
                    [v.detach() for v in values],
                    leaf.grad.clone(),
                    [p.grad.clone() for p in model.parameters() if p.requires_grad],
                )
            )
        for before, after in zip(results[0][0], results[1][0], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
        for before, after in zip(results[0][2], results[1][2], strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_mixed_merge_aware_qkv_keeps_other_projections_optimized(monkeypatch):
    from axolotl.kernels import lora as kernels

    model = nn.Module()
    for name in ["q_proj", "k_proj", "v_proj"]:
        _, projection = _wrapped_model(with_pts=name == "q_proj")
        setattr(model, name, projection.cuda())
    install_merge_aware_lora_linears(model)
    original = kernels.LoRA_O.apply
    calls = []

    def apply(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(kernels.LoRA_O, "apply", apply)
    set_merge_aware_enabled(True)
    x = torch.randn(2, 3, IN, device="cuda", dtype=torch.bfloat16)
    reference = tuple(getattr(model, n)(x) for n in ["q_proj", "k_proj", "v_proj"])
    actual = kernels.apply_lora_qkv(model, x)
    assert len(calls) == 2
    torch.testing.assert_close(actual[0], reference[0], rtol=0, atol=0)
    for before, after in zip(reference[1:], actual[1:], strict=True):
        torch.testing.assert_close(before, after, rtol=0.02, atol=0.01)
    sum(v.float().square().sum() for v in actual).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp32_adapter_matches_export_effective_weight(autocast, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(918)
    model, lora = _wrapped_model()
    model.to(device)
    lora.lora_A["default"].float()
    lora.lora_B["default"].float()
    lora.scaling["default"] = 1.3
    base = lora.get_base_layer()
    delta = lora.get_delta_weight("default")
    writer_weight = (base.weight.float() + delta.float()).to(base.weight.dtype)
    previous_weight = (
        base.weight
        + (lora.lora_B["default"].weight @ lora.lora_A["default"].weight).to(
            base.weight.dtype
        )
        * 1.3
    )
    assert not torch.equal(writer_weight, previous_weight)
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(True)
    x = torch.randn(4, IN, dtype=torch.bfloat16, device=device, requires_grad=True)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
        actual = model(x)
        expected = F.linear(x, fake_quant_nvfp4(writer_weight, base._nvfp4_pts))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().sum().backward()
    assert x.grad is not None
    assert lora.lora_A["default"].weight.grad.dtype == torch.float32
    assert lora.lora_B["default"].weight.grad.dtype == torch.float32


@pytest.mark.parametrize("adapter_dtype", [torch.bfloat16, torch.float32])
def test_adapter_dtype_matches_canonical_writer_effective_weight(adapter_dtype):
    torch.manual_seed(921)
    model, lora = _wrapped_model()
    lora.lora_A["default"].to(adapter_dtype)
    lora.lora_B["default"].to(adapter_dtype)
    with torch.no_grad():
        lora.lora_B["default"].weight.normal_(std=0.03)
    lora.scaling["default"] = 1.3
    base = lora.get_base_layer()
    writer_weight = (
        base.weight.float()
        + (
            lora.lora_B["default"].weight.float()
            @ lora.lora_A["default"].weight.float()
        )
        * lora.scaling["default"]
    ).to(base.weight.dtype)
    install_merge_aware_lora_linears(model)
    set_merge_aware_enabled(True)
    x = torch.randn(4, IN, dtype=torch.bfloat16)

    torch.testing.assert_close(
        model(x),
        F.linear(x, fake_quant_nvfp4(writer_weight, base._nvfp4_pts)),
        rtol=0,
        atol=0,
    )


def test_factor_bias_warns_without_installing_incomplete_forward():
    from unittest.mock import patch

    model, lora = _wrapped_model()
    lora.lora_B["default"].bias = nn.Parameter(torch.zeros(OUT, dtype=torch.bfloat16))
    forward = lora.forward
    with patch(
        "axolotl.integrations.kernels.merge_aware_linear.LOG.warning"
    ) as warning:
        assert install_merge_aware_lora_linears(model) == 0
    assert lora.forward == forward
    assert lora._axolotl_merge_aware_unsupported
    assert "NVFP4 MERGE WARNING" in warning.call_args.args[0]
