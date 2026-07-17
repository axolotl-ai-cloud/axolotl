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
        base.weight
        + (lora.lora_B["default"].weight @ lora.lora_A["default"].weight)
        * lora.scaling["default"]
    )
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
    w_eff = base.weight + (B_o @ A_o) * lora.scaling["default"]
    w_fq = w_eff + (fake_quant_nvfp4(w_eff.detach(), base._nvfp4_pts) - w_eff.detach())
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
        base.weight
        + (lora.lora_B["default"].weight @ lora.lora_A["default"].weight)
        * lora.scaling["default"]
    )
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
