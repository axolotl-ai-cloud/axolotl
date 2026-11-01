"""Grouped-expert edge cases that run without the grouped GEMMs: mxfp8 experts off the grouped
path (no LoRA on both projections), and an expert-parallel rank with no local tokens."""

import pytest
import torch
from torch import nn

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels

from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    experts as experts_mod,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.fp8_grouped import (  # noqa: E402
    _Fp8BlockGroupedExperts,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (  # noqa: E402
    _GroupedExperts,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.mxfp8_grouped import (  # noqa: E402
    _Mxfp8GroupedExperts,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.mxfp8_requant import (  # noqa: E402
    requantize_experts_to_mxfp8,
)

E, H, I, R = 4, 64, 32, 2  # noqa: E741


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * 0.5)
        self.down_proj = nn.Parameter(torch.randn(E, H, I) * 0.5)


@pytest.mark.parametrize("lora", [None, "gate_up_only"])
def test_mxfp8_experts_without_both_loras_dequantize_with_scales(lora):
    mod = _Experts()
    ref_gu, ref_dn = mod.gate_up_proj.detach().clone(), mod.down_proj.detach().clone()
    assert requantize_experts_to_mxfp8(mod) == 1
    assert mod.gate_up_proj.dtype == torch.float8_e4m3fn
    gup_lora = (torch.zeros(R * E, H), torch.zeros(2 * I, R * E), 1.0) if lora else None
    idx = torch.arange(E)
    gu, dn, *_ = experts_mod._prepare_weights_and_lora(
        mod.gate_up_proj,
        mod.down_proj,
        idx,
        idx + 1,
        E,
        gup_lora,
        None,
        torch.bfloat16,
        mx_scales=experts_mod._mx_scales(mod),
    )
    for got, want in ((gu, ref_gu), (dn, ref_dn)):
        want = want.transpose(2, 1)
        assert got.shape == want.shape and got.dtype == torch.bfloat16
        # mxfp8 rounding only; the raw e4m3 values are off by the 2^k block scales
        rel = (got.float() - want).norm() / want.norm()
        assert rel < 0.1, rel


def _stacked_lora():
    def p(*shape):
        return torch.randn(*shape, requires_grad=True)

    return p(E, R, H), p(E, 2 * I, R), p(E, R, I), p(E, H, R)


@pytest.mark.parametrize(
    "fn,trailing",
    [
        (_Fp8BlockGroupedExperts, ("silu", None)),
        (_Mxfp8GroupedExperts, ("silu",)),
        (_GroupedExperts, ("nvfp4", "silu", True)),
    ],
    ids=["fp8_block", "mxfp8", "nvfp4"],
)
def test_empty_expert_route_backward(fn, trailing):
    # an EP rank whose tokens all route elsewhere: no rows, no tiles
    x = torch.zeros(0, H, requires_grad=True)
    Agu, Bgu, Adn, Bdn = _stacked_lora()
    m_indices = torch.zeros(0, dtype=torch.int32)
    offs = torch.zeros(E, dtype=torch.int32)
    out = fn.apply(
        x, None, None, Agu, Bgu, Adn, Bdn, m_indices, offs, 1.0, 1e30, *trailing
    )
    assert out.shape == (0, H)
    out.sum().backward()
    assert x.grad.shape == (0, H)
    for t in (Agu, Bgu, Adn, Bdn):
        assert t.grad is not None and not t.grad.any()
