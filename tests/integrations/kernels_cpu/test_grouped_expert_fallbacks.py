"""Grouped-expert edge cases that run without the grouped GEMMs: mxfp8 experts off the grouped
path (no LoRA on both projections)."""

import pytest
import torch
from torch import nn

pytest.importorskip("triton")  # the scattermoe_lora package imports its Triton kernels

from axolotl.integrations.kernels.libs.scattermoe_lora import (  # noqa: E402
    experts as experts_mod,
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
