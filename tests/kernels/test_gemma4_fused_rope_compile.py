"""torch.compile traceability tests for the fused RMSNorm+RoPE kernel."""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _make_inputs(B=2, S=64, H=4, D=64, n_rot=64, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(B, S, H, D, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(D, device="cuda", dtype=dtype, requires_grad=True)
    cos = torch.randn(B, S, n_rot, device="cuda", dtype=dtype)
    sin = torch.randn(B, S, n_rot, device="cuda", dtype=dtype)
    return x, w, cos, sin


def test_fused_rms_norm_rope_compile_forward_matches_eager():
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    x, w, cos, sin = _make_inputs(seed=1)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = w.detach().clone().requires_grad_(True)

    y_eager = fused_rms_norm_rope(x_ref, w_ref, cos, sin, eps=1e-6)

    compiled = torch.compile(fused_rms_norm_rope, fullgraph=True, dynamic=False)
    y_compiled = compiled(x, w, cos, sin, eps=1e-6)

    torch.testing.assert_close(y_compiled, y_eager, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(y_compiled).all()


def test_fused_rms_norm_rope_compile_backward():
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

    x, w, cos, sin = _make_inputs(seed=2)

    compiled = torch.compile(fused_rms_norm_rope, fullgraph=True, dynamic=False)
    y = compiled(x, w, cos, sin, eps=1e-6)
    y.sum().backward()

    assert x.grad is not None and x.grad.isfinite().all() and x.grad.abs().sum() > 0
    assert w.grad is not None and w.grad.isfinite().all() and w.grad.abs().sum() > 0


def test_fused_rms_norm_noscale_compile():
    from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_noscale

    torch.manual_seed(3)
    x = torch.randn(
        2, 32, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_ref = x.detach().clone().requires_grad_(True)

    y_eager = fused_rms_norm_noscale(x_ref, eps=1e-6)

    compiled = torch.compile(fused_rms_norm_noscale, fullgraph=True, dynamic=False)
    y_compiled = compiled(x, eps=1e-6)

    torch.testing.assert_close(y_compiled, y_eager, rtol=1e-2, atol=1e-2)

    y_compiled.sum().backward()
    assert x.grad is not None and x.grad.isfinite().all() and x.grad.abs().sum() > 0
