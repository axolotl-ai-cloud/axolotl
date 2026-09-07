"""Correctness of the fused GeGLU LoRA MLP kernel vs an eager autograd reference.

This is the path re-enabled for MoE models by ``disable_mlp_kernel`` (the dense *shared* MLP,
e.g. gemma4's per-layer ``Gemma4TextMLP``, gets ``lora_mlp_kernel`` while the routed experts are
handled by the MoE kernel). Verifies forward output, input grad, and all six LoRA A/B grads match
an eager reference — over both a bf16 base and an nf4-quantized base (the low-VRAM gemma4 case),
since the fused kernel dequantizes the base in-kernel.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from axolotl.kernels.lora import LoRA_MLP  # noqa: E402

try:
    import bitsandbytes.functional as bnb_F  # noqa: E402
except Exception:  # pragma: no cover
    bnb_F = None

DEV = "cuda"


def _geglu_eager(X, gw, gA, gB, uw, uA, uB, dw, dA, dB, s):
    """Eager gated-GeGLU MLP with LoRA: down(gelu_tanh(gate(x)) * up(x)), gelu tanh-approx."""
    g = X @ gw.t() + s * ((X @ gA.t()) @ gB.t())
    u = X @ uw.t() + s * ((X @ uA.t()) @ uB.t())
    h = F.gelu(g, approximate="tanh") * u
    return h @ dw.t() + s * ((h @ dA.t()) @ dB.t())


def _mk(*shape, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (
        torch.randn(*shape, device=DEV, dtype=torch.float16, generator=g) * 0.1
    ).requires_grad_(True)


@pytest.mark.parametrize("quantized", [False, True])
def test_fused_geglu_lora_matches_eager(quantized):
    if quantized and bnb_F is None:
        pytest.skip("bitsandbytes required for the quantized-base case")

    from axolotl.kernels.geglu import geglu_backward, geglu_forward

    H, IM, r, B, T = 256, 512, 16, 2, 8  # hidden, intermediate, rank, batch, seq
    s = 0.5

    # base weights (frozen). down maps IM->H; gate/up map H->IM.
    gw = torch.randn(IM, H, device=DEV, dtype=torch.float16) * 0.05
    uw = torch.randn(IM, H, device=DEV, dtype=torch.float16) * 0.05
    dw = torch.randn(H, IM, device=DEV, dtype=torch.float16) * 0.05

    if quantized:
        # nf4-quantize each base; the eager ref dequantizes the SAME packed weight (via the kernel's
        # dequantize) so only fused-vs-eager kernel numerics differ, not the quantization.
        from axolotl.kernels.quantize import dequantize

        gw_p, gq = bnb_F.quantize_4bit(gw, quant_type="nf4", compress_statistics=True)
        uw_p, uq = bnb_F.quantize_4bit(uw, quant_type="nf4", compress_statistics=True)
        dw_p, dq = bnb_F.quantize_4bit(dw, quant_type="nf4", compress_statistics=True)
        gw_ref = dequantize(gw_p.t(), gq).t().to(torch.float16)
        uw_ref = dequantize(uw_p.t(), uq).t().to(torch.float16)
        dw_ref = dequantize(dw_p.t(), dq).t().to(torch.float16)
        gw_k, uw_k, dw_k = gw_p, uw_p, dw_p
    else:
        gq = uq = dq = None
        gw_ref, uw_ref, dw_ref = gw, uw, dw
        gw_k, uw_k, dw_k = gw, uw, dw

    # LoRA params (seeded). down LoRA: A[r,I], B[H,r]; gate/up LoRA: A[r,H], B[I,r].
    def fresh():
        gA, gB = _mk(r, H, seed=1), _mk(IM, r, seed=2)
        uA, uB = _mk(r, H, seed=3), _mk(IM, r, seed=4)
        dA, dB = _mk(r, IM, seed=5), _mk(H, r, seed=6)
        return gA, gB, uA, uB, dA, dB

    Xv = torch.randn(B, T, H, device=DEV, dtype=torch.float16) * 0.1
    grad_out = torch.randn(B, T, H, device=DEV, dtype=torch.float16)

    # --- fused ---
    Xf = Xv.clone().requires_grad_(True)
    gA, gB, uA, uB, dA, dB = fresh()
    out_f = LoRA_MLP.apply(
        Xf,
        None,
        gw_k,
        None,
        gq,
        gA,
        gB,
        s,
        None,
        None,
        uw_k,
        None,
        uq,
        uA,
        uB,
        s,
        None,
        None,
        dw_k,
        None,
        dq,
        dA,
        dB,
        s,
        None,
        None,
        geglu_forward,
        geglu_backward,
        True,
    )
    out_f.backward(grad_out)
    gf = [t.grad.clone() for t in (Xf, gA, gB, uA, uB, dA, dB)]

    # --- eager ---
    Xe = Xv.clone().requires_grad_(True)
    eA, eB, euA, euB, edA, edB = fresh()
    out_e = _geglu_eager(Xe, gw_ref, eA, eB, uw_ref, euA, euB, dw_ref, edA, edB, s)
    out_e.backward(grad_out)
    ge = [t.grad.clone() for t in (Xe, eA, eB, euA, euB, edA, edB)]

    def rel(a, b):
        return (a - b).float().abs().max().item() / max(
            b.float().abs().max().item(), 1e-6
        )

    tol = 3e-2  # fp16 + Triton reduction order (+ nf4 dequant rounding shared by both)
    assert torch.isfinite(out_f).all()
    assert rel(out_f, out_e) < tol, f"forward rel={rel(out_f, out_e):.4f}"
    names = ["dX", "dgate_A", "dgate_B", "dup_A", "dup_B", "ddown_A", "ddown_B"]
    for n, a, b in zip(names, gf, ge, strict=True):
        assert torch.isfinite(a).all(), f"{n} non-finite"
        assert rel(a, b) < tol, f"{n} rel={rel(a, b):.4f}"
