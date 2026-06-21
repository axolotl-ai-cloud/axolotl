"""Correctness tests for the Triton head_dim=512 flash attention (fwd+bwd, dense + varlen)."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

DEV = "cuda"
D = 512


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), 0).item()


@pytest.mark.parametrize(
    "B,H,N,causal", [(1, 8, 1024, True), (2, 8, 1024, True), (1, 8, 1024, False)]
)
def test_dense_fwd_bwd_matches_sdpa(B, H, N, causal):
    from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    out = flash_d512(q, k, v, causal)
    out.float().pow(2).mean().backward()
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=causal)
    ref.float().pow(2).mean().backward()
    assert _cos(out, ref) > 0.999
    assert _cos(q.grad, qr.grad) > 0.999
    assert _cos(k.grad, kr.grad) > 0.999
    assert _cos(v.grad, vr.grad) > 0.999


@pytest.mark.parametrize("docs", [[512, 512], [300, 400, 324], [200, 300, 524]])
def test_varlen_matches_per_document(docs):
    """Packed (varlen) fwd+bwd must match per-document independent attention."""
    from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

    B, H, N = 1, 8, sum(docs)
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, N, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    pos = torch.cat([torch.arange(d) for d in docs]).to(DEV)[None]
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))

    out = flash_d512(q, k, v, True, position_ids=pos)
    out.float().pow(2).mean().backward()

    outs, off = [], 0
    for d in docs:
        with sdpa_kernel([SDPBackend.MATH]):
            outs.append(
                F.scaled_dot_product_attention(
                    qr[:, :, off : off + d],
                    kr[:, :, off : off + d],
                    vr[:, :, off : off + d],
                    is_causal=True,
                )
            )
        off += d
    ref = torch.cat(outs, 2)
    ref.float().pow(2).mean().backward()

    assert _cos(out, ref) > 0.999
    assert _cos(q.grad, qr.grad) > 0.999
    assert _cos(k.grad, kr.grad) > 0.999
    assert _cos(v.grad, vr.grad) > 0.999
