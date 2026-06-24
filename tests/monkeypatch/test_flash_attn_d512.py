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


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


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


@pytest.mark.parametrize("scale", [None, 1.0])
def test_noncontiguous_gqa_grads_match_sdpa(scale):
    """Regression: real attention feeds non-contiguous q ([B,S,H,D].transpose(1,2)) with contiguous
    GQA-repeated k/v. The backward must not read wrong memory from the stride mismatch. Uses a
    magnitude-sensitive relative-error check (cosine alone misses the gradient blow-up)."""
    from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

    B, Hq, Hkv, N = 1, 16, 4, 1024
    ng = Hq // Hkv
    torch.manual_seed(0)
    # non-contiguous q exactly as produced by attention: [B, N, Hq, D] -> transpose -> [B, Hq, N, D]
    q = torch.randn(B, N, Hq, D, device=DEV, dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(B, Hkv, N, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, N, D, device=DEV, dtype=torch.bfloat16)
    assert not q.is_contiguous()
    q, k, v = (t.detach().requires_grad_() for t in (q, k, v))
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    eff = D**-0.5 if scale is None else scale

    out = flash_d512(
        q, k.repeat_interleave(ng, 1), v.repeat_interleave(ng, 1), True, scale=scale
    )
    out.float().pow(2).mean().backward()
    with sdpa_kernel([SDPBackend.MATH]):
        ref = F.scaled_dot_product_attention(
            qr,
            kr.repeat_interleave(ng, 1),
            vr.repeat_interleave(ng, 1),
            is_causal=True,
            scale=eff,
        )
    ref.float().pow(2).mean().backward()

    assert _rel(out, ref) < 0.01
    assert _rel(q.grad, qr.grad) < 0.02
    assert _rel(k.grad, kr.grad) < 0.02
    assert _rel(v.grad, vr.grad) < 0.02


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
