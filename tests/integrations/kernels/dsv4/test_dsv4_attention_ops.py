"""Parity tests for the DSV4 attention custom ops (sliding, CSA dense, CSA top-k gather).

Each kernel's fwd/bwd launch is registered as a ``torch.ops.axolotl.*`` custom op; these
tests check the ops exist and that the autograd entry points still match a dense fp32
eager reference (fwd outputs + input grads, including the per-head sink grad).
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEV = "cuda"
B, H, S, D = 2, 8, 128, 128
T, K, W = 64, 32, 32


def _rel(a, b):
    return (a.float() - b.float()).norm().item() / max(b.float().norm().item(), 1e-12)


def _sink_softmax(logits, sinks):
    """Append the per-head sink as a logit-only column, softmax, drop it."""
    Bb, Hh, Ss, _ = logits.shape
    sink = sinks.float().view(1, Hh, 1, 1).expand(Bb, Hh, Ss, 1)
    return torch.cat([logits, sink], -1).softmax(-1)[..., :-1]


def _window_mask(s_q, s_k, window, device):
    i = torch.arange(s_q, device=device)[:, None]
    j = torch.arange(s_k, device=device)[None, :]
    return (j <= i) & (i - j < window)


def test_ops_registered():
    from axolotl.integrations.kernels.libs import dsv4  # noqa: F401

    for name in (
        "dsv4_sliding_attn_fwd",
        "dsv4_sliding_attn_bwd",
        "dsv4_csa_attn_fwd",
        "dsv4_csa_attn_bwd",
        "dsv4_csa_topk_attn_fwd",
        "dsv4_csa_topk_attn_bwd",
    ):
        assert hasattr(torch.ops.axolotl, name), name


def test_sliding_attn_matches_eager():
    from axolotl.integrations.kernels.libs.dsv4 import sliding_attn

    torch.manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    qr, kr, vr, sr = (t.detach().clone().requires_grad_() for t in (q, k, v, sinks))

    out = sliding_attn(q, k, v, sinks, None, W)
    out.float().pow(2).mean().backward()

    scale = D**-0.5
    scores = qr.float() @ kr[:, 0].float()[:, None].transpose(-1, -2) * scale
    scores = scores.masked_fill(~_window_mask(S, S, W, DEV), float("-inf"))
    p = _sink_softmax(scores, sr)
    ref = p @ vr[:, 0].float()[:, None]
    ref.pow(2).mean().backward()

    assert _rel(out, ref) < 0.02
    assert _rel(q.grad, qr.grad) < 0.05
    assert _rel(k.grad, kr.grad) < 0.05
    assert _rel(v.grad, vr.grad) < 0.05
    assert _rel(sinks.grad, sr.grad) < 0.05


def test_csa_attn_matches_eager():
    from axolotl.integrations.kernels.libs.dsv4 import csa_attn

    torch.manual_seed(1)
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    kvs = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    kvc = torch.randn(B, 1, T, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    # additive block bias: finite bias on causally-valid compressed slots, -inf elsewhere
    comp_valid = (
        torch.arange(T, device=DEV)[None, :] * 2 <= torch.arange(S, device=DEV)[:, None]
    )
    bb = torch.where(
        comp_valid, torch.randn(B, S, T, device=DEV), float("-inf")
    ).unsqueeze(1)
    qr, ksr, kcr, sr = (
        t.detach().clone().requires_grad_() for t in (q, kvs, kvc, sinks)
    )

    out = csa_attn(q, kvs, kvc, bb, sinks, None, W)
    out.float().pow(2).mean().backward()

    scale = D**-0.5
    ss = qr.float() @ ksr[:, 0].float()[:, None].transpose(-1, -2) * scale
    ss = ss.masked_fill(~_window_mask(S, S, W, DEV), float("-inf"))
    sc = qr.float() @ kcr[:, 0].float()[:, None].transpose(-1, -2) * scale + bb.float()
    p = _sink_softmax(torch.cat([ss, sc], -1), sr)
    ref = (
        p[..., :S] @ ksr[:, 0].float()[:, None]
        + p[..., S:] @ kcr[:, 0].float()[:, None]
    )
    ref.pow(2).mean().backward()

    assert _rel(out, ref) < 0.02
    assert _rel(q.grad, qr.grad) < 0.05
    assert _rel(kvs.grad, ksr.grad) < 0.05
    assert _rel(kvc.grad, kcr.grad) < 0.05
    assert _rel(sinks.grad, sr.grad) < 0.05


def test_csa_attn_topk_matches_eager():
    from axolotl.integrations.kernels.libs.dsv4 import csa_attn_topk

    torch.manual_seed(2)
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    kvs = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    kvc = torch.randn(B, 1, T, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    sinks = torch.randn(H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    # unique top-k per position (duplicates would double-count in any implementation);
    # early positions get -1 (invalid) tails like the real indexer emits
    idx = torch.stack(
        [
            torch.stack([torch.randperm(T, device=DEV)[:K] for _ in range(S)])
            for _ in range(B)
        ]
    ).to(torch.int32)
    idx[:, : S // 8, K // 2 :] = -1
    qr, ksr, kcr, sr = (
        t.detach().clone().requires_grad_() for t in (q, kvs, kvc, sinks)
    )

    out = csa_attn_topk(q, kvs, kvc, idx, sinks, None, W)
    out.float().pow(2).mean().backward()

    sel = torch.zeros(B, S, T, dtype=torch.bool, device=DEV)
    for b in range(B):
        for s in range(S):
            row = idx[b, s]
            sel[b, s, row[row >= 0].long()] = True
    scale = D**-0.5
    ss = qr.float() @ ksr[:, 0].float()[:, None].transpose(-1, -2) * scale
    ss = ss.masked_fill(~_window_mask(S, S, W, DEV), float("-inf"))
    sc = qr.float() @ kcr[:, 0].float()[:, None].transpose(-1, -2) * scale
    sc = sc.masked_fill(~sel[:, None], float("-inf"))
    p = _sink_softmax(torch.cat([ss, sc], -1), sr)
    ref = (
        p[..., :S] @ ksr[:, 0].float()[:, None]
        + p[..., S:] @ kcr[:, 0].float()[:, None]
    )
    ref.pow(2).mean().backward()

    assert _rel(out, ref) < 0.02
    assert _rel(q.grad, qr.grad) < 0.05
    assert _rel(kvs.grad, ksr.grad) < 0.05
    assert _rel(kvc.grad, kcr.grad) < 0.05
    assert _rel(sinks.grad, sr.grad) < 0.05
