"""fp8_block grouped experts with an expert dim that isn't a whole 128-block (Gemma-4's
moe_intermediate_size=704 under FineGrainedFP8Config): the base GEMMs avoid DeepGEMM and the
marlin packs, and forward + grads match a dequantized fp32 reference."""

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.scattermoe_lora import fp8_grouped
from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import RUNTIME

DEV = "cuda"
BLOCK = 128

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
        reason="requires CUDA with FP8 support (sm89+)",
    ),
]


@pytest.fixture(autouse=True)
def default_runtime():
    RUNTIME.reset()
    yield
    RUNTIME.reset()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def _quantize(E, N, K):
    """128x128 block-fp8 with a ceil-div scale grid; returns (wq, ws, exact dequant)."""
    w = torch.randn(E, N, K, device=DEV) * 0.05
    rn, rk = -(-N // BLOCK), -(-K // BLOCK)
    padded = F.pad(w, (0, rk * BLOCK - K, 0, rn * BLOCK - N))
    blocks = padded.reshape(E, rn, BLOCK, rk, BLOCK)
    ws = blocks.abs().amax(dim=(2, 4)).clamp_min(1e-12) / 448.0
    scale = ws.repeat_interleave(BLOCK, 1).repeat_interleave(BLOCK, 2)[:, :N, :K]
    wq = (w / scale).to(torch.float8_e4m3fn)
    return wq, ws, wq.float() * scale


def _reference(hidden, idx, wts, w_gu, w_dn, Ag, Bg, Ad, Bd, s):
    E, two_i, H = w_gu.shape
    inter, r = two_i // 2, Ag.shape[0] // E
    Ag, Bg, Ad, Bd = (t.float() for t in (Ag, Bg, Ad, Bd))
    out = hidden.new_zeros(hidden.shape, dtype=torch.float32)
    for e in range(E):
        tok, k = (idx == e).nonzero(as_tuple=True)
        x = hidden[tok].float()
        gu = (
            x @ w_gu[e].t()
            + s * (x @ Ag.reshape(E, r, H)[e].t()) @ Bg.reshape(two_i, E, r)[:, e].t()
        )
        h = F.gelu(gu[:, :inter], approximate="tanh") * gu[:, inter:]
        dn = (
            h @ w_dn[e].t()
            + s * (h @ Ad.reshape(E, r, inter)[e].t()) @ Bd.reshape(H, E, r)[:, e].t()
        )
        out = out.index_add(0, tok, wts[tok, k, None].float() * dn)
    return out


@pytest.mark.parametrize("grouped_backend", [None, "marlin"])
def test_unaligned_intermediate_matches_reference(monkeypatch, grouped_backend):
    if grouped_backend == "marlin":
        from axolotl.integrations.kernels.libs.scattermoe_lora.marlin_w4a16 import (
            marlin_w4a16_available,
        )

        if not marlin_w4a16_available():
            pytest.skip("marlin backend unavailable")
    RUNTIME.grouped_backend = grouped_backend
    seen = []
    base_forward = fp8_grouped._fp8_base_forward

    def spy(base, *args):
        seen.append(base[0])
        return base_forward(base, *args)

    monkeypatch.setattr(fp8_grouped, "_fp8_base_forward", spy)

    torch.manual_seed(0)
    E, H, inter, N, topk, r, s = 4, 256, 192, 96, 2, 8, 2.0
    gu_q, gu_s, gu_ref = _quantize(E, 2 * inter, H)
    dn_q, dn_s, dn_ref = _quantize(E, H, inter)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx = torch.randint(0, E, (N, topk), device=DEV)
    wts = torch.softmax(torch.randn(N, topk, device=DEV), -1)
    params = [
        torch.randn(shape, device=DEV, dtype=torch.bfloat16) * 0.05
        for shape in ((r * E, H), (2 * inter, r * E), (r * E, inter), (H, r * E))
    ]

    def run(fn, *weights):
        x = hidden.clone().requires_grad_(True)
        lora = [p.clone().requires_grad_(True) for p in params]
        out = fn(x, *weights, lora)
        out.float().square().mean().backward()
        return out.detach(), [x.grad] + [p.grad for p in lora]

    out, grads = run(
        lambda x, lora: fp8_grouped.grouped_fp8_moe_train(
            x,
            idx,
            wts,
            gu_q,
            gu_s,
            dn_q,
            dn_s,
            (lora[0], lora[1], s),
            (lora[2], lora[3], s),
            None,
            act_type="gelu_tanh",
            dx_cache={},
        )
    )
    ref, ref_grads = run(
        lambda x, lora: _reference(x, idx, wts, gu_ref, dn_ref, *lora, s)
    )

    expected = "marlin_fp8" if grouped_backend == "marlin" else "blockfp8_triton"
    assert set(seen) == {expected}
    assert _rel(out, ref) < 3e-2
    for got, want in zip(grads, ref_grads, strict=True):
        assert _rel(got, want) < 5e-2
