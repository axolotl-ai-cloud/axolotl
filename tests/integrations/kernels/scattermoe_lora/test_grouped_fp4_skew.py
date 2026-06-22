"""Skew-robustness tests for the grouped/marlin NVFP4 LoRA path.

The grouped LoRA op must be correct AND memory-bounded for ANY routing distribution — balanced,
power-law, or pathological all-to-one. A capacity-padded implementation (cap = max per-expert
tokens) blows up under skew (peak ~ E * max_cap * dim instead of ~ routed_tokens * dim) and OOMs at
all_to_one; the robust implementation (ragged scatter / bulk-bmm+tail-scatter) does not.

Verifiable (deterministic) guards live here:
  - correctness across skew (cosine vs bf16 oracle) — CI-safe
  - memory bound: peak transient <= C * routed_tokens * dim — catches the capacity blowup
  - no-OOM under a simulated GPU memory cap at all_to_one
Speed is ratio-based and marked `perf` (noisy).
"""

from __future__ import annotations

import pytest
import torch

# reuse the validated helpers from the sibling correctness module
from .test_grouped_fp4_train import (
    DEV,
    _bf16_oracle,
    _cos,
    quantize_nvfp4,
)

_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] in (
    10,
    12,
)
pytestmark = pytest.mark.skipif(
    not _BLACKWELL, reason="grouped fp4 needs sm100/sm120 (Blackwell)"
)

SKEWS = ["balanced", "moderate", "extreme", "all_to_one"]

# Correctness uses a modest E (fast). Memory uses real E=128 so a capacity-padded impl's all_to_one
# blowup (E * N * dim, most experts empty-but-padded) dwarfs balanced (~E*(routed/E)*dim) -> ~16x.
_CFG = dict(
    name="gemma4",
    E=32,
    H=2816,
    I=704,
    topk=8,
    N=512,
    r=16,
    act_type="gelu_tanh",
    limit=1e30,
    scaling=2.0,
)
_CFG_MEM = dict(_CFG, E=128, N=1024)


def make_routing(N: int, E: int, top_k: int, skew: str, seed: int = 0):
    """Deterministic (idx[N,top_k], wts[N,top_k]) with a controllable routing skew.
    balanced: uniform distinct top_k; moderate: power-law (prob ~ 1/rank); extreme: 90% of tokens
    to experts 0..top_k-1; all_to_one: every token -> experts 0..top_k-1 (the adversarial case that
    forces cap == N for a capacity-padded impl)."""
    g = torch.Generator().manual_seed(seed)
    hot = torch.arange(top_k)
    rows = []
    if skew == "balanced":
        rows = [torch.randperm(E, generator=g)[:top_k] for _ in range(N)]
    elif skew == "all_to_one":
        rows = [hot.clone() for _ in range(N)]
    elif skew == "extreme":
        # 90% hot, 10% random
        rows = [
            hot.clone() if (i % 10) else torch.randperm(E, generator=g)[:top_k]
            for i in range(N)
        ]
    elif skew == "moderate":
        w = 1.0 / torch.arange(1, E + 1).float()  # zipf-ish over expert rank
        for _ in range(N):
            perm = torch.multinomial(w, top_k, replacement=False, generator=g)
            rows.append(perm)
    else:
        raise ValueError(skew)
    idx = torch.stack(rows).to(DEV)
    wts = (
        torch.softmax(torch.randn(N, top_k, generator=g), -1).to(torch.bfloat16).to(DEV)
    )
    return idx, wts


def _build(cfg, skew, seed=0):
    E, H, I, topk, N, r = cfg["E"], cfg["H"], cfg["I"], cfg["topk"], cfg["N"], cfg["r"]  # noqa: E741
    twoI = 2 * I
    torch.manual_seed(seed)
    Wgu = torch.randn(E, twoI, H, device=DEV, dtype=torch.bfloat16) * 0.04
    Wdn = torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16) * 0.04
    gu_nv, dn_nv = quantize_nvfp4(Wgu), quantize_nvfp4(Wdn)
    from axolotl.integrations.kernels.libs.scattermoe_lora.dequant_grouped import (
        nvfp4_dequant_bf16,
    )

    pt = torch.ones(E, device=DEV)
    Wgu_b = nvfp4_dequant_bf16(gu_nv.qdata, gu_nv.scale, pt)
    Wdn_b = nvfp4_dequant_bf16(dn_nv.qdata, dn_nv.scale, pt)
    hidden = torch.randn(N, H, device=DEV, dtype=torch.bfloat16) * 0.5
    idx, wts = make_routing(N, E, topk, skew, seed)

    def mk(*s):
        return (
            torch.randn(*s, device=DEV, dtype=torch.bfloat16) * 0.02
        ).requires_grad_(True)

    lora = (mk(r * E, H), mk(twoI, r * E), mk(r * E, I), mk(H, r * E))
    return hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, lora


def _run(cfg, gu_nv, dn_nv, hidden, idx, wts, lora, cache):
    # cache is a persistent dict (like the per-module mxfp4 cache in real training): the marlin path
    # frees qdata after the first call and reads the cache thereafter, so repeated calls on the same
    # nv tensors MUST share one cache.
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_moe_train,
    )

    Agu, Bgu, Adn, Bdn = lora
    s = cfg["scaling"]
    return grouped_fp4_moe_train(
        hidden,
        idx,
        wts,
        gu_nv,
        dn_nv,
        (Agu, Bgu, s),
        (Adn, Bdn, s),
        cfg["limit"],
        "nvfp4",
        act_type=cfg["act_type"],
        mxfp4_cache=cache,
    )


# ---------------------------------------------------------------------------
# 1. Correctness across skew (deterministic, CI-safe)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("skew", SKEWS)
def test_correctness_across_skew(skew):
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend")
    hidden, idx, wts, gu_nv, dn_nv, Wgu_b, Wdn_b, lora = _build(_CFG, skew)
    Agu, Bgu, Adn, Bdn = lora
    out = _run(
        _CFG,
        gu_nv,
        dn_nv,
        hidden.clone(),
        idx,
        wts,
        (Agu.detach(), Bgu.detach(), Adn.detach(), Bdn.detach()),
        {},
    )
    assert torch.isfinite(out).all(), f"{skew}: non-finite output"
    ref = _bf16_oracle(
        hidden.clone(),
        idx,
        wts,
        Wgu_b,
        Wdn_b,
        Agu,
        Bgu,
        Adn,
        Bdn,
        _CFG["act_type"],
        _CFG["limit"],
        _CFG["scaling"],
    )
    cos = _cos(out, ref)
    assert cos > 0.97, f"{skew}: fwd cosine {cos:.4f} < 0.97"


# ---------------------------------------------------------------------------
# 2. Memory bound across skew (THE skew guard) — a robust op's peak must NOT scale with routing
#    skew. Measure each skew's peak transient and assert it stays within a small factor of the
#    BALANCED peak (scale-robust, no magic absolute). A capacity-padded impl (cap=max) blows up at
#    all_to_one (E*N*dim, most experts empty-but-padded) -> many x balanced -> fails here.
# ---------------------------------------------------------------------------
def _peak_transient(cfg, skew):
    hidden, idx, wts, gu_nv, dn_nv, _, _, lora = _build(cfg, skew)
    det = tuple(p.detach() for p in lora)
    cache = {}
    o = _run(cfg, gu_nv, dn_nv, hidden.clone(), idx, wts, det, cache)
    del o  # warmup + build cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    _run(cfg, gu_nv, dn_nv, hidden.clone(), idx, wts, det, cache)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 1e6


def test_memory_bound_across_skew():
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend")
    cfg = _CFG_MEM
    peaks = {s: _peak_transient(cfg, s) for s in SKEWS}
    base = peaks["balanced"]
    msg = " | ".join(f"{s}={peaks[s]:.0f}MB" for s in SKEWS)
    # robust: skew peak within 1.5x of balanced. capacity-padded all_to_one is many x -> fails.
    for s in SKEWS:
        assert peaks[s] <= 1.5 * base, f"skew '{s}' transient blowup vs balanced: {msg}"


# ---------------------------------------------------------------------------
# 3. No-OOM under a simulated GPU cap at the adversarial all_to_one routing.
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_no_oom_all_to_one_under_cap():
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend")
    total = torch.cuda.get_device_properties(0).total_memory
    # cap this process to a fraction that comfortably fits the robust path but not an E*cap blowup
    hidden, idx, wts, gu_nv, dn_nv, _, _, lora = _build(_CFG, "all_to_one")
    needed = _CFG["N"] * _CFG["topk"] * max(2 * _CFG["I"], _CFG["H"]) * 2
    frac = min(
        0.5, (needed * 12 + 4 * 2**30) / total
    )  # robust needs ~12x routed + model; blowup needs ~Ex more
    torch.cuda.set_per_process_memory_fraction(frac, 0)
    try:
        out = _run(
            _CFG,
            gu_nv,
            dn_nv,
            hidden.clone(),
            idx,
            wts,
            tuple(p.detach() for p in lora),
            {},
        )
        out.float().sum().backward() if out.requires_grad else None
        assert torch.isfinite(out).all()
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)


# ---------------------------------------------------------------------------
# 4. Speed: skew must not catastrophically slow the op vs balanced (ratio-based, noisy).
# ---------------------------------------------------------------------------
@pytest.mark.perf
def test_speed_skew_within_factor_of_balanced():
    from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
        grouped_fp4_available,
    )

    if not grouped_fp4_available("nvfp4"):
        pytest.skip("no grouped fp4 backend")

    def timed(skew, it=10, wu=4):
        hidden, idx, wts, gu_nv, dn_nv, _, _, lora = _build(_CFG, skew)
        det = tuple(p.detach() for p in lora)
        cache = {}
        for _ in range(wu):
            _run(_CFG, gu_nv, dn_nv, hidden.clone(), idx, wts, det, cache)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(it):
            _run(_CFG, gu_nv, dn_nv, hidden.clone(), idx, wts, det, cache)
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / it

    tb = timed("balanced")
    ts = timed("extreme")
    assert ts <= 4.0 * tb, f"extreme-skew step {ts:.1f}ms > 4x balanced {tb:.1f}ms"
