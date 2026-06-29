"""Length-aware dispatch for GLM-5.2 DSA attention — robust across short and long sequences.

The sparse gather kernel only wins when ``S >> index_topk`` (it trades MMA locality for skipping
non-selected keys). Below a hardware-dependent crossover, a DENSE flash over the (absorbed) shared
KV with the top-k applied as an additive mask is faster; and at ``S <= index_topk`` the top-k
selects every causal key, so it degenerates to plain causal attention anyway.

``mla_attn`` dispatches on ``S``: dense flash below the crossover, gather above. Both produce the
same ``out_latent`` and are differentiable, so training is correct in either regime. The crossover
is **auto-calibrated once** per ``(topk, H, dtype, device)`` by microbenchmarking the two paths —
so it adapts to sm120 (~22k) vs a datacenter GPU (much lower) instead of hardcoding a threshold.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .attention_mla_absorb import DQK, mla_absorb_attn
from .config import KV_LORA_RANK

_CROSSOVER: dict[tuple, int] = {}
_GATHER_OK: dict[int, bool] = {}


def _gather_supported(device) -> bool:
    """Whether to use the sparse absorbed-MLA gather kernel (vs the dense fallback).

    Validated on sm120 (Ada / consumer-Blackwell) and sm90 (Hopper). The earlier sm90 blockers are
    fixed: the backward's nondeterministic invalid-PC (the full autotune grid trialing — each config
    is individually clean) is avoided by forcing a single config on sm90 (``_bwd_prune_sm90_safe`` in
    ``attention_mla_absorb``), and the leading-all-masked-block softmax NaN is guarded in the kernel.
    Set ``GLM_DSA_DISABLE_GATHER=1`` to force the dense path, or ``GLM_DSA_FORCE_GATHER=1`` to use the
    gather on an unlisted arch."""
    import os

    if os.environ.get("GLM_DSA_DISABLE_GATHER"):
        return False
    if os.environ.get("GLM_DSA_FORCE_GATHER"):
        return True
    dev = getattr(device, "index", None)
    if dev is None:
        dev = torch.cuda.current_device()
    if dev not in _GATHER_OK:
        _GATHER_OK[dev] = torch.cuda.get_device_capability(dev) in ((9, 0), (12, 0))
    return _GATHER_OK[dev]


def _topk_causal_mask(topk_idx, Skv, q_offset, dtype, device, seq_q=None, seq_k=None):
    """Additive [B,1,S,Skv] mask: 0 at selected-and-causal global keys, -inf elsewhere. ``topk_idx``
    references GLOBAL key positions; query i (local) is global position ``q_offset + i``. Under
    sample packing, ``seq_q`` [B,S] / ``seq_k`` [B,Skv] further forbid cross-document keys (the
    selected set may include earlier-document keys when ``topk == Skv``)."""
    B, S = topk_idx.shape[:2]
    sel = torch.zeros(B, S, Skv, dtype=torch.bool, device=device)
    sel.scatter_(-1, topk_idx.long(), True)
    kpos = torch.arange(Skv, device=device)
    qpos = q_offset + torch.arange(S, device=device)
    keep = sel & (kpos[None, None, :] <= qpos[None, :, None])
    if seq_q is not None and seq_k is not None:
        keep = keep & (seq_k[:, None, :] == seq_q[:, :, None])
    mask = torch.zeros(B, 1, S, Skv, dtype=dtype, device=device)
    return mask.masked_fill_(~keep.unsqueeze(1), float("-inf"))


def dense_masked_out_latent(
    q_abs, k_shared, topk_idx, scale, q_offset=0, seq_q=None, seq_k=None
):
    """Dense flash over the absorbed shared KV with the top-k as an additive mask (O(S) memory via
    SDPA; the mask itself is O(S·Skv)). Differentiable. Faster than the gather below the crossover.
    ``k_shared`` may be the global (gathered) KV under context parallel; ``q_offset`` is local q0's
    global position. ``seq_q``/``seq_k`` enable per-document masking under sample packing."""
    B, H, S, _ = q_abs.shape
    Skv = k_shared.shape[1]
    c_kv = k_shared[..., :KV_LORA_RANK]
    k_e = k_shared.unsqueeze(1).expand(B, H, Skv, k_shared.shape[-1])
    v_e = c_kv.unsqueeze(1).expand(B, H, Skv, KV_LORA_RANK)
    mask = _topk_causal_mask(
        topk_idx, Skv, q_offset, q_abs.dtype, q_abs.device, seq_q, seq_k
    )
    return F.scaled_dot_product_attention(q_abs, k_e, v_e, attn_mask=mask, scale=scale)


def _bench(fn, it=8):
    import time

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return time.time() - t


def calibrate_crossover(topk, H, dtype, device, B=1) -> int:
    """Microbench gather vs dense at increasing S; return the smallest S where the gather wins
    (memoized). Falls back to a sparsity heuristic if a probe OOMs."""
    key = (topk, H, str(dtype), torch.cuda.get_device_name(device))
    if key in _CROSSOVER:
        return _CROSSOVER[key]
    crossover = None
    for mult in (2, 3, 4, 6, 8, 12, 16):
        S = topk * mult
        try:
            q = torch.randn(B, H, S, DQK, device=device, dtype=dtype)
            k = torch.randn(B, S, DQK, device=device, dtype=dtype)
            idx = torch.stack(
                [torch.arange(topk, device=device).clamp(max=s).int() for s in range(S)]
            ).unsqueeze(0)
            tg = _bench(lambda q=q, k=k, idx=idx: mla_absorb_attn(q, k, idx, 1.0).sum())
            td = _bench(
                lambda q=q, k=k, idx=idx: dense_masked_out_latent(q, k, idx, 1.0).sum()
            )
        except torch.cuda.OutOfMemoryError:
            crossover = crossover or S  # dense OOMed -> must use gather from here
            break
        if tg < td:
            crossover = S
            break
    crossover = crossover or topk * 16
    _CROSSOVER[key] = crossover
    return crossover


def mla_attn(
    q_abs,
    k_shared,
    topk_idx,
    scale,
    q_offset=0,
    crossover: int | None = None,
    seq_q=None,
    seq_k=None,
):
    """Differentiable DSA attention, dispatched by total key length (the sparsity that matters):
    dense flash below the (auto-calibrated) crossover, sparse gather above. ``k_shared`` may be the
    global gathered KV under context parallel; ``q_offset`` is local query 0's global position.
    Returns out_latent [B,H,S,kv_lora].

    Under sample packing (``seq_q``/``seq_k`` document ids set) the gather is doc-aware: it masks
    cross-document keys in-kernel (``seq_k[idx] == seq_q[s]``), so packing keeps the sparse path
    instead of falling back to the dense per-document mask."""
    Skv = k_shared.shape[1]
    topk = topk_idx.shape[-1]
    # Check arch support BEFORE calibrating — calibrate_crossover benchmarks the gather, so on an
    # unsupported GPU calibration would itself compile/run the gather Triton path we're avoiding.
    if _gather_supported(q_abs.device):
        if crossover is None:
            crossover = calibrate_crossover(
                topk, q_abs.shape[1], q_abs.dtype, q_abs.device
            )
        if Skv >= crossover:
            return mla_absorb_attn(
                q_abs, k_shared, topk_idx, scale, q_offset, seq_q, seq_k
            )
    return dense_masked_out_latent(
        q_abs, k_shared, topk_idx, scale, q_offset, seq_q, seq_k
    )
