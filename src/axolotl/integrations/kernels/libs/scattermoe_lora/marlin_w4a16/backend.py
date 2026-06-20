"""Marlin W4A16 forward base for the grouped NVFP4 MoE (sm120).

Prep the frozen NVFP4 experts into Marlin layout ONCE (``build_marlin_forward_base``); run the
bf16-activation forward GEMM (``marlin_base_forward``) on the pre-scattered, pad-to-TILE grouped
layout that ``grouped_fp4_moe_train`` already builds. Marlin reads the activation directly in bf16
and gathers per expert via identity ``sorted_token_ids`` + per-block ``expert_ids`` (derived from the
per-tile ``m_indices``) + ``top_k=1`` — validated bit-correct on that layout.

The marlin path pads to TILE=64 (vs CUTLASS's forced 128): at thin-M that halves the padded rows
(2.7x -> 1.4x), which is where the speedup comes from (each expert weight is read once either way).
"""

from __future__ import annotations

import torch

from . import load_ext
from .prep import marlin_make_workspace_new, marlin_moe_gemm, prepare_nvfp4_weight_for_marlin

# Pad granularity for the marlin path. Must equal the marlin MoE block (BSM) so each padded tile maps
# to exactly one expert; 64 halves the thin-M padding vs CUTLASS's 128 and recovers Marlin's speed.
MARLIN_TILE = 64
_BSM = 64


def _pt(nv, E, dev):
    p = getattr(nv, "per_tensor_scale", None)
    if p is None:
        return torch.ones(E, device=dev)
    p = p.reshape(-1).float()
    return p.expand(E) if p.numel() == 1 else p


def _cached_prep(nv, size_n, size_k, ext, cache, key):
    """Prep one NVFP4 weight -> Marlin layout, cached (experts are frozen). Prefer a persistent
    module-level ``cache`` dict keyed by ``key`` — under FSDP2 the gathered param is a fresh tensor
    each step, so a module cache (not a per-tensor attr) avoids re-repacking 256 experts/forward.
    Falls back to a per-tensor attribute, then a one-shot compute."""
    if cache is not None and key in cache:
        return cache[key]
    cached = getattr(nv, "_marlin_w4a16", None)
    if cached is None:
        E, dev = nv.qdata.size(0), nv.qdata.device
        cached = prepare_nvfp4_weight_for_marlin(
            nv.qdata, nv.scale, _pt(nv, E, dev), size_n, size_k, torch.bfloat16, ext.gptq_marlin_repack)
        try:
            nv._marlin_w4a16 = cached
        except (AttributeError, RuntimeError):
            pass
    if cache is not None:
        cache[key] = cached
    return cached


def build_marlin_forward_base(gate_up_nv, down_nv, cache=None):
    """Prep frozen NVFP4 experts -> Marlin forward weights (cached; experts are frozen).

    gate_up_nv: NVFP4Tensor [E, 2I, H]; down_nv: NVFP4Tensor [E, H, I]. ``cache`` is the optional
    persistent dict (the same ``mxfp4_cache`` threaded through ``grouped_fp4_moe_train``).
    Returns the base tuple ``('marlin', (gu_w, dn_w), (twoI, H, I), workspace)`` consumed by
    ``marlin_base_forward`` and dispatched in ``grouped_train._base_forward``."""
    ext = load_ext()
    twoI = gate_up_nv.qdata.size(1)
    H = gate_up_nv.qdata.size(2) * 2  # gate_up K = H (packed /2)
    I = down_nv.qdata.size(2) * 2     # down K = I (packed /2)
    gu_w = _cached_prep(gate_up_nv, twoI, H, ext, cache, "marlin_gate_up")
    dn_w = _cached_prep(down_nv, H, I, ext, cache, "marlin_down")
    ws = marlin_make_workspace_new(torch.device(gate_up_nv.qdata.device), 4)
    return ("marlin", (gu_w, dn_w), (twoI, H, I), ws)


def marlin_route(m_indices, Mt, dev):
    """Build Marlin's routing for the pre-scattered pad-TILE layout: identity ``sorted_token_ids``
    (the activation is already grouped+padded), per-BSM-block ``expert_ids`` from the per-tile
    ``m_indices``, ``num_tokens_post_padded`` = Mt, and unit ``topk_weights`` (top_k=1)."""
    tile = Mt // m_indices.numel()
    si = torch.arange(Mt, dtype=torch.int32, device=dev)
    ei = m_indices.repeat_interleave(tile // _BSM).to(torch.int32)
    ntpp = torch.tensor([Mt], dtype=torch.int32, device=dev)
    tw = torch.ones(Mt, 1, dtype=torch.float32, device=dev)
    return si, ei, ntpp, tw


def marlin_base_forward(base, which, x, m_indices):
    """Frozen-expert base GEMM via Marlin (bf16 act). which=0 gate_up (x[Mt,H] -> [Mt,2I]);
    which=1 down (x[Mt,I] -> [Mt,H]). ``m_indices`` is the per-TILE expert id from the routing.
    Returns bf16."""
    ext = load_ext()
    _, (gu_w, dn_w), (twoI, H, I), ws = base
    Mt = x.size(0)
    si, ei, ntpp, tw = marlin_route(m_indices, Mt, x.device)
    if which == 0:
        return marlin_moe_gemm(ext, x, gu_w[0], gu_w[1], gu_w[2], ws, si, ei, ntpp, tw, _BSM, 1, False, Mt, twoI, H)
    return marlin_moe_gemm(ext, x, dn_w[0], dn_w[1], dn_w[2], ws, si, ei, ntpp, tw, _BSM, 1, False, Mt, H, I)
