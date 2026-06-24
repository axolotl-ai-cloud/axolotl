"""Marlin W4A16 forward base for the grouped NVFP4 MoE (Ampere+; sm80/89/90/100/120).

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
from .prep import (
    marlin_make_workspace_new,
    marlin_moe_gemm,
    prepare_nvfp4_weight_for_marlin,
)

# Must equal the marlin MoE block (BSM) so each padded tile maps to exactly one expert; 64
# halves the thin-M padding vs CUTLASS's 128.
MARLIN_TILE = 64
_BSM = 64

_BASE_SCATTER_LUT: dict[torch.device, torch.Tensor] = {}


def _pt(nv, E, dev):
    p = getattr(nv, "per_tensor_scale", None)
    if p is None:
        return torch.ones(E, device=dev)
    p = p.reshape(-1).float()
    return p.expand(E) if p.numel() == 1 else p


def _build_base_scatter(dev: torch.device) -> torch.Tensor:
    """Build and cache the 1024-entry scatter LUT: qdata flat nibble pos -> marlin flat nibble pos.
    Probes the base tile (N_t=64, K_t=16) using gptq_marlin_repack; result is device-resident."""
    if dev in _BASE_SCATTER_LUT:
        return _BASE_SCATTER_LUT[dev]
    ext = load_ext()
    perm = torch.empty(0, dtype=torch.int, device=dev)
    N_t, K_t = 64, 16
    scatter = torch.full((N_t * K_t,), -1, dtype=torch.int32, device="cpu")
    for n_b in range(N_t):
        for k_b in range(K_t):
            qd = torch.zeros(N_t, K_t // 2, dtype=torch.uint8, device=dev)
            byte_idx = k_b // 2
            qd[n_b, byte_idx] = 0x0F if k_b % 2 == 0 else 0xF0
            qw_i = qd.view(torch.int32).T.contiguous()
            qw_m = ext.gptq_marlin_repack(qw_i, perm, K_t, N_t, 4, False)
            flat = qw_m.reshape(-1).cpu().tolist()
            for wi, word in enumerate(flat):
                if word == 0:
                    continue
                for bit in range(8):
                    if (word >> (bit * 4)) & 0xF == 0xF:
                        scatter[n_b * K_t + k_b] = wi * 8 + bit
                        break
                else:
                    continue
                break
    lut = scatter.to(dev)
    _BASE_SCATTER_LUT[dev] = lut
    return lut


def _cached_prep(nv, size_n, size_k, ext, cache, key):
    """Prep one NVFP4 weight -> Marlin layout, cached (experts are frozen). Prefer a persistent
    module-level ``cache`` dict keyed by ``key`` — under FSDP2 the gathered param is a fresh tensor
    each step, so a module cache (not a per-tensor attr) avoids re-repacking 256 experts/forward.
    Falls back to a per-tensor attribute, then a one-shot compute.

    On the first build (single-GPU only), saves original scales + per-tensor scale in cache under
    key+"_bwd" for the fused backward dequant, then frees nv.qdata to drop the duplicate 4-bit copy.
    Under FSDP the gathered param is fresh each step so weight_recipe() re-gathers it anyway; skips
    the free when distributed is active."""
    bwd_key = key + "_bwd"
    if cache is not None and key in cache:
        return cache[key]
    assert not getattr(nv, "is_swizzled_scales", False), (
        "marlin_w4a16 needs raw (non-swizzled) NVFP4 scales; got is_swizzled_scales=True"
    )
    cached = getattr(nv, "_marlin_w4a16", None)
    qdata_fresh = cached is None
    if cached is None:
        E, dev = nv.qdata.size(0), nv.qdata.device
        cached = prepare_nvfp4_weight_for_marlin(
            nv.qdata,
            nv.scale,
            _pt(nv, E, dev),
            size_n,
            size_k,
            torch.bfloat16,
            ext.gptq_marlin_repack,
        )
        try:
            nv._marlin_w4a16 = cached
        except (AttributeError, RuntimeError):
            pass
    if cache is not None:
        cache[key] = cached
        if qdata_fresh and bwd_key not in cache:
            import torch.distributed as dist

            _is_distributed = (
                dist.is_available()
                and dist.is_initialized()
                and dist.get_world_size() > 1
            )
            if not _is_distributed:
                E, dev = nv.qdata.size(0), nv.qdata.device
                # Reference (don't clone) the original scales for the backward dequant: the
                # marlin scales subsample [:, 1::2] so they're lossy and unusable for backward.
                # Single-GPU: nv.scale stays live on the frozen param, so a clone just dups ~1.3 GiB.
                # _pt may return .expand() with stride(0)==0; Triton needs a real stride.
                pt_bwd = _pt(nv, E, dev)
                if pt_bwd.stride(0) == 0:
                    pt_bwd = pt_bwd.contiguous()
                cache[bwd_key] = (nv.scale, pt_bwd, E, size_n, size_k)
                # 3-D [E, N, 0] placeholder (not flat empty(0)): NVFP4Tensor clone/save needs
                # ndim==3, and numel()==0 still routes through the marlin qdata-free paths.
                try:
                    _N = nv.qdata.size(1)
                    nv.qdata.data = torch.empty(
                        (E, _N, 0), dtype=nv.qdata.dtype, device=dev
                    )
                except (AttributeError, RuntimeError):
                    pass
    return cached


def _qdata_sizes(nv, cache_key, cache):
    """Return (size_n, size_k_packed, device) for the NVFP4 weight.
    Falls back to bwd cache when nv.qdata has been freed (single-GPU memory-free path)."""
    qdata = nv.qdata
    if qdata.numel() > 0:
        return qdata.size(1), qdata.size(2), torch.device(qdata.device)
    if cache is not None:
        bwd = cache.get(cache_key + "_bwd")
        if bwd is not None:
            # bwd = (scale, pt, E, size_n, size_k); size_k_packed = size_k // 2
            size_n, size_k = bwd[3], bwd[4]
            dev = bwd[0].device
            return size_n, size_k // 2, dev
    raise RuntimeError(
        f"marlin_w4a16: nv.qdata has been freed but no backward cache found for '{cache_key}'. "
        "Pass a module-level mxfp4_cache to grouped_fp4_moe_train."
    )


def build_marlin_forward_base(gate_up_nv, down_nv, cache=None):
    """Prep frozen NVFP4 experts -> Marlin forward weights (cached; experts are frozen).

    gate_up_nv: NVFP4Tensor [E, 2I, H]; down_nv: NVFP4Tensor [E, H, I]. ``cache`` is the optional
    persistent dict (the same ``mxfp4_cache`` threaded through ``grouped_fp4_moe_train``).
    Returns the base tuple ``('marlin', (gu_w, dn_w), (twoI, H, I), workspace, cache)`` consumed by
    ``marlin_base_forward`` and dispatched in ``grouped_train._base_forward``. The cache is threaded
    through so the backward can retrieve the (scale, pt, E, N, K) bwd tuple from cache[key+"_bwd"]."""
    ext = load_ext()
    twoI, H_packed, dev = _qdata_sizes(gate_up_nv, "marlin_gate_up", cache)
    H = H_packed * 2
    _, I_packed, _ = _qdata_sizes(down_nv, "marlin_down", cache)
    I = I_packed * 2  # noqa: E741
    gu_w = _cached_prep(gate_up_nv, twoI, H, ext, cache, "marlin_gate_up")
    dn_w = _cached_prep(down_nv, H, I, ext, cache, "marlin_down")
    ws = marlin_make_workspace_new(dev, 4)
    return ("marlin", (gu_w, dn_w), (twoI, H, I), ws, cache)


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
    _, (gu_w, dn_w), (twoI, H, I), ws = base[0], base[1], base[2], base[3]  # noqa: E741
    Mt = x.size(0)
    si, ei, ntpp, tw = marlin_route(m_indices, Mt, x.device)
    if which == 0:
        return marlin_moe_gemm(
            ext,
            x,
            gu_w[0],
            gu_w[1],
            gu_w[2],
            ws,
            si,
            ei,
            ntpp,
            tw,
            _BSM,
            1,
            False,
            Mt,
            twoI,
            H,
        )
    return marlin_moe_gemm(
        ext,
        x,
        dn_w[0],
        dn_w[1],
        dn_w[2],
        ws,
        si,
        ei,
        ntpp,
        tw,
        _BSM,
        1,
        False,
        Mt,
        H,
        I,
    )


def marlin_bwd_data(base):
    """Return (gu_bwd, dn_bwd) tuples from the cache, or None if not available.
    Each bwd tuple is (original_scale, pt, E, size_n, size_k); marlin qw is at cache[fwd_key][0]."""
    cache = base[4] if len(base) > 4 else None
    if cache is None:
        return None
    gu_bwd = cache.get("marlin_gate_up_bwd")
    dn_bwd = cache.get("marlin_down_bwd")
    if gu_bwd is None or dn_bwd is None:
        return None
    gu_qw = cache.get("marlin_gate_up")
    dn_qw = cache.get("marlin_down")
    if gu_qw is None or dn_qw is None:
        return None
    return (gu_qw[0], gu_bwd), (dn_qw[0], dn_bwd)
