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
    prepare_transposed_dx_weight,
)


def _marlin_dx_mode() -> str | bool:
    """Transposed-Marlin backward dX mode: False (off, dequant dX) | 'resident' (packs on GPU —
    fastest, ~doubles expert residency) | 'offload' (packs pinned in CPU RAM, streamed per-layer
    one-ahead during backward — GPU holds ~2 layers' packs). Config knob ``moe_marlin_dx``
    (RUNTIME.marlin_dx); env ``AXOLOTL_MARLIN_DX`` (0 | 1 | offload) overrides at runtime."""
    import os

    from ..runtime import RUNTIME, _canon_marlin_dx

    env = os.environ.get("AXOLOTL_MARLIN_DX")
    if env is not None:
        return _canon_marlin_dx(env if env == "offload" else env != "0")
    return _canon_marlin_dx(RUNTIME.marlin_dx)


class _DxPackStreamer:
    """One-ahead H2D prefetcher for CPU-pinned dX packs.

    Backward visits layers in a fixed reverse order, so the first backward RECORDS the access
    sequence (synchronous copies) and every later step PREFETCHES entry i+1 on a side stream while
    entry i's GEMM runs — the copy rides the DMA engine under compute. GPU residency = the in-use
    pack + the in-flight prefetch (~2 layers' packs)."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.order: list = []  # slot keys in backward order (first pass records)
        self.recorded = False
        self.inflight: dict = {}  # slot -> (gpu_tensors, event)

    def _copy(self, cpu_pack, dev):
        with torch.cuda.stream(self.stream):
            gpu = tuple(
                t.to(dev, non_blocking=True) if isinstance(t, torch.Tensor) else t
                for t in cpu_pack
            )
            ev = torch.cuda.Event()
            ev.record(self.stream)
        return gpu, ev

    def _prefetch(self, slot, cpu_pack, dev):
        if slot not in self.inflight:
            self.inflight[slot] = self._copy(cpu_pack, dev)

    def acquire(self, slot, cpu_pack, dev, lookup):
        """Return GPU tensors for ``slot``, then queue the next slot in the recorded order.
        ``lookup(slot) -> cpu_pack`` resolves the prefetch target."""
        if not self.recorded:
            if (
                slot in self.order
            ):  # second pass over a recorded slot -> order is complete
                self.recorded = True
            else:
                self.order.append(slot)
        got = self.inflight.pop(slot, None)
        if got is None:
            got = self._copy(cpu_pack, dev)
        gpu, ev = got
        torch.cuda.current_stream().wait_event(ev)
        if self.recorded and self.order:
            try:
                nxt = self.order[(self.order.index(slot) + 1) % len(self.order)]
                nxt_pack = lookup(nxt)
                if nxt_pack is not None:
                    self._prefetch(nxt, nxt_pack, dev)
            except ValueError:
                pass
        return gpu


_DX_STREAMER: _DxPackStreamer | None = None
# registry of offloaded packs: slot -> cpu_pack, so the streamer can prefetch across layers
# (each layer has its own module cache; slots are (id(cache), key)).
_DX_CPU_PACKS: dict = {}


def _dx_streamer() -> _DxPackStreamer:
    global _DX_STREAMER
    if _DX_STREAMER is None:
        _DX_STREAMER = _DxPackStreamer()
    return _DX_STREAMER


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


def _marlin_cache_under_fsdp(device) -> bool:
    """Whether to persist marlin-prepped experts in the per-layer module cache under FSDP.

    DEFAULT: NO. The per-layer cache holds each layer's gathered (un-sharded) marlin experts, so it
    accumulates the whole model across depth (~0.6 GiB/proj × 2 × num_layers — e.g. ~94 GiB for GLM-5.2)
    on top of the resident sharded model → OOM. Profiled: a layer's marlin re-prep is only a ~17 GiB
    transient that frees immediately, so re-prepping every forward is far cheaper than caching, and the
    per-layer MoE fwd+bwd peak is ~17 GiB — fits easily beside the ~60 GiB resident shard.

    OPT-IN for roomy GPUs (e.g. B200, or small models) where the full cache fits: set
    ``AXOLOTL_MARLIN_CACHE_MIN_FREE_GB=<N>`` to cache while ≥ N GiB VRAM is free (``0`` = always cache).
    Unset → never cache under FSDP."""
    import os

    val = os.environ.get("AXOLOTL_MARLIN_CACHE_MIN_FREE_GB")
    if val is None:
        return False
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:  # pylint: disable=broad-except
        return True
    return free > float(val) * (1024**3)


def _cached_prep(nv, size_n, size_k, ext, cache, key):
    """Prep one NVFP4 weight -> Marlin layout, cached (experts are frozen). Prefer a persistent
    module-level ``cache`` dict keyed by ``key`` — under FSDP2 the gathered param is a fresh tensor
    each step, so a module cache (not a per-tensor attr) avoids re-repacking 256 experts/forward.
    Falls back to a per-tensor attribute, then a one-shot compute.

    On the first build of a rank-resident weight (single GPU or DDP), saves original scales +
    per-tensor scale in cache under key+"_bwd" for the fused backward dequant, then frees nv.qdata
    to drop the duplicate 4-bit copy. Under FSDP/DeepSpeed the gathered param is fresh each step so
    weight_recipe() re-gathers it anyway; skips the free there."""
    import torch.distributed as dist

    from ..runtime import RUNTIME

    _gathered = (
        RUNTIME.experts_gathered
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
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
        # Stash the full-layer marlin weight on the tensor ONLY when rank-resident. Under FSDP2 the
        # gathered param is a fresh tensor every forward AND every backward recompute, so this attr
        # never produces a cache hit — it only pins ~9 GiB/proj alive on each layer's gathered tensor,
        # and during the backward sweep FSDP keeps several layers' gathered tensors live at once →
        # the marlin weights accumulate across depth and OOM (bypassing the module-cache guard below).
        if not _gathered:
            try:
                nv._marlin_w4a16 = cached
            except (AttributeError, RuntimeError):
                pass
    dxt_key = key + "_dxT"
    if (
        cache is not None
        and dxt_key not in cache
        and nv.qdata.numel() > 0
        and _marlin_dx_mode()
        and (not _gathered or _marlin_cache_under_fsdp(nv.qdata.device))
        # marlin tiling: dxT output dim (fwd size_k) needs %64, reduction (fwd size_n) %16
        and size_k % 64 == 0
        and size_n % 16 == 0
    ):
        E, dev = nv.qdata.size(0), nv.qdata.device
        pack = prepare_transposed_dx_weight(
            nv.qdata, nv.scale, _pt(nv, E, dev), size_k, size_n, ext.gptq_marlin_repack
        )
        # offload: pinned-CPU storage streamed per-layer in backward (one-ahead prefetch),
        # bounding GPU residency to ~2 layers' packs instead of the whole model's.
        stash_dx_pack(cache, dxt_key, pack, _marlin_dx_mode())
    if cache is not None and not _gathered:
        # Rank-resident: persist the marlin weight and drop the duplicate 4-bit qdata.
        cache[key] = cached
        if qdata_fresh and bwd_key not in cache:
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
    elif cache is not None and _marlin_cache_under_fsdp(nv.qdata.device):
        # FSDP2: the gathered weight is a fresh, transient tensor per forward (meant to reshard), so
        # persisting its marlin copy for EVERY layer re-materializes the whole un-sharded model on
        # each rank — OOMing large checkpoints. The cache is still a real perf win (no re-repacking
        # 256 experts/forward) when VRAM is roomy, so cache only while there's headroom; once it
        # tightens, fall through to per-forward re-prep (the transient ``_marlin_w4a16`` attr handles
        # in-forward reuse and dies on reshard). Don't free qdata here — backward re-gathers it.
        cache[key] = cached
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


def marlin_dx_data(base):
    """(gu_dxT, dn_dxT) marlin weight tuples for the transposed backward-dX GEMM, or None if the
    packs weren't built (env-disabled, no cache, FSDP without headroom, or odd dims)."""
    cache = base[4] if len(base) > 4 else None
    if cache is None:
        return None
    gu = cache.get("marlin_gate_up_dxT")
    dn = cache.get("marlin_down_dxT")
    if gu is None or dn is None:
        return None
    return gu, dn


def resolve_dx_pack(w, dev):
    """Resolve a dX-pack cache entry to GPU tensors: pass through a resident tuple, or stream in a
    ('cpu', slot) offload marker via the one-ahead prefetcher."""
    if isinstance(w, tuple) and len(w) == 2 and w[0] == "cpu":
        slot = w[1]
        w = _dx_streamer().acquire(slot, _DX_CPU_PACKS[slot], dev, _DX_CPU_PACKS.get)
        for t in w:
            if isinstance(t, torch.Tensor):
                t.record_stream(torch.cuda.current_stream())
    return w


def stash_dx_pack(cache, key, pack, mode):
    """Store a built dX pack per the mode: GPU-resident tuple, or pinned-CPU + ('cpu', slot)
    marker for the streamer."""
    if mode == "offload":
        slot = (id(cache), key)
        _DX_CPU_PACKS[slot] = tuple(
            t.cpu().pin_memory() if isinstance(t, torch.Tensor) else t for t in pack
        )
        del pack
        cache[key] = ("cpu", slot)
    else:
        cache[key] = pack


def marlin_dx_gemm(pack, g, m_indices, size_n, size_k, b_type_id, moe_block_size=64):
    """Generic transposed-pack marlin dX: ``g [Mt, size_k] -> [Mt, size_n]`` on the contiguous-
    grouped layout. ``pack`` = (qw, scales, global_scale_or_None), resident or offload marker."""
    ext = load_ext()
    qw, sc, gs = resolve_dx_pack(pack, g.device)
    Mt = g.size(0)
    si, ei, ntpp, tw = marlin_route(m_indices, Mt, g.device)
    ws = marlin_make_workspace_new(torch.device(g.device.type, g.device.index or 0), 4)
    return marlin_moe_gemm(
        ext,
        g,
        qw,
        sc,
        gs,
        ws,
        si,
        ei,
        ntpp,
        tw,
        moe_block_size,
        1,
        False,
        Mt,
        size_n,
        size_k,
        b_type_id=b_type_id,
    )


def marlin_base_dx(base, dx_data, which, g, m_indices):
    """Backward base-dX via the transposed-Marlin pack: one W4A16 grouped GEMM, no dequant.
    which=0: dx = dgu @ Wgu  (g[Mt,2I] -> [Mt,H]); which=1: dh = d_dn @ Wdn  (g[Mt,H] -> [Mt,I]).
    Same routing layout as the forward (identical m_indices/tile)."""
    ext = load_ext()
    _, _, (twoI, H, I), ws = base[0], base[1], base[2], base[3]  # noqa: E741
    w = resolve_dx_pack(dx_data[which], g.device)
    Mt = g.size(0)
    si, ei, ntpp, tw = marlin_route(m_indices, Mt, g.device)
    size_n, size_k = (H, twoI) if which == 0 else (I, H)
    return marlin_moe_gemm(
        ext,
        g,
        w[0],
        w[1],
        w[2],
        ws,
        si,
        ei,
        ntpp,
        tw,
        _BSM,
        1,
        False,
        Mt,
        size_n,
        size_k,
    )
