"""Native block-fp8 grouped MoE-LoRA training path (no requant).

The Mistral-Large-4 / DeepSeek-V3 checkpoints ship experts as genuine 128x128 block-fp8 (weight
``float8_e4m3fn`` + ``weight_scale_inv``). The nvfp4 grouped path (``grouped_fp4_moe_train``)
requires a lossy fp8->nvfp4 requant (~6e-2 vs bf16). This path keeps the block-fp8 weights and runs
the base GEMM natively:

  * **base forward** — DeepGEMM ``m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`` (fp8-act x block-fp8-
    weight) on sm90 (H100/H200) + sm100 (B200/B300) — the DeepSeek-V3 MoE recipe. This is the
    datacenter target for the largest checkpoints. Preserves fp8 fidelity (~4e-3 vs bf16, ~17x better than
    requant). Fallback: dequant block-fp8 -> bf16 + ``torch._grouped_mm`` (verifiable on any GPU,
    but materializes bf16 weights so it is a dev/verify path, not memory-efficient at scale).
  * **base dX backward** — dequant the FORWARD block-fp8 weight to bf16 + grouped GEMM (bf16 grad
    is required; fp8 grad = 33% error, per grouped_train's rationale).
  * **LoRA + SwiGLU + routing** — reused verbatim from the nvfp4 path (base-agnostic).

Routing mirrors ``grouped_fp4_moe_train`` exactly (sort-by-expert, TILE-pad, contiguous grouping).
"""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

from .grouped_lora import grouped_lora_bwd, grouped_lora_fwd
from .grouped_train import TILE, _gmm, _lora_stack, _swiglu, _swiglu_bwd

LOG = get_logger(__name__)
_BACKEND_LOGGED = False


def fp8_block_grouped_available() -> bool:
    """Native fp8xfp8 (DeepGEMM sm90/sm100) OR the bf16-dequant fallback (any CUDA GPU)."""
    return torch.cuda.is_available()


def _dequant_fp8_3d(wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """Batched 128x128 block-fp8 dequant of a fused-3D weight ``[E, N, K]`` -> bf16."""
    E, N, K = wq.shape
    sr, sc = ws.shape[-2], ws.shape[-1]
    if sr and sc and N % sr == 0 and K % sc == 0:
        bn, bk = N // sr, K // sc
        return (
            (
                wq.to(torch.float32).reshape(E, sr, bn, sc, bk)
                * ws.float()[:, :, None, :, None]
            )
            .to(torch.bfloat16)
            .reshape(E, N, K)
        )
    from axolotl.kernels.quantize import dequantize_fp8

    return torch.stack([dequantize_fp8(wq[e], ws[e], torch.bfloat16) for e in range(E)])


def _fp8_train_backend() -> str:
    from .dequant_grouped import deepgemm_fp8fp8_available
    from .runtime import RUNTIME

    # marlin W8A16 forward is OPT-IN (moe_grouped_backend: marlin): ~2.6-2.9x the Triton fwd GEMM
    # at DeepSeek-scale dims, but the resident pack ~doubles fp8 expert weight residency.
    if RUNTIME.grouped_backend == "marlin":
        from .marlin_w4a16 import marlin_w4a16_available

        if marlin_w4a16_available():
            return "marlin_fp8"
    if deepgemm_fp8fp8_available():
        return "deepgemm_fp8"  # native fp8xfp8, sm90 (H200) / sm100 (B200,B300)
    if torch.cuda.is_available():
        return "blockfp8_triton"  # native fp8-read Triton (W8A16), portable incl. sm120
    return "dequant"


_FWD_PACK_KEYS = ("fp8_gate_up_fwd", "fp8_down_fwd")


def _fp8_fwd_pack(wq, ws, cache, which, probe_only=False, no_cache=False):
    """Marlin W8A16 forward pack for one projection: cached under the headroom opt-in
    (AXOLOTL_MARLIN_CACHE_MIN_FREE_GB), else built TRANSIENTLY for this GEMM and freed on
    return — memory-neutral at any depth (incl. FSDP, where the gathered weight is itself a
    fresh per-step copy). ``probe_only`` returns only a cache hit; ``no_cache`` never stores
    (chunked builds). Returns None to fall back to the Triton forward."""
    key = _FWD_PACK_KEYS[which]
    if cache is not None:
        if cache.get("fp8_fwd_failed"):
            return None
        got = cache.get(key)
        if got is not None:
            return got
    if probe_only:
        return None
    # marlin tiling: out(N) %64, reduction(K) %16; reduction must span >=2 scale groups (a
    # single group infers channelwise group_size=-1, not instantiated for fp8).
    if wq.size(1) % 64 or wq.size(2) % 16 or wq.size(2) <= 128:
        return None
    try:
        from .marlin_w4a16 import load_ext, marlin_w4a16_available
        from .marlin_w4a16.backend import _marlin_cache_under_fsdp
        from .marlin_w4a16.prep import prepare_blockfp8_fwd_weight

        if not marlin_w4a16_available():
            return None
        pack = prepare_blockfp8_fwd_weight(wq, ws, load_ext().gptq_marlin_repack)
        if not no_cache and cache is not None and _marlin_cache_under_fsdp(wq.device):
            cache[key] = pack
        return pack
    except Exception as ex:  # pragma: no cover - fall back to the Triton forward
        LOG.warning("fp8 marlin fwd pack build failed (%s); using Triton", ex)
        if cache is not None:
            cache["fp8_fwd_failed"] = True
        return None


def _marlin_fp8_forward(wq, ws, cache, which, x, m_indices):
    """Marlin W8A16 forward for one projection; None -> caller falls back to Triton.

    AXOLOTL_MARLIN_FWD_CHUNKS=<C> (transient mode only) builds the pack in C column chunks,
    each freed after its GEMM — the pack transient shrinks C-fold for one extra [Mt, N] concat
    copy. Column ranges of W are independent packs (128-block-aligned so the scale grid slices
    cleanly)."""
    import os

    from .marlin_w4a16.backend import marlin_dx_gemm
    from .marlin_w4a16.prep import FLOAT8_E4M3FN_ID

    E, Nout, Kin = wq.shape
    pack = _fp8_fwd_pack(wq, ws, cache, which, probe_only=True)
    if pack is not None:  # resident-cached
        return marlin_dx_gemm(pack, x, m_indices, Nout, Kin, FLOAT8_E4M3FN_ID)
    env_chunks = os.environ.get("AXOLOTL_MARLIN_FWD_CHUNKS")
    if env_chunks:
        chunks = int(env_chunks)
    else:
        # auto: bound the transient pack chunk at ~2.5 GiB (memory-neutral at large expert
        # dims — measured peak == the Triton baseline — for ~2% over the unchunked build)
        chunks = max(
            1, (E * Nout * Kin + (1 << 31) + (1 << 30) - 1) // ((1 << 31) + (1 << 30))
        )
    n_blocks = Nout // 128
    if chunks <= 1 or n_blocks < chunks:
        pack = _fp8_fwd_pack(wq, ws, cache, which)
        if pack is None:
            return None
        return marlin_dx_gemm(pack, x, m_indices, Nout, Kin, FLOAT8_E4M3FN_ID)
    nc = ((n_blocks + chunks - 1) // chunks) * 128
    outs = []
    for n0 in range(0, Nout, nc):
        n1 = min(n0 + nc, Nout)
        pack = _fp8_fwd_pack(
            wq[:, n0:n1], ws[:, n0 // 128 : n1 // 128], cache, which, no_cache=True
        )
        if pack is None:
            return None
        outs.append(marlin_dx_gemm(pack, x, m_indices, n1 - n0, Kin, FLOAT8_E4M3FN_ID))
        del pack
    return torch.cat(outs, dim=1)


def _fp8_base_forward(base, which, x, m_indices, offs):
    """Frozen block-fp8 base GEMM for gate_up (which=0) or down (which=1).
    base = (backend, (gu_fp8, gu_scale), (dn_fp8, dn_scale), pack_cache); weights [E, out, in]."""
    backend = base[0]
    wq, ws = base[1] if which == 0 else base[2]
    if backend == "marlin_fp8":
        y = _marlin_fp8_forward(wq, ws, base[3], which, x, m_indices)
        if y is not None:
            return y
        backend = "blockfp8_triton"
    if backend == "deepgemm_fp8":
        from .dequant_grouped import deepgemm_grouped_fp8_fp8

        # DeepGEMM's contiguous layout wants a per-row expert id; nt = weight is [E, out, in].
        return deepgemm_grouped_fp8_fp8(x, wq, ws, m_indices.repeat_interleave(TILE))
    if backend == "blockfp8_triton":
        from .blockfp8_gemm import blockfp8_grouped_gemm

        # native fp8-read: upcasts each 128x128 block in-register (no dequant materialization).
        return blockfp8_grouped_gemm(x, wq, ws, m_indices, TILE)
    # bf16-dequant fallback: W[E,out,in] -> x[Mt,in] @ W^T grouped = _gmm(x, W^T, offs)
    W = _dequant_fp8_3d(wq, ws)
    return _gmm(x, W.transpose(1, 2), offs)


def _fp8_base_dx(w_fp8, g, offs, m_indices, tile, dx_pack=None):
    """dX through the frozen block-fp8 weight: g[Mt,out] @ W[E,out,in] -> [Mt,in] (per group).

    ``dx_pack`` (moe_marlin_dx): the EXACT transposed marlin W8A16 pack (zero requant error —
    128x128 block scales are transpose-symmetric) -> one grouped GEMM, ~1.5-1.7x the Triton dX at
    sparse-MoE scale. Otherwise reads fp8 W in-register + upcasts to bf16 (bf16 grad required;
    grad-consistent with the forward, no bf16 weight materialization). Falls back to a bf16-dequant
    grouped_mm off-CUDA."""
    wq, ws = w_fp8
    if dx_pack is not None and g.is_cuda:
        from .marlin_w4a16.backend import marlin_dx_gemm
        from .marlin_w4a16.prep import FLOAT8_E4M3FN_ID

        E, N, K = wq.shape
        return marlin_dx_gemm(dx_pack, g, m_indices, K, N, FLOAT8_E4M3FN_ID)
    if g.is_cuda:
        from .blockfp8_gemm import blockfp8_grouped_dx

        return blockfp8_grouped_dx(g, wq, ws, m_indices, tile)
    W = _dequant_fp8_3d(wq, ws)  # [E, out, in]
    return _gmm(g, W, offs)


def _build_fp8_dx_packs(gate_up, down, dx_cache):
    """Build (once) the exact transposed marlin dX packs for both projections into ``dx_cache``
    (moe_marlin_dx opt-in). Returns (gu_pack, dn_pack) cache entries or (None, None).

    Cache-only (unlike the transient forward packs): a transient dX pack would ride ctx across
    the whole backward graph — all layers' packs alive at once."""
    from .marlin_w4a16.backend import _marlin_dx_enabled

    if not _marlin_dx_enabled() or dx_cache is None or dx_cache.get("fp8_dxT_failed"):
        return None, None
    try:
        import torch.distributed as dist

        from .marlin_w4a16 import load_ext, marlin_w4a16_available
        from .marlin_w4a16.backend import _marlin_cache_under_fsdp

        if not marlin_w4a16_available():
            return None, None
        # Caching a per-layer pack holds a copy across depth; under FSDP it additionally
        # re-materializes the un-sharded experts (OOM) — same policy as the nvfp4 marlin cache:
        # only with the explicit headroom opt-in.
        if (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
            and not _marlin_cache_under_fsdp(gate_up[0].device)
        ):
            return None, None
        if "fp8_gate_up_dxT" not in dx_cache:
            from .marlin_w4a16.prep import prepare_blockfp8_dx_weight

            ext = load_ext()
            for key, (wq, ws) in (
                ("fp8_gate_up_dxT", gate_up),
                ("fp8_down_dxT", down),
            ):
                # marlin tiling: dX output dim (fwd K) %64, reduction (fwd N) %16, and the
                # reduction must span >=2 scale groups (single group infers channelwise
                # group_size=-1, not instantiated for fp8).
                if wq.size(2) % 64 or wq.size(1) % 16 or wq.size(1) <= 128:
                    return None, None
                dx_cache[key] = prepare_blockfp8_dx_weight(
                    wq, ws, ext.gptq_marlin_repack
                )
    except Exception as ex:  # pragma: no cover - fall back to the Triton dX
        LOG.warning("fp8 marlin dX pack build failed (%s); using Triton dX", ex)
        dx_cache["fp8_dxT_failed"] = True
        return None, None
    return dx_cache.get("fp8_gate_up_dxT"), dx_cache.get("fp8_down_dxT")


class _Fp8BlockGroupedExperts(torch.autograd.Function):
    """x[Mt,H] -> gate_up(block-fp8 base + LoRA) -> gated-act -> down(...) -> [Mt,H].
    Frozen block-fp8 experts; trainable LoRA A/B. Mirrors _GroupedExperts but for block-fp8."""

    @staticmethod
    def forward(
        ctx,
        x,
        base,
        weight_recipe,
        Agu,
        Bgu,
        Adn,
        Bdn,
        m_indices,
        offs,
        scaling,
        limit,
        act_type,
        dx_packs=None,
    ):
        E = Agu.size(0)
        gu = _fp8_base_forward(base, 0, x, m_indices, offs)
        from .swiglu_epilogue import grouped_lora_swiglu_fwd, swiglu_epilogue_enabled

        if swiglu_epilogue_enabled():
            gu, h, xAg = grouped_lora_swiglu_fwd(
                x, Agu, Bgu, scaling, offs, E, gu, act_type, limit
            )
        else:
            gu, xAg = grouped_lora_fwd(x, Agu, Bgu, scaling, offs, E, residual=gu)
            h = _swiglu(gu, limit, act_type).to(x.dtype)
        dn = _fp8_base_forward(base, 1, h, m_indices, offs)
        dn, hAd = grouped_lora_fwd(h, Adn, Bdn, scaling, offs, E, residual=dn)
        ctx.save_for_backward(x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices)
        ctx.weight_recipe, ctx.scaling, ctx.limit, ctx.act_type = (
            weight_recipe,
            scaling,
            limit,
            act_type,
        )
        ctx.dx_packs = dx_packs  # (gu_pack, dn_pack) marlin dX or None
        return dn

    @staticmethod
    def backward(ctx, d_dn):
        x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices = ctx.saved_tensors
        s, lim, act_type = ctx.scaling, ctx.limit, ctx.act_type
        d_dn = d_dn.contiguous().to(x.dtype)
        gu_fp8, dn_fp8 = ctx.weight_recipe()  # ((wq,ws), (wq,ws)) — FSDP-safe re-read
        gu_pack, dn_pack = ctx.dx_packs or (None, None)
        E = Agu.size(0)
        tile = x.size(0) // m_indices.numel()  # routing pad granularity (TILE)
        dh = _fp8_base_dx(dn_fp8, d_dn, offs, m_indices, tile, dx_pack=dn_pack)
        dh, dAdn, dBdn = grouped_lora_bwd(
            d_dn, h, Adn, Bdn, hAd, s, offs, E, residual=dh
        )
        dgu = _swiglu_bwd(dh, gu, lim, act_type).to(x.dtype)
        del dh
        dx = _fp8_base_dx(gu_fp8, dgu, offs, m_indices, tile, dx_pack=gu_pack)
        dx, dAgu, dBgu = grouped_lora_bwd(
            dgu, x, Agu, Bgu, xAg, s, offs, E, residual=dx
        )
        return (
            dx,
            None,
            None,
            dAgu,
            dBgu,
            dAdn,
            dBdn,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def grouped_fp8_moe_train(
    hidden,
    idx,
    wts,
    gate_up_fp8,
    gate_up_scale,
    down_fp8,
    down_scale,
    gup_lora,
    down_lora,
    limit,
    act_type="silu",
    weight_recipe=None,
    dx_cache=None,
):
    """Training-capable native block-fp8 grouped MoE forward (no requant).

    hidden[N,H], idx/wts[N,topk]. Experts block-fp8: gate_up_fp8 [E,2I,H] float8 + gate_up_scale
    [E,2I/128,H/128] fp32; down_fp8 [E,H,I] + down_scale. ``*_lora`` = (A,B,scaling) scattermoe
    layout. Returns [N,H]; differentiable to hidden + LoRA A/B. ``weight_recipe`` -> ((gu_fp8,
    gu_scale),(dn_fp8,dn_scale)) for the FSDP-safe backward re-read (defaults to the forward
    tensors)."""
    global _BACKEND_LOGGED
    if weight_recipe is None:
        weight_recipe = lambda: (  # noqa: E731
            (gate_up_fp8, gate_up_scale),
            (down_fp8, down_scale),
        )
    N, H = hidden.shape
    E, twoI, _ = gate_up_fp8.shape
    I = down_fp8.shape[2]  # noqa: E741
    dev = hidden.device
    backend = _fp8_train_backend()
    if not _BACKEND_LOGGED:
        _BACKEND_LOGGED = True
        LOG.info("grouped block-fp8 MoE base-GEMM backend: %s", backend)
    tile = TILE

    # --- routing: sort-by-expert, TILE-pad, contiguous grouping (mirrors grouped_fp4_moe_train) ---
    flat = idx.reshape(-1)
    _tok = torch.arange(N, device=dev).repeat_interleave(idx.size(1))
    _wf = wts.reshape(-1)
    if (flat < 0).any():
        keep = flat >= 0
        flat, _tok, _wf = flat[keep], _tok[keep], _wf[keep]
    order = flat.argsort()
    rep = _tok[order]
    wflat = _wf[order]
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    ptiles = (counts + tile - 1) // tile
    roff = torch.cat([ptiles.new_zeros(1), ptiles.cumsum(0)]) * tile
    coff = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    padded_row = roff[exp_sorted] + (
        torch.arange(exp_sorted.numel(), device=dev) - coff[exp_sorted]
    )
    m_indices = torch.repeat_interleave(
        torch.arange(E, dtype=torch.int32, device=dev), ptiles
    )
    offs = (ptiles * tile).cumsum(0).to(torch.int32)
    Mt = int(ptiles.sum()) * tile

    base = (backend, (gate_up_fp8, gate_up_scale), (down_fp8, down_scale), dx_cache)
    dx_packs = _build_fp8_dx_packs(
        (gate_up_fp8, gate_up_scale), (down_fp8, down_scale), dx_cache
    )
    if dx_packs == (None, None):
        dx_packs = None
    Agu, Bgu, sgu = _lora_stack(gup_lora, E, H, twoI)
    Adn, Bdn, sdn = _lora_stack(down_lora, E, I, H)
    assert sgu == sdn, "gate_up/down LoRA scaling must match"
    lim = float(limit) if limit is not None else 1e30

    from .scatter_fuse import padded_gather, weighted_scatter

    A = padded_gather(hidden, rep, padded_row, Mt)
    dn = _Fp8BlockGroupedExperts.apply(
        A,
        base,
        weight_recipe,
        Agu,
        Bgu,
        Adn,
        Bdn,
        m_indices,
        offs,
        sgu,
        lim,
        act_type,
        dx_packs,
    )
    return weighted_scatter(dn, wflat, rep, padded_row, N)
