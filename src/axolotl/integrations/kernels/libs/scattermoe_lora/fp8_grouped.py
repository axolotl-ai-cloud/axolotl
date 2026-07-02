"""Native block-fp8 grouped MoE-LoRA training path (no requant).

The Mistral-Large-4 / DeepSeek-V3 checkpoints ship experts as genuine 128x128 block-fp8 (weight
``float8_e4m3fn`` + ``weight_scale_inv``). The nvfp4 grouped path (``grouped_fp4_moe_train``)
requires a lossy fp8->nvfp4 requant (~6e-2 vs bf16). This path keeps the block-fp8 weights and runs
the base GEMM natively:

  * **base forward** — DeepGEMM ``m_grouped_gemm_fp8_fp8_bf16_nt_contiguous`` (fp8-act x block-fp8-
    weight) on sm90 (H100/H200) + sm100 (B200/B300) — the DeepSeek-V3 MoE recipe. This is the
    datacenter target for the 1050B model. Preserves fp8 fidelity (~4e-3 vs bf16, ~17x better than
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

    if deepgemm_fp8fp8_available():
        return "deepgemm_fp8"  # native fp8xfp8, sm90 (H200) / sm100 (B200,B300)
    if torch.cuda.is_available():
        return "blockfp8_triton"  # native fp8-read Triton (W8A16), portable incl. sm120
    return "dequant"


def _fp8_base_forward(base, which, x, m_indices, offs):
    """Frozen block-fp8 base GEMM for gate_up (which=0) or down (which=1).
    base = (backend, (gu_fp8, gu_scale), (dn_fp8, dn_scale)); weights [E, out, in]."""
    backend = base[0]
    wq, ws = base[1] if which == 0 else base[2]
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
    per the ``moe_marlin_dx`` mode. Returns (gu_pack, dn_pack) cache entries or (None, None)."""
    from .marlin_w4a16.backend import _marlin_dx_mode, stash_dx_pack

    mode = _marlin_dx_mode()
    if not mode or dx_cache is None:
        return None, None
    try:
        import torch.distributed as dist

        from .marlin_w4a16 import load_ext, marlin_w4a16_available
        from .marlin_w4a16.backend import _marlin_cache_under_fsdp

        if not marlin_w4a16_available():
            return None, None
        # Under FSDP the gathered weight is a fresh full copy per step; caching a per-layer pack
        # re-materializes the un-sharded experts across depth (OOM) — same policy as the nvfp4
        # marlin cache: only with the explicit headroom opt-in.
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
                # marlin tiling: dX output dim (fwd K) %64, reduction (fwd N) %16; block-fp8
                # dims are 128-multiples so this only guards exotic shapes.
                if wq.size(2) % 64 or wq.size(1) % 16:
                    return None, None
                pack = prepare_blockfp8_dx_weight(wq, ws, ext.gptq_marlin_repack)
                stash_dx_pack(dx_cache, key, pack, mode)
    except Exception as ex:  # pragma: no cover - fall back to the Triton dX
        LOG.warning("fp8 marlin dX pack build failed (%s); using Triton dX", ex)
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

    base = (backend, (gate_up_fp8, gate_up_scale), (down_fp8, down_scale))
    dx_packs = _build_fp8_dx_packs(
        (gate_up_fp8, gate_up_scale), (down_fp8, down_scale), dx_cache
    )
    if dx_packs == (None, None):
        dx_packs = None
    Agu, Bgu, sgu = _lora_stack(gup_lora, E, H, twoI)
    Adn, Bdn, sdn = _lora_stack(down_lora, E, I, H)
    assert sgu == sdn, "gate_up/down LoRA scaling must match"
    lim = float(limit) if limit is not None else 1e30

    A = hidden.new_zeros(Mt, H).index_copy(0, padded_row, hidden[rep])
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
    out = hidden.new_zeros(N, H)
    return out.index_add(
        0, rep, (dn[padded_row] * wflat[:, None].to(dn.dtype)).to(out.dtype)
    )
