"""Native OCP-MX fp8 grouped MoE-LoRA training path (``moe_grouped_mode: mxfp8``).

Experts are mxfp8: weight ``float8_e4m3fn`` [E, N, K] + e8m0 block scale [E, N, K/32] (32-wide MX
blocks along the contraction axis). The base GEMM reads the fp8 weight in-register and applies the
e8m0 scale during accumulation — no bf16 weight materialization (W8A16):

  * **base forward / dX** — Triton ``mxfp8_grouped_gemm`` / ``mxfp8_grouped_dx`` (arch-portable incl.
    sm120). Fallback: dequant mxfp8 -> bf16 + ``torch._grouped_mm`` (off-CUDA verify path only).
  * **LoRA + SwiGLU + routing** — reused verbatim from the fp8_block / nvfp4 paths (base-agnostic).

Same 1 byte/elem storage as ``fp8_block`` but hardware-MX-native (finer 32-wide scale); mirrors
``grouped_fp8_moe_train`` exactly (sort-by-expert, TILE-pad, contiguous grouping).
"""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

from .grouped_lora import grouped_lora_bwd, grouped_lora_fwd
from .grouped_train import TILE, _gmm, _lora_stack, _swiglu, _swiglu_bwd

LOG = get_logger(__name__)
_BACKEND_LOGGED = False


def mxfp8_grouped_available() -> bool:
    """Native mxfp8-read Triton (any CUDA GPU) OR the bf16-dequant fallback."""
    return torch.cuda.is_available()


def _dequant_mx_3d(wq: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """Dequant a fused-3D mxfp8 weight ``[E, N, K]`` (e4m3) + scale ``[E, N, K/32]`` (e8m0) -> bf16."""
    E, N, K = wq.shape
    nb = ws.shape[-1]
    bk = K // nb
    wf = wq.to(torch.float32).reshape(E, N, nb, bk)
    s = torch.exp2(ws.view(torch.uint8).to(torch.float32) - 127.0)[..., None]
    return (wf * s).to(torch.bfloat16).reshape(E, N, K)


# ---- hardware-MX (Blackwell e8m0 tensor-core) forward ----------------------------------------
# sm100 (datacenter Blackwell) exposes a native grouped MX GEMM (torch._scaled_grouped_mm accepts
# e4m3 data + e8m0 scales); sm120 (consumer Blackwell) only has the per-tile torch._scaled_mm MX
# path (no grouped primitive), where per-expert launch overhead makes it slower than the Triton
# W8A16 kernel — so hardware-MX is auto-selected only on sm100. See _mx_backend().
_GROUPED_MX_OK = (
    None  # None=untested, True/False after a one-time self-check vs the per-tile loop
)
_FORCE_BACKEND = None  # test override: 'mma' | 'triton' | None


def set_mxfp8_backend(which) -> None:
    """Force the base-GEMM backend ('mma' | 'triton' | None=auto). For tests / benchmarking; the
    per-tile hardware-MX path is exercisable on sm120 this way even though auto-select skips it."""
    global _FORCE_BACKEND
    _FORCE_BACKEND = which


def _hw_mx_grouped_available() -> bool:
    """True on sm100 (datacenter Blackwell): native grouped MX GEMM via torch._scaled_grouped_mm
    (e8m0 scales in hardware). sm120 lacks the grouped primitive; sm90 has no e8m0 MX MMA."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 10 and hasattr(
        torch, "_scaled_grouped_mm"
    )


def _to_blocked(s):
    from torchao.prototype.mx_formats.utils import to_blocked

    return to_blocked(s)


def _mma_per_group(aq, asc, wq, ws_e8, offs):
    """Hardware-MX W8A8 forward via a per-expert torch._scaled_mm loop (verified on sm120). Contiguous
    TILE-padded groups -> each expert's rows are a [Me, K] slice, Me a multiple of 128."""
    E, N, _ = wq.shape
    starts = torch.cat([offs.new_zeros(1), offs]).tolist()
    out = aq.new_empty(aq.size(0), N, dtype=torch.bfloat16)
    for e in range(E):
        t0, t1 = starts[e], starts[e + 1]
        if t1 == t0:
            continue
        ab = _to_blocked(asc[t0:t1])
        wb = _to_blocked(ws_e8[e])
        out[t0:t1] = torch._scaled_mm(
            aq[t0:t1], wq[e].t(), ab, wb, out_dtype=torch.bfloat16
        )
    return out


def _mma_grouped(aq, asc, wq, ws_e8, offs, m_indices):
    """Single-launch hardware-MX grouped GEMM (sm100). SF blocked per group; safety-checked once vs
    _mma_per_group by the caller, so a layout mismatch degrades to the loop rather than corrupting."""
    scale_a = _to_blocked(asc)
    scale_b = torch.stack([_to_blocked(ws_e8[e]) for e in range(wq.size(0))])
    return torch._scaled_grouped_mm(
        aq, wq.transpose(-2, -1), scale_a, scale_b, offs=offs, out_dtype=torch.bfloat16
    )


def _mx_hw_forward(x, wq, ws, m_indices, offs):
    from .cutlass_fp4.quant_mxfp8 import mxfp8_quant

    aq, asc = mxfp8_quant(x)  # e4m3 [Mt,K], e8m0 [Mt,K/32]
    ws_e8 = ws.view(torch.float8_e8m0fnu)
    global _GROUPED_MX_OK
    if _hw_mx_grouped_available() and _GROUPED_MX_OK is not False:
        try:
            g_out = _mma_grouped(aq, asc, wq, ws_e8, offs, m_indices)
            if (
                _GROUPED_MX_OK is None
            ):  # validate once against the trusted per-tile loop
                ref = _mma_per_group(aq, asc, wq, ws_e8, offs)
                rel = (
                    (g_out.float() - ref.float()).norm()
                    / ref.float().norm().clamp_min(1e-9)
                ).item()
                _GROUPED_MX_OK = rel < 5e-2
                LOG.info(
                    "mxfp8 _scaled_grouped_mm self-check rel=%.2e -> %s",
                    rel,
                    "grouped" if _GROUPED_MX_OK else "per-tile fallback",
                )
            if _GROUPED_MX_OK:
                return g_out
        except Exception as ex:  # noqa: BLE001
            _GROUPED_MX_OK = False
            LOG.warning(
                "mxfp8 _scaled_grouped_mm unavailable (%s); using per-tile _scaled_mm.",
                ex,
            )
    return _mma_per_group(aq, asc, wq, ws_e8, offs)


def _mx_base_forward(base, which, x, m_indices, offs):
    """Frozen mxfp8 base GEMM for gate_up (which=0) or down (which=1).
    base = (backend, (gu_q, gu_scale), (dn_q, dn_scale)); weights [E, out, in]."""
    backend = base[0]
    wq, ws = base[1] if which == 0 else base[2]
    if backend == "mxfp8_mma":
        return _mx_hw_forward(x, wq, ws, m_indices, offs)
    if backend == "mxfp8_triton":
        from .mxfp8_gemm import mxfp8_grouped_gemm

        return mxfp8_grouped_gemm(x, wq, ws, m_indices, TILE)
    # bf16-dequant fallback: W[E,out,in] -> x[Mt,in] @ W^T grouped = _gmm(x, W^T, offs)
    W = _dequant_mx_3d(wq, ws)
    return _gmm(x, W.transpose(1, 2), offs)


def _mx_base_dx(w_mx, g, offs, m_indices, tile):
    """dX through the frozen mxfp8 weight: g[Mt,out] @ W[E,out,in] -> [Mt,in] (per group). Reads
    fp8 W in-register + applies the e8m0 scale (bf16 grad; grad-consistent with the forward). Falls
    back to a bf16-dequant grouped_mm off-CUDA."""
    wq, ws = w_mx
    if g.is_cuda:
        from .mxfp8_gemm import mxfp8_grouped_dx

        return mxfp8_grouped_dx(g, wq, ws, m_indices, tile)
    W = _dequant_mx_3d(wq, ws)  # [E, out, in]
    return _gmm(g, W, offs)


def _mx_backend() -> str:
    """Base-GEMM forward backend. Auto: hardware-MX W8A8 (``mxfp8_mma``) on sm100 where the native
    grouped MX GEMM exists; portable W8A16 Triton (``mxfp8_triton``) on sm120/sm90 (sm120's per-tile
    hardware path is slower than Triton — launch overhead). The backward dX is always the Triton
    W8A16 kernel (contract-N needs N-blocked weight scales the HW path can't cheaply provide, and it
    is grad-consistent). ``set_mxfp8_backend`` forces a choice for tests."""
    if not torch.cuda.is_available():
        return "dequant"
    if _FORCE_BACKEND == "mma":
        return "mxfp8_mma"
    if _FORCE_BACKEND == "triton":
        return "mxfp8_triton"
    return "mxfp8_mma" if _hw_mx_grouped_available() else "mxfp8_triton"


class _Mxfp8GroupedExperts(torch.autograd.Function):
    """x[Mt,H] -> gate_up(mxfp8 base + LoRA) -> gated-act -> down(...) -> [Mt,H].
    Frozen mxfp8 experts; trainable LoRA A/B. Mirrors _Fp8BlockGroupedExperts."""

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
    ):
        E = Agu.size(0)
        gu = _mx_base_forward(base, 0, x, m_indices, offs)
        gu, xAg = grouped_lora_fwd(x, Agu, Bgu, scaling, offs, E, residual=gu)
        h = _swiglu(gu, limit, act_type).to(x.dtype)
        dn = _mx_base_forward(base, 1, h, m_indices, offs)
        dn, hAd = grouped_lora_fwd(h, Adn, Bdn, scaling, offs, E, residual=dn)
        ctx.save_for_backward(x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices)
        ctx.weight_recipe, ctx.scaling, ctx.limit, ctx.act_type = (
            weight_recipe,
            scaling,
            limit,
            act_type,
        )
        return dn

    @staticmethod
    def backward(ctx, d_dn):
        x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices = ctx.saved_tensors
        s, lim, act_type = ctx.scaling, ctx.limit, ctx.act_type
        d_dn = d_dn.contiguous().to(x.dtype)
        gu_mx, dn_mx = ctx.weight_recipe()  # ((wq,ws), (wq,ws)) — FSDP-safe re-read
        E = Agu.size(0)
        tile = x.size(0) // m_indices.numel()  # routing pad granularity (TILE)
        dh = _mx_base_dx(dn_mx, d_dn, offs, m_indices, tile)
        dh, dAdn, dBdn = grouped_lora_bwd(
            d_dn, h, Adn, Bdn, hAd, s, offs, E, residual=dh
        )
        dgu = _swiglu_bwd(dh, gu, lim, act_type).to(x.dtype)
        del dh
        dx = _mx_base_dx(gu_mx, dgu, offs, m_indices, tile)
        dx, dAgu, dBgu = grouped_lora_bwd(
            dgu, x, Agu, Bgu, xAg, s, offs, E, residual=dx
        )
        return (dx, None, None, dAgu, dBgu, dAdn, dBdn, None, None, None, None, None)


def grouped_mxfp8_moe_train(
    hidden,
    idx,
    wts,
    gate_up_q,
    gate_up_scale,
    down_q,
    down_scale,
    gup_lora,
    down_lora,
    limit,
    act_type="silu",
    weight_recipe=None,
):
    """Training-capable native mxfp8 grouped MoE forward.

    hidden[N,H], idx/wts[N,topk]. Experts mxfp8: gate_up_q [E,2I,H] e4m3 + gate_up_scale [E,2I,H/32]
    e8m0; down_q [E,H,I] + down_scale. ``*_lora`` = (A,B,scaling) scattermoe layout. Returns [N,H];
    differentiable to hidden + LoRA A/B. ``weight_recipe`` -> ((gu_q,gu_scale),(dn_q,dn_scale)) for
    the FSDP-safe backward re-read (defaults to the forward tensors)."""
    global _BACKEND_LOGGED
    if weight_recipe is None:
        weight_recipe = lambda: (  # noqa: E731
            (gate_up_q, gate_up_scale),
            (down_q, down_scale),
        )
    N, H = hidden.shape
    E, twoI, _ = gate_up_q.shape
    I = down_q.shape[2]  # noqa: E741
    dev = hidden.device
    backend = _mx_backend()
    if not _BACKEND_LOGGED:
        _BACKEND_LOGGED = True
        LOG.info("grouped mxfp8 MoE base-GEMM backend: %s", backend)
    tile = TILE

    # --- routing: sort-by-expert, TILE-pad, contiguous grouping (mirrors grouped_fp8_moe_train) ---
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

    base = (backend, (gate_up_q, gate_up_scale), (down_q, down_scale))
    Agu, Bgu, sgu = _lora_stack(gup_lora, E, H, twoI)
    Adn, Bdn, sdn = _lora_stack(down_lora, E, I, H)
    assert sgu == sdn, "gate_up/down LoRA scaling must match"
    lim = float(limit) if limit is not None else 1e30

    A = hidden.new_zeros(Mt, H).index_copy(0, padded_row, hidden[rep])
    dn = _Mxfp8GroupedExperts.apply(
        A, base, weight_recipe, Agu, Bgu, Adn, Bdn, m_indices, offs, sgu, lim, act_type
    )
    out = hidden.new_zeros(N, H)
    return out.index_add(
        0, rep, (dn[padded_row] * wflat[:, None].to(dn.dtype)).to(out.dtype)
    )
