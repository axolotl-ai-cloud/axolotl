"""Unified grouped NVFP4 MoE forward for DeepSeek-V4 experts (config-gated, dsv4_fp4_grouped_mode).

Contiguous-grouped: tokens sorted by expert, padded per-expert to TILE(128); ONE grouped gate_up
GEMM -> clamped-SwiGLU -> ONE grouped down GEMM, with GPU-vectorized routing/pack/scatter (no
per-expert Python loop). The base GEMM auto-dispatches to the best fp4 path:
    DeepGEMM (sm90/sm100)  ->  CUTLASS grouped (sm120)  ->  chunked-dequant (any GPU, fallback).

OFF unless cfg.dsv4_fp4_grouped_mode is set; existing fused-Triton/eager paths untouched.
Bench (RTX PRO 6000, H=4096 I=2048 top6, FORWARD): vs chunked-dequant E=32 1.6-1.9x, E=256
3.5-3.7x (single-launch grouped vs chunked's per-chunk loop).

STATUS: forward path (base + clamped-swiglu + routing) implemented + validated. TRAINING is the
remaining piece — the cutlass/deepgemm engines are forward-only, so dX (hidden grad) and
LoRA-on-experts backward need autograd.Function wrappers (tasks #14, #18, #19). Until then this
path is correct for inference/eval; training still uses the fused-Triton path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TILE = 128
_ENGINE_CACHE: dict = {}


def grouped_fp4_backend(mode: str) -> str | None:
    """Best available base-GEMM backend: 'cutlass' (sm120) | 'deepgemm' (sm90/100) | 'chunked'."""
    try:
        from .cutlass_fp4 import cutlass_fp4_available

        if mode == "nvfp4" and cutlass_fp4_available():
            return "cutlass"
    except Exception:
        pass
    try:
        from .dequant_grouped import deepgemm_grouped_available

        if deepgemm_grouped_available():
            return "deepgemm"
    except Exception:
        pass
    return "chunked" if torch.cuda.is_available() else None


def _route(idx, E, dev):
    flat = idx.reshape(-1)
    order = flat.argsort()
    rep = torch.arange(idx.size(0), device=dev).repeat_interleave(idx.size(1))[order]
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    ptiles = (counts + TILE - 1) // TILE
    roff = torch.cat([ptiles.new_zeros(1), ptiles.cumsum(0)]) * TILE
    coff = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    local = torch.arange(exp_sorted.numel(), device=dev) - coff[exp_sorted]
    padded_row = roff[exp_sorted] + local
    m_indices = torch.repeat_interleave(torch.arange(E, dtype=torch.int32, device=dev), ptiles)
    Mt = int(ptiles.sum()) * TILE
    return rep, padded_row, m_indices, counts, Mt


def _engine(Mt, N, K, E, mode):
    key = (Mt, N, K, E, mode)
    eng = _ENGINE_CACHE.get(key)
    if eng is None:
        from .cutlass_fp4.grouped import GroupedFp4Gemm

        _ENGINE_CACHE[key] = eng = GroupedFp4Gemm(Mt, N, K, E, mode)
    return eng


def _quant_weight(W_nv, mode):
    """Per-expert weight quant for the grouped engines. W_nv: NVFP4Tensor [E,N,K]."""
    E = W_nv.qdata.size(0)
    if mode == "nvfp4":
        from .cutlass_fp4.quant_nvfp4 import nvfp4_quant

        deq = W_nv.dequantize(torch.bfloat16)
        qs = [nvfp4_quant(deq[e].contiguous()) for e in range(E)]
        return torch.stack([q for q, _ in qs]), torch.stack([s for _, s in qs])
    from torchao.prototype.mx_formats.mx_tensor import MXTensor

    deq = W_nv.dequantize(torch.bfloat16)
    ts = [MXTensor.to_mx(deq[e].contiguous(), torch.float4_e2m1fn_x2, 32) for e in range(E)]
    return torch.stack([t.qdata for t in ts]), torch.stack([t.scale for t in ts])


@torch.no_grad()
def grouped_fp4_moe_forward(hidden, idx, wts, gate_up_nv, down_nv, limit, mode, backend=None):
    """Forward-only grouped NVFP4 MoE. hidden[N,H], idx/wts[N,topk], experts NVFP4Tensor.

    Returns [N,H]. backend auto-selected if None. TRAINING NOT YET (forward-only engines).
    """
    N, H = hidden.shape
    E = gate_up_nv.qdata.size(0)
    Idim = down_nv.qdata.size(2) * 2  # down K = I (packed K/2)
    dev = hidden.device
    backend = backend or grouped_fp4_backend(mode)
    rep, padded_row, m_indices, counts, Mt = _route(idx, E, dev)
    wflat = wts.reshape(-1)[idx.reshape(-1).argsort()]

    A = hidden.new_zeros(Mt, H)
    A[padded_row] = hidden[rep]

    if backend == "cutlass":
        from .cutlass_fp4.grouped import quant_act

        gu_eng = _engine(Mt, 2 * (down_nv.qdata.size(2) * 2), H, E, mode)  # N = 2I
        gu_eng.set_weights(*_quant_weight(gate_up_nv, mode))
        aq, as_ = quant_act(A, mode)
        gu = gu_eng.forward(aq.unsqueeze(0), as_.unsqueeze(0), m_indices).float()
    elif backend == "deepgemm":  # native fp8(act) x mxfp4(weight) grouped GEMM (SM90/SM100)
        from .dequant_grouped import _cached_mxfp4, deepgemm_grouped_fp8_fp4

        wq, ws = _cached_mxfp4(gate_up_nv, _per_tensor(gate_up_nv, E))
        gu = deepgemm_grouped_fp8_fp4(A, wq, ws, m_indices.repeat_interleave(TILE)).float()
    else:  # chunked fallback: dequant weight -> grouped bf16 matmul
        from .dequant_grouped import nvfp4_dequant_bf16

        pt = _per_tensor(gate_up_nv, E)
        Wb = nvfp4_dequant_bf16(gate_up_nv.qdata, gate_up_nv.scale, pt)  # [E,2I,H]
        offs = (torch.bincount(m_indices, minlength=E) * TILE).cumsum(0).to(torch.int32)
        gu = torch._grouped_mm(A, Wb.transpose(1, 2), offs=offs).float()

    g, u = gu.chunk(2, dim=-1)
    h = (F.silu(g.clamp(max=limit)) * u.clamp(min=-limit, max=limit)).to(hidden.dtype)

    if backend == "cutlass":
        from .cutlass_fp4.grouped import quant_act

        dn_eng = _engine(Mt, H, down_nv.qdata.size(2) * 2, E, mode)  # K = I
        dn_eng.set_weights(*_quant_weight(down_nv, mode))
        hq, hs = quant_act(h.contiguous(), mode)
        dn = dn_eng.forward(hq.unsqueeze(0), hs.unsqueeze(0), m_indices)
    elif backend == "deepgemm":
        from .dequant_grouped import _cached_mxfp4, deepgemm_grouped_fp8_fp4

        wq, ws = _cached_mxfp4(down_nv, _per_tensor(down_nv, E))
        dn = deepgemm_grouped_fp8_fp4(h.contiguous(), wq, ws, m_indices.repeat_interleave(TILE))
    else:
        from .dequant_grouped import nvfp4_dequant_bf16

        pt = _per_tensor(down_nv, E)
        Wb = nvfp4_dequant_bf16(down_nv.qdata, down_nv.scale, pt)  # [E,H,I]
        offs = (torch.bincount(m_indices, minlength=E) * TILE).cumsum(0).to(torch.int32)
        dn = torch._grouped_mm(h.contiguous(), Wb.transpose(1, 2), offs=offs)

    out = hidden.new_zeros(N, H)
    return out.index_add(0, rep, (dn[padded_row] * wflat[:, None].to(dn.dtype)).to(out.dtype))


def _per_tensor(w_nv, E):
    pt = getattr(w_nv, "per_tensor_scale", None)
    return torch.ones(E, device="cuda") if pt is None else pt.reshape(-1).float()
