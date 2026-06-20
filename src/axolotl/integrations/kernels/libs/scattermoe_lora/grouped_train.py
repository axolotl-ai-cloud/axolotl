"""Training-capable grouped NVFP4 MoE (fwd+bwd) for DeepSeek-V4 experts — the config-gated
cutlass-fp4 path. Forward: cutlass fp4 grouped GEMM (fast). Backward: chunked bf16-dequant +
cuBLAS grouped_mm (accurate, bounded memory) — this beat the existing fused Triton kernel at the
real E=256 scale on BOTH speed (2.24x) and memory (0.87x). LoRA-on-experts fused as a grouped
correction. All cute imports lazy (module loads clean off sm120).

Backward design rationale (explored exhaustively): bf16 grad is REQUIRED (fp8 grad = 33% error);
dequant+cuBLAS beats the fused in-kernel-decode Triton dX 5.7x (Triton GEMM << cuBLAS); chunking
over expert-groups bounds the bf16-weight transient.
"""

from __future__ import annotations

import torch

TILE = 128
# backward base-dX dequant chunk: bounds the bf16-weight transient. Since the LoRA grads are now
# hoisted out of this loop (full-E grouped_mm), the bwd curve is flat past CHUNK_E~8; E=256 knee
# (B200, BK=1024 dequant): CHUNK_E=16 = 11.2ms bwd / 2.6GB peak vs 10.1ms floor (CHUNK_E=256 =
# 10GB). 16 = speed/memory sweet spot.
CHUNK_E = 16
_ENGINES: dict = {}


def grouped_fp4_available(mode: str) -> bool:
    """True iff the grouped fp4 training path can run here for `mode`: nvfp4 on sm120 (CUTLASS
    fused-decode) or sm90/sm100 (DeepGEMM fp8-act x mxfp4-weight). Backward is GPU-agnostic
    (chunked bf16-dequant). chunked-only fallback tracked separately."""
    if mode != "nvfp4":
        return False
    try:
        from .cutlass_fp4 import cutlass_fp4_available
        if cutlass_fp4_available():
            return True
    except Exception:
        pass
    try:
        from .dequant_grouped import deepgemm_grouped_available
        return deepgemm_grouped_available()
    except Exception:
        return False


def _train_backend(mode: str) -> str | None:
    """Base-GEMM backend for the training forward: 'cutlass' (sm120) | 'deepgemm' (sm90/100)."""
    if mode != "nvfp4":
        return None
    try:
        from .cutlass_fp4 import cutlass_fp4_available
        if cutlass_fp4_available():
            return "cutlass"
    except Exception:
        pass
    try:
        from .dequant_grouped import deepgemm_grouped_available
        if deepgemm_grouped_available():
            return "deepgemm"
    except Exception:
        pass
    return None


def _gmm(a, b, offs):
    return torch._grouped_mm(a, b, offs=offs)


def _gmm_w(a, b, offs):  # per-expert a_e^T @ b_e -> [E,A,B] (a.mT view, no copy)
    return torch._grouped_mm(a.mT, b, offs=offs)


def _swiglu(gu, limit):
    from .cutlass_fp4.swiglu import swiglu_fwd
    return swiglu_fwd(gu, limit)


def _swiglu_bwd(dh, gu, limit):
    from .cutlass_fp4.swiglu import swiglu_bwd
    return swiglu_bwd(gu, dh, limit)


def _base_forward(base, which, x, m_indices, mode):
    """Frozen-expert base GEMM for gate_up (which=0) or down (which=1). `base` is
    ('cutlass', gu_eng, dn_eng) or ('deepgemm', (guq,gus), (dnq,dns))."""
    backend, gw, dw = base
    if backend == "cutlass":
        from .cutlass_fp4.grouped import quant_act
        eng = gw if which == 0 else dw
        aq, as_ = quant_act(x, mode)
        return eng.forward(aq.unsqueeze(0), as_.unsqueeze(0), m_indices)
    from .dequant_grouped import deepgemm_grouped_fp8_fp4
    wq, ws = gw if which == 0 else dw
    # cutlass uses per-tile m_indices; DeepGEMM's grouped_layout is per-row (length Mt)
    return deepgemm_grouped_fp8_fp4(x, wq, ws, m_indices.repeat_interleave(TILE))


def _engine(Mt, N, K, E, mode):
    key = (Mt, N, K, E, mode)
    eng = _ENGINES.get(key)
    if eng is None:
        from .cutlass_fp4.grouped import GroupedFp4Gemm
        _ENGINES[key] = eng = GroupedFp4Gemm(Mt, N, K, E, mode)
    return eng


def _base_dx(w_nv, pt, g, out_k, offs):
    """Chunked base-weight contraction g @ W (frozen NVFP4 experts): dequant CHUNK_E experts to
    bf16 at a time (bounds the transient) and grouped_mm. g[Mt,N], W[E,N,out_k] -> [Mt,out_k]."""
    from .dequant_grouped import nvfp4_dequant_bf16

    E = w_nv.qdata.size(0)
    starts = torch.cat([offs.new_zeros(1), offs]).tolist()  # one sync, not per-iter
    out = g.new_empty(g.size(0), out_k)
    for c0 in range(0, E, CHUNK_E):
        c1 = min(c0 + CHUNK_E, E)
        t0, t1 = starts[c0], starts[c1]
        if t1 == t0:
            continue
        loc = (offs[c0:c1] - t0).to(torch.int32)
        Wc = nvfp4_dequant_bf16(w_nv.qdata[c0:c1], w_nv.scale[c0:c1], pt[c0:c1])
        out[t0:t1] = _gmm(g[t0:t1], Wc, loc)
    return out


def _pt(nv, E, dev):
    p = getattr(nv, "per_tensor_scale", None)
    if p is None:
        return torch.ones(E, device=dev)
    p = p.reshape(-1).float()
    return p.expand(E) if p.numel() == 1 else p


class _GroupedExperts(torch.autograd.Function):
    """x[Mt,H] -> gate_up(cutlass fp4 base + LoRA) -> clamped-swiglu -> down(...) -> [Mt,H].
    Frozen NVFP4 experts; trainable LoRA A/B (stacked [E,r,K]/[E,N,r])."""

    @staticmethod
    def forward(ctx, x, base, weight_recipe, Agu, Bgu, Adn, Bdn, m_indices, offs, scaling, limit, mode):
        # base: ('cutlass', gu_eng, dn_eng) | ('deepgemm', (guq,gus), (dnq,dns)) — frozen experts
        gu = _base_forward(base, 0, x, m_indices, mode)
        xAg = _gmm(x, Agu.transpose(1, 2), offs)
        gu = gu + scaling * _gmm(xAg, Bgu.transpose(1, 2), offs)
        h = _swiglu(gu, limit).to(x.dtype)
        dn = _base_forward(base, 1, h, m_indices, mode)
        hAd = _gmm(h, Adn.transpose(1, 2), offs)
        dn = dn + scaling * _gmm(hAd, Bdn.transpose(1, 2), offs)
        ctx.save_for_backward(x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd)
        # FSDP-safe: don't pin the gathered NVFP4 weight; re-read the (re-gathered) param in backward
        ctx.weight_recipe, ctx.scaling, ctx.limit = weight_recipe, scaling, limit
        return dn

    @staticmethod
    def backward(ctx, d_dn):
        x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd = ctx.saved_tensors
        s, lim = ctx.scaling, ctx.limit
        gu_nv, dn_nv = ctx.weight_recipe()
        E, dev = gu_nv.qdata.size(0), x.device
        ptg, ptd = _pt(gu_nv, E, dev), _pt(dn_nv, E, dev)
        d_dn = d_dn.contiguous().to(x.dtype)
        # Only the base dX contractions need the dequanted bf16 weight -> chunk those (bounds the
        # bf16-weight transient). All LoRA grads are dequant-free -> one full-E grouped_mm each
        # (avoids ~6 tiny grouped_mm per chunk; the small launches dominated the profile).
        dh = _base_dx(dn_nv, ptd, d_dn, h.size(1), offs)            # chunked: d_dn @ Wdn
        d_hAd = s * _gmm(d_dn, Bdn, offs)
        dh = dh + _gmm(d_hAd, Adn, offs)
        dBdn = s * _gmm_w(d_dn, hAd, offs)
        dAdn = _gmm_w(d_hAd, h, offs)
        dgu = _swiglu_bwd(dh, gu, lim).to(x.dtype); del dh
        dx = _base_dx(gu_nv, ptg, dgu, x.size(1), offs)             # chunked: dgu @ Wgu
        d_xAg = s * _gmm(dgu, Bgu, offs)
        dx = dx + _gmm(d_xAg, Agu, offs)
        dBgu = s * _gmm_w(dgu, xAg, offs)
        dAgu = _gmm_w(d_xAg, x, offs)
        # grads align to forward args: x, base, weight_recipe, Agu, Bgu, Adn, Bdn,
        # m_indices, offs, scaling, limit, mode
        return (dx, None, None, dAgu, dBgu, dAdn, dBdn, None, None, None, None, None)


def _lora_stack(lora, E, K, out):
    """scattermoe LoRA (A[r*E,K], B[out,r*E], scaling) -> stacked (Agu[E,r,K], Bgu[E,out,r], s)."""
    A, B, scaling = lora
    r = A.shape[0] // E
    As = A.reshape(E, r, K).contiguous()
    Bs = B.reshape(out, E, r).permute(1, 0, 2).contiguous()
    return As, Bs, float(scaling)


def grouped_fp4_moe_train(hidden, idx, wts, gate_up_nv, down_nv, gup_lora, down_lora, limit, mode,
                          weight_recipe=None, mxfp4_cache=None):
    """Training-capable grouped NVFP4 MoE forward. hidden[N,H], idx/wts[N,topk]; experts NVFP4Tensor;
    *_lora = (A,B,scaling) scattermoe layout. Returns [N,H]; differentiable to hidden + LoRA A/B.
    weight_recipe: optional callable -> (gate_up_nv, down_nv) re-read for the FSDP-safe backward
    (defaults to the forward tensors). mxfp4_cache: optional persistent dict (e.g. on the owning
    module) so the DeepGEMM backend requantizes the frozen weight once across FSDP re-gathers."""
    if weight_recipe is None:
        weight_recipe = lambda: (gate_up_nv, down_nv)  # noqa: E731
    N, H = hidden.shape
    E = gate_up_nv.qdata.size(0)
    twoI = gate_up_nv.qdata.size(1)          # gate_up N dim = 2I
    I = down_nv.qdata.size(2) * 2            # down K dim = I (packed /2)
    dev = hidden.device
    flat = idx.reshape(-1)
    order = flat.argsort()
    rep = torch.arange(N, device=dev).repeat_interleave(idx.size(1))[order]
    wflat = wts.reshape(-1)[order]
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=E)
    ptiles = (counts + TILE - 1) // TILE
    roff = torch.cat([ptiles.new_zeros(1), ptiles.cumsum(0)]) * TILE
    coff = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    padded_row = roff[exp_sorted] + (torch.arange(exp_sorted.numel(), device=dev) - coff[exp_sorted])
    m_indices = torch.repeat_interleave(torch.arange(E, dtype=torch.int32, device=dev), ptiles)
    offs = (ptiles * TILE).cumsum(0).to(torch.int32)
    Mt = int(ptiles.sum()) * TILE

    backend = _train_backend(mode)
    if backend == "deepgemm":
        from .dequant_grouped import _cached_mxfp4
        base = ("deepgemm",
                _cached_mxfp4(gate_up_nv, _pt(gate_up_nv, E, dev), mxfp4_cache, "gate_up"),
                _cached_mxfp4(down_nv, _pt(down_nv, E, dev), mxfp4_cache, "down"))
    else:
        gu_eng = _engine(Mt, twoI, H, E, mode); gu_eng.set_weights(gate_up_nv.qdata, gate_up_nv.scale)
        dn_eng = _engine(Mt, H, I, E, mode); dn_eng.set_weights(down_nv.qdata, down_nv.scale)
        base = ("cutlass", gu_eng, dn_eng)
    Agu, Bgu, sgu = _lora_stack(gup_lora, E, H, twoI)
    Adn, Bdn, sdn = _lora_stack(down_lora, E, I, H)
    assert sgu == sdn, "gate_up/down LoRA scaling must match"

    lim = float(limit) if limit is not None else 1e30  # no clamp when the model has no swiglu_limit
    A = hidden.new_zeros(Mt, H).index_copy(0, padded_row, hidden[rep])
    dn = _GroupedExperts.apply(A, base, weight_recipe, Agu, Bgu, Adn, Bdn,
                               m_indices, offs, sgu, lim, mode)
    out = hidden.new_zeros(N, H)
    return out.index_add(0, rep, (dn[padded_row] * wflat[:, None].to(dn.dtype)).to(out.dtype))
