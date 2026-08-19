"""Training-capable grouped NVFP4 MoE (fwd+bwd) for DeepSeek-V4 experts — the config-gated
cutlass-fp4 path. Forward: cutlass fp4 grouped GEMM (fast). Backward: chunked bf16-dequant +
cuBLAS grouped_mm (accurate, bounded memory) — this beat the existing fused Triton kernel at the
real E=256 scale on BOTH speed (2.24x) and memory (0.87x). LoRA-on-experts fused via
scatter2scatter single-launch grouped GEMMs. All cute imports lazy (module loads clean off sm120).

Backward design rationale (explored exhaustively): bf16 grad is REQUIRED (fp8 grad = 33% error);
dequant+cuBLAS beats the fused in-kernel-decode Triton dX 5.7x (Triton GEMM << cuBLAS); chunking
over expert-groups bounds the bf16-weight transient.

LoRA GEMM design: scatter2scatter kernels (kernels/ops.py) operate directly on the TILE-padded
expert-sorted layout with x_grouped=y_grouped=True — single launch, ragged-native, no padding
transient.  grouped_lora_fwd/bwd in grouped_lora.py encapsulate the entire implementation;
no selection or threshold logic lives here.
"""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

from .grouped_lora import grouped_lora_bwd, grouped_lora_fwd

LOG = get_logger(__name__)

_BACKEND_LOGGED = False  # one-time log of the resolved base-GEMM backend


# thin compat wrapper over the centralized runtime module. grouped_backend (cfg.moe_grouped_backend):
# None/"auto" = capability auto-select; marlin|cutlass|deepgemm = force if available (else warn +
# auto); "dequant" = force the chunked-dequant fallback.
from .runtime import RUNTIME  # noqa: E402


def set_grouped_backend_override(backend) -> None:
    RUNTIME.grouped_backend = str(backend).lower() if backend else None


def _backend_available(name: str) -> bool:
    try:
        if name == "marlin":
            from .marlin_w4a16 import marlin_w4a16_available

            return marlin_w4a16_available()
        if name == "cutlass":
            from .cutlass_fp4 import cutlass_fp4_available

            return cutlass_fp4_available()
        if name == "deepgemm":
            from .dequant_grouped import deepgemm_grouped_available

            return deepgemm_grouped_available()
    except Exception:
        return False
    return False


def _auto_backend() -> str | None:
    """Capability + arch auto-select. Order is arch-aware so each GPU class gets its tuned default
    while Marlin (the only fused W4A16 path that runs on Ampere/Ada) backs up everything:
      - sm120 (consumer Blackwell): Marlin (~1.79x CUTLASS, bit-correct) > CUTLASS > DeepGEMM
      - sm100 (datacenter Blackwell, e.g. B200): DeepGEMM (tuned fp8-act x mxfp4) > Marlin
      - sm90 (Hopper) / sm80 / sm89 (Ampere / Ada): Marlin only (the DeepGEMM fp8xfp4 grouped
        kernel is sm100-only; deepgemm_grouped_available() returns False below sm100 -> Marlin)
    Marlin stays force-selectable everywhere via cfg.moe_grouped_backend."""
    import torch

    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    order: tuple[str, ...]
    if major >= 11:
        order = ("marlin", "cutlass", "deepgemm")
    elif major in (9, 10):
        order = ("deepgemm", "marlin")
    else:
        order = ("marlin",)
    for name in order:
        if _backend_available(name):
            return name
    return None


TILE = 128
# backward base-dX dequant chunk; bounds the bf16-weight transient.
# E=256 knee (B200, BK=1024 dequant): CHUNK_E=16 = 11.2ms bwd / 2.6GB peak vs 10.1ms floor.
CHUNK_E = 16
_ENGINES: dict = {}


def grouped_fp4_available(mode: str) -> bool:
    """True iff the grouped fp4 training path can run here for `mode`: nvfp4 on sm120 (CUTLASS
    fused-decode) or sm90/sm100 (DeepGEMM fp8-act x mxfp4-weight). Backward is GPU-agnostic
    (chunked bf16-dequant). chunked-only fallback tracked separately."""
    if mode != "nvfp4":
        return False
    try:
        from .marlin_w4a16 import marlin_w4a16_available

        if marlin_w4a16_available():
            return True
    except Exception:
        pass
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
    """Base-GEMM backend for the training forward: 'marlin' (sm120, W4A16 bf16-act — preferred) |
    'cutlass' (sm120, W4A4 fp4-act) | 'deepgemm' (sm90/100), or None for the chunked-dequant
    fallback. Auto-selects by capability unless cfg.moe_grouped_backend forced one (see
    set_grouped_backend_override): an unavailable forced backend warns and falls back to auto;
    'dequant' forces the fallback (None)."""
    if mode != "nvfp4":
        return None
    override = RUNTIME.grouped_backend
    if override and override != "auto":
        if override == "dequant":
            return None  # chunked-dequant fallback
        if _backend_available(override):
            return override
        LOG.warning(
            "moe_grouped_backend=%r is not available on this GPU; falling back to auto-select.",
            override,
        )
    return _auto_backend()


def _gmm(a, b, offs):
    return torch._grouped_mm(a, b, offs=offs)


def _swiglu(gu, limit, act_type="silu"):
    from .cutlass_fp4.swiglu import swiglu_fwd

    return swiglu_fwd(gu, limit, act_type)


def _swiglu_bwd(dh, gu, limit, act_type="silu"):
    from .cutlass_fp4.swiglu import swiglu_bwd

    return swiglu_bwd(gu, dh, limit, act_type)


def _base_forward(base, which, x, m_indices, mode):
    """Frozen-expert base GEMM for gate_up (which=0) or down (which=1). `base` is
    ('marlin', (gu_w, dn_w), dims, ws) | ('cutlass', gu_eng, dn_eng) | ('deepgemm', (guq,gus), (dnq,dns))."""
    if base[0] == "marlin":
        from .marlin_w4a16.backend import marlin_base_forward

        return marlin_base_forward(base, which, x, m_indices)
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


def _fp8_read_dx_ok():
    """fp8-read backward dX (#3744) wins on sm120 (bandwidth-bound: half weight bytes -> ~1.5x +
    half memory). On sm100 it's speed-neutral (cuBLAS bf16 is fast) so keep the bf16 path there."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 12


def _base_dx(
    w_nv, pt, g, out_k, offs, m_indices, prefer_fp8=True, tile=TILE, _marlin_raw=None
):
    """Chunked base-weight contraction g @ W (frozen NVFP4 experts). bf16 path: dequant CHUNK_E
    experts to bf16 + cuBLAS grouped_mm. fp8-read path (sm120): dequant to fp8 (half bytes) + a
    Triton grouped GEMM that reads fp8 and upcasts in-register (~1.5x faster, half the transient).
    g[Mt,N], W[E,N,out_k] -> [Mt,out_k]; m_indices is the per-`tile` expert id (for the fp8 GEMM).
    `tile` is the routing pad granularity (128 cutlass, 64 marlin) and is passed to the fp8 dX
    kernel as its row-block (BM) so one expert maps to each block. Both fp8-read and bf16-dequant
    are gradient-consistent (they dequant the FORWARD weight); fp8-read is the faster default.
    _marlin_raw: optional (qw_flat_E, orig_scale, pt_E) for the marlin memory-free path — reads from
    the marlin qweight cache via fused Triton dequant instead of the freed nv.qdata."""
    from .dequant_grouped import nvfp4_dequant_bf16

    fp8 = _fp8_read_dx_ok() and prefer_fp8
    if fp8:
        from .dequant_grouped import grouped_dx_fp8, nvfp4_dequant_fp8

    if _marlin_raw is not None:
        # Marlin memory-free path: dequant from marlin int32 layout via fused Triton kernel.
        # _marlin_raw = (qw [E, words_per_expert] int32, orig_scale [E, N, K//16] fp8, pt [E] f32)
        from .marlin_w4a16.backend import _build_base_scatter
        from .marlin_w4a16.fused_dequant import marlin_dequant_bf16, marlin_dequant_fp8
        from .mx_weights import fp4_codebook

        qw, orig_scale, pt_e = _marlin_raw
        # pt_e may have stride(0) from .expand(); Triton needs element-stride=1 for a correct load.
        if pt_e.stride(0) == 0:
            pt_e = pt_e.contiguous()
        E, N, Kg = orig_scale.shape
        K = Kg * 16
        scatter_lut = _build_base_scatter(qw.device)
        cb = fp4_codebook(qw.device).float()
        starts = torch.cat([offs.new_zeros(1), offs]).tolist()
        out = g.new_empty(g.size(0), out_k)
        for c0 in range(0, E, CHUNK_E):
            c1 = min(c0 + CHUNK_E, E)
            t0, t1 = starts[c0], starts[c1]
            if t1 == t0:
                continue
            C = c1 - c0
            if fp8:
                Wc = marlin_dequant_fp8(
                    qw[c0:c1].reshape(C, -1),
                    orig_scale[c0:c1],
                    pt_e[c0:c1],
                    scatter_lut,
                    cb,
                    N,
                    K,
                    C,
                )
                mi = (m_indices[t0 // tile : t1 // tile] - c0).to(torch.int32)
                out[t0:t1] = grouped_dx_fp8(g[t0:t1], Wc, mi, tile)
            else:
                Wc = marlin_dequant_bf16(
                    qw[c0:c1].reshape(C, -1),
                    orig_scale[c0:c1],
                    pt_e[c0:c1],
                    scatter_lut,
                    cb,
                    N,
                    K,
                    C,
                )
                loc = (offs[c0:c1] - t0).to(torch.int32)
                out[t0:t1] = _gmm(g[t0:t1], Wc, loc)
        return out

    qdata, scale = w_nv.qdata, w_nv.scale
    E = qdata.size(0)
    starts = torch.cat([offs.new_zeros(1), offs]).tolist()  # one sync, not per-iter
    out = g.new_empty(g.size(0), out_k)
    for c0 in range(0, E, CHUNK_E):
        c1 = min(c0 + CHUNK_E, E)
        t0, t1 = starts[c0], starts[c1]
        if t1 == t0:
            continue
        if fp8:
            Wc = nvfp4_dequant_fp8(qdata[c0:c1], scale[c0:c1], pt[c0:c1])
            mi = (m_indices[t0 // tile : t1 // tile] - c0).to(torch.int32)
            out[t0:t1] = grouped_dx_fp8(g[t0:t1], Wc, mi, tile)
        else:
            loc = (offs[c0:c1] - t0).to(torch.int32)
            Wc = nvfp4_dequant_bf16(qdata[c0:c1], scale[c0:c1], pt[c0:c1])
            out[t0:t1] = _gmm(g[t0:t1], Wc, loc)
    return out


def _pt(nv, E, dev):
    p = getattr(nv, "per_tensor_scale", None)
    if p is None:
        return torch.ones(E, device=dev)
    p = p.reshape(-1).float()
    return p.expand(E) if p.numel() == 1 else p


def _has_nonunit_pt(*nvs) -> bool:
    """True iff any NVFP4 weight carries a non-unit per_tensor_scale (weight_scale_2). The cutlass
    forward (set_weights) folds only the E4M3 block scale, dropping weight_scale_2, while the
    backward applies it, giving a wrong forward + grad mismatch when it != 1 (the real DSV4 ckpt).
    Folding weight_scale_2 into the cutlass weight + saved swiglu input is unimplemented, so the
    backend selection skips cutlass in that case."""
    for nv in nvs:
        p = getattr(nv, "per_tensor_scale", None)
        if p is None:
            continue
        if not torch.allclose(
            p.reshape(-1).float(), torch.ones((), device=p.device, dtype=torch.float32)
        ):
            return True
    return False


class _GroupedExperts(torch.autograd.Function):
    """x[Mt,H] -> gate_up(cutlass fp4 base + LoRA) -> gated-activation -> down(...) -> [Mt,H].
    Frozen NVFP4 experts; trainable LoRA A/B (stacked [E,r,K]/[E,N,r]).
    act_type: 'silu' (DSV4 clamped SwiGLU) or 'gelu_tanh' (Gemma4 GeGLU)."""

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
        mode,
        act_type,
        prefer_fp8_dx=True,
    ):
        E = Agu.size(0)
        gu = _base_forward(base, 0, x, m_indices, mode)
        # LoRA-B GEMM folds the base output in via residual=gu -> gu = base + lora, no separate add.
        gu, xAg = grouped_lora_fwd(x, Agu, Bgu, scaling, offs, E, residual=gu)
        h = _swiglu(gu, limit, act_type).to(x.dtype)
        dn = _base_forward(base, 1, h, m_indices, mode)
        dn, hAd = grouped_lora_fwd(h, Adn, Bdn, scaling, offs, E, residual=dn)
        ctx.save_for_backward(x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices)
        # FSDP-safe: don't pin the gathered NVFP4 weight; re-read the (re-gathered) param in backward
        ctx.weight_recipe, ctx.scaling, ctx.limit = weight_recipe, scaling, limit
        ctx.act_type = act_type
        # base dX: fp8-read (fast, ~2% grad) vs bf16-dequant (~0.5%)
        ctx.prefer_fp8_dx = prefer_fp8_dx
        # Marlin memory-free path: build_marlin_forward_base freed nv.qdata and saved (qdata, scale,
        # pt) in the cache (single-GPU only); stash them so backward skips weight_recipe() (which
        # would return the emptied NVFP4 tensor).
        ctx.bwd_marlin = None
        if base[0] == "marlin":
            from .marlin_w4a16.backend import marlin_bwd_data

            ctx.bwd_marlin = marlin_bwd_data(base)
        return dn

    @staticmethod
    def backward(ctx, d_dn):
        x, Agu, Bgu, Adn, Bdn, offs, gu, h, xAg, hAd, m_indices = ctx.saved_tensors
        s, lim, act_type = ctx.scaling, ctx.limit, ctx.act_type
        pf8 = _fp8_read_dx_ok() and ctx.prefer_fp8_dx
        tile = (
            x.size(0) // m_indices.numel()
        )  # routing pad granularity (128 cutlass, 64 marlin)

        d_dn = d_dn.contiguous().to(x.dtype)

        if ctx.bwd_marlin is not None:
            # Marlin memory-free path: nv.qdata was freed after repack; backward reads from the
            # marlin qweight cache via fused Triton dequant (marlin int32 + original scales).
            # bwd_marlin = ((gu_qw, (gu_scale, gu_pt, E, N, K)), (dn_qw, (dn_scale, ...)))
            (gu_qw, gu_bwd), (dn_qw, dn_bwd) = ctx.bwd_marlin
            dn_scale, dn_pt_e, _, _, _ = dn_bwd
            dn_marlin_raw = (dn_qw, dn_scale, dn_pt_e)
            dh = _base_dx(
                None,
                None,
                d_dn,
                h.size(1),
                offs,
                m_indices,
                pf8,
                tile,
                _marlin_raw=dn_marlin_raw,
            )
        else:
            gu_nv, dn_nv = ctx.weight_recipe()
            E, dev = gu_nv.qdata.size(0), x.device
            ptg, ptd = _pt(gu_nv, E, dev), _pt(dn_nv, E, dev)
            dh = _base_dx(dn_nv, ptd, d_dn, h.size(1), offs, m_indices, pf8, tile)

        E = Agu.size(0)
        # dX_lora GEMM folds base dh in via residual=dh -> dh = base_dX + lora_dX.
        dh, dAdn, dBdn = grouped_lora_bwd(
            d_dn, h, Adn, Bdn, hAd, s, offs, E, residual=dh
        )
        dgu = _swiglu_bwd(dh, gu, lim, act_type).to(x.dtype)
        del dh

        if ctx.bwd_marlin is not None:
            gu_scale, gu_pt_e, _, _, _ = ctx.bwd_marlin[0][1]
            gu_marlin_raw = (gu_qw, gu_scale, gu_pt_e)
            dx = _base_dx(
                None,
                None,
                dgu,
                x.size(1),
                offs,
                m_indices,
                pf8,
                tile,
                _marlin_raw=gu_marlin_raw,
            )
        else:
            dx = _base_dx(gu_nv, ptg, dgu, x.size(1), offs, m_indices, pf8, tile)

        dx, dAgu, dBgu = grouped_lora_bwd(
            dgu, x, Agu, Bgu, xAg, s, offs, E, residual=dx
        )
        # grads align to forward args: x, base, weight_recipe, Agu, Bgu, Adn, Bdn,
        # m_indices, offs, scaling, limit, mode, act_type, prefer_fp8_dx
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
            None,
        )


def _lora_stack(lora, E, K, out):
    """scattermoe LoRA (A[r*E,K], B[out,r*E], scaling) -> stacked (Agu[E,r,K], Bgu[E,out,r], s)."""
    A, B, scaling = lora
    r = A.shape[0] // E
    As = A.reshape(E, r, K).contiguous()
    Bs = B.reshape(out, E, r).permute(1, 0, 2).contiguous()
    return As, Bs, float(scaling)


def grouped_fp4_moe_train(
    hidden,
    idx,
    wts,
    gate_up_nv,
    down_nv,
    gup_lora,
    down_lora,
    limit,
    mode,
    act_type="silu",
    weight_recipe=None,
    mxfp4_cache=None,
    prefer_fp8_dx=True,
):
    """Training-capable grouped NVFP4 MoE forward. hidden[N,H], idx/wts[N,topk]; experts NVFP4Tensor;
    *_lora = (A,B,scaling) scattermoe layout. Returns [N,H]; differentiable to hidden + LoRA A/B.
    act_type: 'silu' (DSV4 clamped SwiGLU, default) or 'gelu_tanh' (Gemma4 GeGLU, no clamp).
    weight_recipe: optional callable -> (gate_up_nv, down_nv) re-read for the FSDP-safe backward
    (defaults to the forward tensors). mxfp4_cache: optional persistent dict (e.g. on the owning
    module) so the DeepGEMM backend requantizes the frozen weight once across FSDP re-gathers.
    prefer_fp8_dx: base dX backward — True = fp8-read (fast, ~2% grad error, sm120 default);
    False = bf16-dequant (slower, ~0.5%, max gradient fidelity)."""
    if weight_recipe is None:
        weight_recipe = lambda: (gate_up_nv, down_nv)  # noqa: E731
    N, H = hidden.shape
    # NVFP4Tensor.shape reports full-precision dims and survives the marlin qdata-free path (qdata
    # replaced with empty tensor). SimpleNamespace mocks (unit tests) lack .shape, so fall through
    # to the qdata.size() branch.
    _qd = gate_up_nv.qdata
    if _qd.numel() > 0:
        E = _qd.size(0)
        twoI = _qd.size(1)
        I = down_nv.qdata.size(2) * 2  # noqa: E741  (down K dim, packed K/2 * 2)
    else:
        # Marlin qdata-free path: qdata was freed; use the NVFP4Tensor wrapper shape.
        _gu_shape = gate_up_nv.shape
        E, twoI = int(_gu_shape[0]), int(_gu_shape[1])
        I = int(down_nv.shape[2])  # noqa: E741
    dev = hidden.device
    backend = _train_backend(mode)
    # cutlass weight_scale_2 (per_tensor_scale) folding is unimplemented: set_weights drops it on the
    # forward while the backward applies it, giving a wrong forward + grad mismatch when it != 1.
    # Marlin and DeepGEMM both fold _pt() into the weight, so prefer one of those (else hard-error)
    # for non-unit scales.
    if backend == "cutlass" and _has_nonunit_pt(gate_up_nv, down_nv):
        for _alt in ("marlin", "deepgemm"):
            if _backend_available(_alt):
                backend = _alt
                break
        else:
            major = (
                torch.cuda.get_device_capability()[0]
                if torch.cuda.is_available()
                else 0
            )
            raise RuntimeError(
                "grouped NVFP4 MoE: cutlass was selected but the weight has a non-unit "
                "per_tensor_scale (weight_scale_2) that the cutlass forward cannot fold "
                f"(unimplemented). No weight_scale_2-correct backend resolved on sm{major}x "
                "(marlin/deepgemm unavailable). Run on a GPU with marlin or deepgemm, or "
                "requantize the experts with a unit per_tensor_scale."
            )
    if not _BACKEND_LOGGED:
        globals()["_BACKEND_LOGGED"] = True
        LOG.info(
            "grouped fp4 MoE base-GEMM backend: %s",
            backend or "cutlass",
        )
    # Marlin (sm120 W4A16) pads to 64, half CUTLASS's 128 at thin-M (the padding is the cost since
    # each expert weight is read once either way), and its bf16-act kernel is bit-correct + faster.
    if backend == "marlin":
        from .marlin_w4a16.backend import MARLIN_TILE

        tile = MARLIN_TILE
    else:
        tile = TILE
    flat = idx.reshape(-1)
    # The DeepEP local path tags remote-routed slots with -1; drop them (bincount/grouping require
    # expert ids >= 0, and a dropped slot must contribute nothing to its token — correct, since that
    # (token, slot) is handled on the rank that owns the expert). No-op for the non-EP path (no -1).
    _tok = torch.arange(N, device=dev).repeat_interleave(idx.size(1))
    _wf = wts.reshape(-1)
    if (flat < 0).any():
        keep = flat >= 0
        flat = flat[keep]
        _tok = _tok[keep]
        _wf = _wf[keep]
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

    if backend == "marlin":
        from .marlin_w4a16.backend import build_marlin_forward_base

        base = build_marlin_forward_base(gate_up_nv, down_nv, mxfp4_cache)
    elif backend == "deepgemm":
        from .dequant_grouped import _cached_mxfp4

        base = (
            "deepgemm",
            _cached_mxfp4(gate_up_nv, _pt(gate_up_nv, E, dev), mxfp4_cache, "gate_up"),
            _cached_mxfp4(down_nv, _pt(down_nv, E, dev), mxfp4_cache, "down"),
        )
    else:
        gu_eng = _engine(Mt, twoI, H, E, mode)
        gu_eng.set_weights(gate_up_nv.qdata, gate_up_nv.scale)
        dn_eng = _engine(Mt, H, I, E, mode)
        dn_eng.set_weights(down_nv.qdata, down_nv.scale)
        base = ("cutlass", gu_eng, dn_eng)
    Agu, Bgu, sgu = _lora_stack(gup_lora, E, H, twoI)
    Adn, Bdn, sdn = _lora_stack(down_lora, E, I, H)
    assert sgu == sdn, "gate_up/down LoRA scaling must match"

    lim = (
        float(limit) if limit is not None else 1e30
    )  # no clamp when the model has no swiglu_limit
    A = hidden.new_zeros(Mt, H).index_copy(0, padded_row, hidden[rep])
    dn = _GroupedExperts.apply(
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
        mode,
        act_type,
        prefer_fp8_dx,
    )
    out = hidden.new_zeros(N, H)
    return out.index_add(
        0, rep, (dn[padded_row] * wflat[:, None].to(dn.dtype)).to(out.dtype)
    )
