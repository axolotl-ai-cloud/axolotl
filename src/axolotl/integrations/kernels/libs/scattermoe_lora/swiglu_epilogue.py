"""Fused LoRA-B GEMM + gated-activation epilogue (moe_swiglu_epilogue).

The activation must apply AFTER the LoRA residual (``h = act(base + lora)``), so it cannot fuse
into the frozen base GEMM — the last producer of the final ``gu`` is the LoRA-B GEMM. This kernel
computes a gate tile and its paired up tile in one program (one ``xa`` load feeds both dots),
folds the base residual, writes ``gu`` back in place (backward is unchanged — it still reads the
pre-activations) and emits ``h`` from registers: the separate swiglu pass's full ``[Mt, 2I]``
read + ``[Mt, I]`` write disappear.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def swiglu_epilogue_enabled() -> bool:
    import os

    from .runtime import RUNTIME

    env = os.environ.get("AXOLOTL_SWIGLU_EPILOGUE")
    if env is not None:
        return env != "0"
    return bool(RUNTIME.swiglu_epilogue)


@triton.jit
def _lora_b_swiglu_kernel(
    XA,  # [Mt, r] scaled inner activation (scaling already folded)
    WB,  # [E, r, 2I]
    GU,  # [Mt, 2I] in: base output; out: base + lora (the saved pre-activations)
    H,  # [Mt, I]
    SEI,  # [Mt] expert id per grouped row
    I,
    R,
    NB,  # cdiv(I, BN) column tiles
    LIMIT,
    sxa_m,
    swb_e,
    swb_r,
    swb_n,
    sgu_m,
    sh_m,
    GELU: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    m = pid // NB
    n = pid % NB
    rows = m * BM + tl.arange(0, BM)
    cols = n * BN + tl.arange(0, BN)
    cm = cols < I
    e = tl.load(SEI + m * BM).to(tl.int64)

    rr = tl.arange(0, BR)
    rm = rr < R
    xa = tl.load(
        XA + rows[:, None] * sxa_m + rr[None, :], mask=rm[None, :], other=0.0
    )  # [BM, BR]
    bg = tl.load(
        WB + e * swb_e + rr[:, None] * swb_r + cols[None, :] * swb_n,
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )
    bu = tl.load(
        WB + e * swb_e + rr[:, None] * swb_r + (I + cols)[None, :] * swb_n,
        mask=rm[:, None] & cm[None, :],
        other=0.0,
    )
    g = tl.dot(xa, bg) + tl.load(
        GU + rows[:, None] * sgu_m + cols[None, :], mask=cm[None, :], other=0.0
    ).to(tl.float32)
    u = tl.dot(xa, bu) + tl.load(
        GU + rows[:, None] * sgu_m + (I + cols)[None, :], mask=cm[None, :], other=0.0
    ).to(tl.float32)
    tl.store(
        GU + rows[:, None] * sgu_m + cols[None, :],
        g.to(GU.dtype.element_ty),
        mask=cm[None, :],
    )
    tl.store(
        GU + rows[:, None] * sgu_m + (I + cols)[None, :],
        u.to(GU.dtype.element_ty),
        mask=cm[None, :],
    )
    # activate the ROUNDED values so h matches the unfused path (which reads the stored bf16 gu)
    g = g.to(GU.dtype.element_ty).to(tl.float32)
    u = u.to(GU.dtype.element_ty).to(tl.float32)
    if GELU:
        k = 0.7978845608028654  # sqrt(2/pi)
        t = libdevice.tanh(k * (g + 0.044715 * g * g * g))
        h = (0.5 * g * (1.0 + t)) * u
    else:
        gc = tl.minimum(g, LIMIT)
        uc = tl.minimum(tl.maximum(u, -LIMIT), LIMIT)
        h = (gc * tl.sigmoid(gc)) * uc
    tl.store(
        H + rows[:, None] * sh_m + cols[None, :],
        h.to(H.dtype.element_ty),
        mask=cm[None, :],
    )


def grouped_lora_swiglu_fwd(x, A, B, scaling, offs, E, gu_base, act_type, limit):
    """gate_up LoRA forward with the activation fused into the LoRA-B epilogue.

    Same contract as ``grouped_lora_fwd(residual=gu_base)`` followed by ``_swiglu`` — returns
    ``(gu, h, xa)`` with ``gu`` (base + scaled lora) updated IN PLACE on ``gu_base`` and ``xa``
    unscaled for the backward (which is unchanged)."""
    from .grouped_lora import _mk_routing
    from .kernels.ops import scatter2scatter

    sei, ssi = _mk_routing(offs, E)
    W_A = A.permute(0, 2, 1).contiguous()
    xa = scatter2scatter(
        X=x,
        W=W_A,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=1,
        x_grouped=True,
        y_grouped=True,
    )
    W_B = B.permute(0, 2, 1).contiguous()  # [E, r, 2I]
    Mt, r = xa.shape
    I = W_B.size(2) // 2  # noqa: E741
    h = torch.empty(Mt, I, device=x.device, dtype=gu_base.dtype)
    BM, BN = 64, 64
    BR = max(16, triton.next_power_of_2(r))
    nb = triton.cdiv(I, BN)
    xa_s = (xa * scaling).to(gu_base.dtype)
    _lora_b_swiglu_kernel[(Mt // BM * nb,)](
        xa_s,
        W_B,
        gu_base,
        h,
        sei,
        I,
        r,
        nb,
        float(limit),
        xa_s.stride(0),
        W_B.stride(0),
        W_B.stride(1),
        W_B.stride(2),
        gu_base.stride(0),
        h.stride(0),
        GELU=act_type == "gelu_tanh",
        BM=BM,
        BN=BN,
        BR=BR,
    )
    return gu_base, h, xa
