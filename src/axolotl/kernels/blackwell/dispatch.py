# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Dispatch helpers: route LoRA ops to the fused sm_120 kernels when applicable,
else return None so the caller falls back to the existing path.

The fused kernels handle the common LoRA SFT case (bf16, frozen unquantized base,
no DoRA/bias/dropout, shared scaling). Anything outside that falls back.
"""

import logging

import torch

LOG = logging.getLogger(__name__)

_IS_SM120: bool | None = None
_logged: set = set()


def _log_once(kind: str) -> None:
    if kind not in _logged:
        LOG.info("Using fused sm_120 (Blackwell) LoRA %s kernel", kind)
        _logged.add(kind)


def is_sm120() -> bool:
    """True on Blackwell GeForce/RTX (sm_120/sm_121), where these kernels run."""
    global _IS_SM120
    if _IS_SM120 is None:
        if not torch.cuda.is_available():
            _IS_SM120 = False
        else:
            major, _ = torch.cuda.get_device_capability()
            _IS_SM120 = major == 12
    return _IS_SM120


def _all_none(*xs) -> bool:
    return all(x is None for x in xs)


def _base_ok(X, *weights) -> bool:
    """Common applicability: sm_120, bf16 input and bf16 frozen base weights."""
    if not is_sm120() or X.dtype != torch.bfloat16:
        return False
    return all(w is not None and w.dtype == torch.bfloat16 for w in weights)


def _proj_lora_ok(specs):
    """specs: list of (W, A, B, S, Q, Mag, b, lb). Check all are fused-eligible
    (unquantized bf16 base, active LoRA, no DoRA/bias) and share one scaling."""
    scalings = set()
    for _W, A, _B, S, Q, Mag, b, lb in specs:
        if A is None or not _all_none(Q, Mag, b, lb):
            return False
        scalings.add(S)
    return len(scalings) == 1


def _maybe_fused_proj(X, specs, fn):
    """Shared guard+dispatch for QKV/QK. specs: list of (W,A,B,S,Q,Mag,b,lb)."""
    weights = [s[0] for s in specs]
    if not _base_ok(X, *weights) or not _proj_lora_ok(specs):
        return None
    try:
        from .forward import _valid_tiles
    except ImportError:
        return None
    H = X.shape[-1]
    M = X.numel() // H
    N = sum(w.shape[0] for w in weights)
    if not _valid_tiles(M, N, H):
        return None
    _log_once("QKV/QK projection")
    return fn()


def maybe_lora_qkv(
    X,
    qW,
    qA,
    qB,
    qS,
    qQ,
    qMag,
    qb,
    qlb,
    kW,
    kA,
    kB,
    kS,
    kQ,
    kMag,
    kb,
    klb,
    vW,
    vA,
    vB,
    vS,
    vQ,
    vMag,
    vb,
    vlb,
):
    """Fused Q/K/V projections, else None (caller passes raw get_lora_parameters)."""
    specs = [
        (qW, qA, qB, qS, qQ, qMag, qb, qlb),
        (kW, kA, kB, kS, kQ, kMag, kb, klb),
        (vW, vA, vB, vS, vQ, vMag, vb, vlb),
    ]

    def run():
        from .attn import lora_qkv

        return lora_qkv(X, qW, qA, qB, kW, kA, kB, vW, vA, vB, qS)

    return _maybe_fused_proj(X, specs, run)


def maybe_lora_qk(
    X, qW, qA, qB, qS, qQ, qMag, qb, qlb, kW, kA, kB, kS, kQ, kMag, kb, klb
):
    """Fused Q/K projections, else None."""
    specs = [(qW, qA, qB, qS, qQ, qMag, qb, qlb), (kW, kA, kB, kS, kQ, kMag, kb, klb)]

    def run():
        from .attn import lora_qk

        return lora_qk(X, qW, qA, qB, kW, kA, kB, qS)

    return _maybe_fused_proj(X, specs, run)


def maybe_lora_o(X, OW, OA, OB, OS, OQ, OMag, Ob, Olb):
    """Fused output projection: out = X @ OW^T + s*(X@OA^T)@OB^T, else None."""
    if not _base_ok(X, OW):
        return None
    if not _all_none(OQ, OMag, Ob, Olb) or OA is None:
        return None
    try:
        from .forward import _valid_tiles
    except ImportError:
        return None
    H = X.shape[-1]
    M = X.numel() // H
    if not _valid_tiles(M, OW.shape[0], H):
        return None
    from .autograd import lora_dense

    _log_once("O projection")
    return lora_dense(X, OW, OA, OB, OS)


def maybe_lora_mlp_glu(
    X,
    gW,
    gA,
    gB,
    gS,
    gQ,
    gMag,
    gb,
    gLB,
    uW,
    uA,
    uB,
    uS,
    uQ,
    uMag,
    ub,
    uLB,
    dW,
    dA,
    dB,
    dS,
    dQ,
    dMag,
    db,
    dLB,
    geglu=False,
):
    """Return the fused-kernel GLU-MLP output if applicable on this hardware/config,
    else None. The caller must only invoke this when no dropout is active (the
    fused kernel does not apply gate/up dropout)."""
    if not _base_ok(X, gW, uW, dW):
        return None
    if not _all_none(gQ, uQ, dQ):  # no quantized base
        return None
    if not _all_none(gMag, uMag, dMag):  # no DoRA
        return None
    if not _all_none(gb, ub, db, gLB, uLB, dLB):  # no bias
        return None
    if gA is None or uA is None or dA is None:  # LoRA must be active
        return None
    if gA.shape[0] != uA.shape[0]:  # gate/up fused -> shared rank
        return None
    if not (gS == uS == dS):  # single shared scaling
        return None

    try:
        from .forward import _valid_tiles
    except ImportError:
        return None  # cutlass-dsl not installed -> fall back

    H = X.shape[-1]
    M = X.numel() // H
    inter = gW.shape[0]
    # kernel needs an exact CTA-tile fit (no OOB tiles): gate+up [M,2I,H], down [M,H,inter]
    if not _valid_tiles(M, 2 * inter, H) or not _valid_tiles(M, H, inter):
        return None

    from .mlp import lora_mlp_geglu, lora_mlp_swiglu

    _log_once("GeGLU MLP" if geglu else "SwiGLU MLP")
    fn = lora_mlp_geglu if geglu else lora_mlp_swiglu
    return fn(X, gW, gA, gB, uW, uA, uB, dW, dA, dB, gS)


def maybe_lora_mlp_swiglu(*args):
    return maybe_lora_mlp_glu(*args, geglu=False)


def maybe_lora_mlp_geglu(*args):
    return maybe_lora_mlp_glu(*args, geglu=True)
