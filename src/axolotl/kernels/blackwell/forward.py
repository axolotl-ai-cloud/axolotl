# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
"""Torch-facing wrapper for the fused sm_120 LoRA dense GEMM.

Computes  Y = X @ W^T + scaling * (X @ A^T) @ B^T  on Blackwell GeForce/RTX
(sm_120) by reusing NVIDIA's pipelined dense-GEMM mainloop for the base path and
fusing the low-rank LoRA correction into the register accumulator (see
``lora_dense_gemm.Sm120GemmKernel``). XA = scaling * (X @ A^T) is precomputed in a
cheap separate pass (once per M-row) and applied as a fused in-register epilogue;
the rank is zero-padded to a multiple of the MMA k-dim. Folding XA into the
mainloop was tried and regressed (it recomputes XA per N-tile) — see project notes.
"""

import json
import os

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass.cute.runtime import from_dlpack

from .lora_dense_gemm import Sm120GemmKernel

_MMA_K = 16
_compiled_cache: dict = {}
_cfg_cache: dict = {}

# CTA tiles (M in {64,128}, N in {64,128}, K in {64,128}). (128,256,64) is pruned:
# it is pathological on this kernel (~20x slower) and never the best choice.
_CANDIDATE_TILES = [
    (64, 64, 64),
    (64, 128, 64),
    (128, 64, 64),
    (128, 128, 64),
    (128, 128, 128),
]
# Warp tiling. ONLY (2,2,1) is correct: the vendored kernel hardcodes the
# ldmatrix.x4 copy atoms and the N-permutation *2 factor around a 2x2 warp grid,
# so other layouts compile and run but produce wrong results (verified). Tuning
# atom_layout would require reworking those copy/epilogue atoms — left as a
# constructor knob but not searched.
_CANDIDATE_ATOMS = [(2, 2, 1)]

# Bump to invalidate on-disk autotune entries when the kernel/config space changes.
_CACHE_VERSION = "v2"


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _valid_tiles(M: int, N: int, K: int):
    # OOB tiles aren't supported (TMA store), so require exact divisibility.
    return [
        t for t in _CANDIDATE_TILES if M % t[0] == 0 and N % t[1] == 0 and K % t[2] == 0
    ]


def _valid_configs(M: int, N: int, K: int):
    """(tile, atom_layout) pairs where the tile fits the problem and the warp
    permutation (a*16) divides the tile."""
    cfgs = []
    for t in _valid_tiles(M, N, K):
        for a in _CANDIDATE_ATOMS:
            if (
                t[0] % (a[0] * 16) == 0
                and t[1] % (a[1] * 16) == 0
                and t[2] % (a[2] * 16) == 0
            ):
                cfgs.append((t, a))
    return cfgs


def _norm_config(c):
    """Accept a bare tile (-> default atom (2,2,1)) or a (tile, atom) pair."""
    if c is None:
        return None
    return c if isinstance(c[0], tuple) else (tuple(c), (2, 2, 1))


# Persist autotune results across runs so training doesn't re-pay the per-shape
# compile+time sweep. Keyed by device name + problem shape.
_CACHE_FILE = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "axolotl",
    "blackwell_lora_autotune.json",
)
_disk_cache = None


def _disk():
    global _disk_cache
    if _disk_cache is None:
        try:
            with open(_CACHE_FILE) as f:
                _disk_cache = json.load(f)
        except Exception:
            _disk_cache = {}
    return _disk_cache


def _disk_save():
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)
        with open(_CACHE_FILE, "w") as f:
            json.dump(_disk_cache, f)
    except Exception:
        pass


def _as_mkl(t: torch.Tensor):
    """Zero-copy (rows, cols, 1) view of a row-major [rows, cols] GPU tensor,
    matching the (M,K,L) k-major layout the kernel expects (L=1, batch stride
    rows*cols). No allocation or host copy."""
    view = t.unsqueeze(0).permute(
        1, 2, 0
    )  # (rows, cols, 1), strides (cols, 1, rows*cols)
    return from_dlpack(view, assumed_align=16)


def _run_lora_gemm(
    X: torch.Tensor,  # [M, K] bf16, row-major
    W: torch.Tensor,  # [N, K] bf16, row-major (the [out, in] base weight)
    XA: torch.Tensor,  # [M, r_pad] bf16  (already scaled, rank padded to %16)
    B_lora: torch.Tensor,  # [N, r_pad] bf16  (rank padded to %16)
    config=None,  # bare tile or (tile, atom_layout); None -> autotune
    out: torch.Tensor = None,  # optional [M, N] bf16 output buffer (in-place)
) -> torch.Tensor:
    """Core fused LoRA GEMM: Y = X @ W^T + XA @ B_lora^T (the LoRA scaling is
    expected to be folded into XA). Used by both forward and the dX backward.

    `out` lets the caller reuse an existing [M, N] buffer (e.g. write dX over the
    input X once X is no longer needed) to avoid a fresh allocation."""
    M, K = X.shape
    N = W.shape[0]
    r_pad = XA.shape[1]

    # output is 16-bit: the epilogue stores via stmatrix.x4 (16b); fp32 accum
    # is converted to bf16 after the in-register LoRA add.
    Y = (
        out
        if out is not None
        else torch.empty(M, N, device=X.device, dtype=torch.bfloat16)
    )
    a_cute = _as_mkl(X)  # (M, K, 1) k-major
    b_cute = _as_mkl(W)  # (N, K, 1) k-major
    c_cute = _as_mkl(Y)  # (M, N, 1) n-major
    mXA = from_dlpack(XA, assumed_align=16)
    mB_lora = from_dlpack(B_lora, assumed_align=16)

    stream = cutlass_torch.default_stream()

    def _compiled_for(cfg):
        tile, atom = cfg
        ck = (M, N, K, r_pad, tile, atom)
        if ck not in _compiled_cache:
            gemm = Sm120GemmKernel(cutlass.Float32, tile, atom)
            gemm.lora_rank_padded = r_pad
            max_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(1)
            _compiled_cache[ck] = cute.compile(
                gemm, a_cute, b_cute, c_cute, mXA, mB_lora, max_clusters, stream
            )
        return _compiled_cache[ck]

    cfg = _norm_config(config)
    if cfg is None:
        sk = (M, N, K, r_pad)
        if sk not in _cfg_cache:
            dkey = (
                f"{_CACHE_VERSION}|{torch.cuda.get_device_name()}|{M},{N},{K},{r_pad}"
            )
            disk = _disk()
            if dkey in disk:
                _cfg_cache[sk] = (tuple(disk[dkey][0]), tuple(disk[dkey][1]))
            else:
                # reference (bf16, what the kernel should compute) to reject any
                # config that produces wrong output — some (tile, atom) combos
                # compile + run fast but are numerically wrong (e.g. (2,4,1)).
                ref = (X @ W.t()).addmm_(XA, B_lora.t())
                best = _autotune(
                    _valid_configs(M, N, K),
                    _compiled_for,
                    (a_cute, b_cute, c_cute, mXA, mB_lora, stream),
                    validate=lambda: (
                        (Y - ref).abs().max() < 0.1 * ref.abs().max() + 1e-3
                    ),
                )
                _cfg_cache[sk] = best
                disk[dkey] = [list(best[0]), list(best[1])]
                _disk_save()
        cfg = _cfg_cache[sk]

    _compiled_for(cfg)(a_cute, b_cute, c_cute, mXA, mB_lora, stream)
    return Y


def pad_rank(mat_cols_first: torch.Tensor, r: int, r_pad: int) -> torch.Tensor:
    """Zero-pad a [*, r] tensor's last dim to r_pad (a multiple of the MMA k-dim)."""
    out = torch.zeros(
        *mat_cols_first.shape[:-1],
        r_pad,
        device=mat_cols_first.device,
        dtype=torch.bfloat16,
    )
    out[..., :r] = mat_cols_first
    return out


def lora_dense_forward(
    X: torch.Tensor,  # [M, K] bf16
    W: torch.Tensor,  # [N, K] bf16  (out, in)
    A: torch.Tensor,  # [r, K] bf16
    B: torch.Tensor,  # [N, r] bf16
    scaling: float,
    config=None,
) -> torch.Tensor:
    """Y = X @ W^T + scaling * (X @ A^T) @ B^T.

    XA = scaling*(X@A^T) is precomputed (cheap, once per M-row) and applied as a
    fused in-register epilogue by the kernel. `config` is a bare tile, a
    (tile, atom_layout) pair, or None to autotune (cached per shape + on disk).
    """
    r = A.shape[0]
    r_pad = _round_up(max(r, _MMA_K), _MMA_K)
    XA = pad_rank(scaling * (X @ A.t()), r, r_pad)  # [M, r_pad]
    B_lora = pad_rank(B, r, r_pad)  # [N, r_pad]
    return _run_lora_gemm(X, W, XA, B_lora, config)


def _autotune(configs, compiled_for, launch_args, validate=None):
    """Pick the fastest valid (tile, atom_layout) for this shape. Configs that
    fail to compile (e.g. SMEM overflow -> ValueError) are skipped, as are
    configs whose output fails `validate` (defends against fast-but-wrong
    layouts that compile and run but compute the wrong result)."""
    best, best_t = None, float("inf")
    for cfg in configs:
        try:
            fn = compiled_for(cfg)
            fn(*launch_args)
            torch.cuda.synchronize()
            if validate is not None and not bool(validate()):
                continue
        except Exception:
            continue
        for _ in range(3):
            fn(*launch_args)
        torch.cuda.synchronize()
        st, en = torch.cuda.Event(True), torch.cuda.Event(True)
        st.record()
        for _ in range(10):
            fn(*launch_args)
        en.record()
        torch.cuda.synchronize()
        ms = st.elapsed_time(en) / 10
        if ms < best_t:
            best_t, best = ms, cfg
    return best or ((128, 128, 64), (2, 2, 1))
