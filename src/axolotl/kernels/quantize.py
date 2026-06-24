"""Dequantization utilities for `bitsandbytes` and FP8 integration."""

import ctypes
from typing import List

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import QuantState, get_ptr

cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4

# Cached per-device: per-call current_stream() measurably slows this hot path.
CUDA_STREAM: dict[torch.device, torch.cuda.Stream] = {}


def _ctypes_nf4_dequant(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: torch.Tensor,
    absmax2: torch.Tensor,
    offset: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Direct-ctypes NF4 double-dequant body (Unsloth-derived fast path)."""
    target_device = W.device
    out = torch.empty(tuple(shape), dtype=dtype, device=target_device)
    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(
        n_elements_absmax, dtype=torch.float32, device=target_device
    )

    stream = CUDA_STREAM.get(target_device)
    if stream is None:
        stream = CUDA_STREAM.setdefault(
            target_device, torch.cuda.current_stream(target_device)
        )

    cdequantize_blockwise_fp32(
        get_ptr(code2),
        get_ptr(absmax),
        get_ptr(absmax2),
        get_ptr(out_absmax),
        ctypes.c_int(blocksize2),
        ctypes.c_int(n_elements_absmax),
        stream,
    )
    out_absmax += offset

    fx = (
        cdequantize_blockwise_fp16_nf4
        if dtype == torch.float16
        else cdequantize_blockwise_bf16_nf4
    )
    fx(
        get_ptr(None),
        get_ptr(W),
        get_ptr(out_absmax),
        get_ptr(out),
        ctypes.c_int(blocksize),
        ctypes.c_int(out.numel()),
        stream,
    )

    # bnb convention: leading-dim-1 packed weight signals a transposed view.
    if W.shape[0] == 1:
        return out.t()
    return out


@torch.library.custom_op("axolotl::nf4_dequantize", mutates_args=())
def _nf4_dequantize_op(
    W: torch.Tensor,
    absmax: torch.Tensor,
    code2: torch.Tensor,
    absmax2: torch.Tensor,
    offset: torch.Tensor,
    blocksize: int,
    blocksize2: int,
    shape: List[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Opaque-to-Dynamo wrapper around the direct-ctypes NF4 dequant body."""
    return _ctypes_nf4_dequant(
        W, absmax, code2, absmax2, offset, blocksize, blocksize2, shape, dtype
    )


@_nf4_dequantize_op.register_fake
def _(W, absmax, code2, absmax2, offset, blocksize, blocksize2, shape, dtype):
    """FakeTensor shape/dtype inference for the registered op (trace-time only)."""
    out = torch.empty(tuple(shape), dtype=dtype, device=W.device)
    if W.shape[0] == 1:
        return out.t()
    return out


def dequantize_fp8(
    W: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 block-quantized weights: W_dequant = W_fp8 * scale_inv."""
    W_float = W.to(dtype)
    if scale_inv.numel() == 1:
        return W_float * scale_inv.to(dtype)
    if scale_inv.dim() == 2 and W.dim() == 2:
        sr, sc = scale_inv.shape
        br = W.shape[0] // sr
        bc = W.shape[1] // sc
        if sr * br == W.shape[0] and sc * bc == W.shape[1]:
            return (
                W_float.reshape(sr, br, sc, bc) * scale_inv[:, None, :, None].to(dtype)
            ).reshape(W.shape)
        # Tail blocks: ceil-div the block size, tile scale_inv, crop to W.
        br_ceil = -(-W.shape[0] // sr)
        bc_ceil = -(-W.shape[1] // sc)
        scale_expanded = (
            scale_inv.to(dtype)
            .repeat_interleave(br_ceil, dim=0)
            .repeat_interleave(bc_ceil, dim=1)
        )[: W.shape[0], : W.shape[1]]
        return W_float * scale_expanded
    return W_float * scale_inv.to(dtype)


_FLOAT8_CLS: type | None = None
_FLOAT8_CHECKED = False


def _is_float8_tensor(W: torch.Tensor) -> bool:
    """A torchao ``Float8Tensor`` (blockwise-FP8 base weight, e.g. DSV4 non-experts).

    It reports a logical bf16 ``dtype`` and carries its own block scale, so the fused
    LoRA kernels pass it as ``W`` with ``quant_state=None`` — detect it by class."""
    global _FLOAT8_CLS, _FLOAT8_CHECKED
    if not _FLOAT8_CHECKED:
        _FLOAT8_CHECKED = True
        try:
            from torchao.quantization import Float8Tensor as _F8

            _FLOAT8_CLS = _F8
        except Exception:
            _FLOAT8_CLS = None
    return _FLOAT8_CLS is not None and isinstance(W, _FLOAT8_CLS)


def dequantize(
    W: torch.Tensor,
    quant_state: QuantState | torch.Tensor | None = None,
) -> torch.Tensor:
    """NF4 / FP8 dequantization; under `torch.compile` NF4 dispatches via `torch.ops.axolotl.nf4_dequantize`."""
    # torchao Float8Tensor carries its own scale (a transposed view is still a Float8Tensor),
    # so dequant to bf16 here and let the downstream matmul/addmm_ stay a plain bf16 GEMM.
    if _is_float8_tensor(W):
        return W.dequantize().to(torch.bfloat16)

    if quant_state is None:
        return W

    if W.dtype == torch.float8_e4m3fn:
        scale_inv = quant_state
        # Caller may pass W.t() (non-contiguous); dequant in original layout, transpose back.
        if not W.is_contiguous() and W.dim() == 2:
            return dequantize_fp8(W.t(), scale_inv).t()
        return dequantize_fp8(W, scale_inv)

    # Non-double-quant: fall back to bnb's wrapper (rare in axolotl QLoRA).
    if quant_state.offset is None or quant_state.state2 is None:
        return bnb.functional.dequantize_4bit(W, quant_state, quant_type="nf4")

    target_device = W.device
    state2 = quant_state.state2
    absmax = quant_state.absmax.to(target_device)
    code2 = state2.code.to(target_device)
    absmax2 = state2.absmax.to(target_device)
    offset = quant_state.offset.to(target_device)

    if torch.compiler.is_compiling():
        return torch.ops.axolotl.nf4_dequantize.default(
            W,
            absmax,
            code2,
            absmax2,
            offset,
            quant_state.blocksize,
            state2.blocksize,
            list(quant_state.shape),
            quant_state.dtype,
        )

    return _ctypes_nf4_dequant(
        W,
        absmax,
        code2,
        absmax2,
        offset,
        quant_state.blocksize,
        state2.blocksize,
        quant_state.shape,
        quant_state.dtype,
    )
