"""Dequantization utilities for `bitsandbytes` and FP8 integration."""

import ctypes

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import QuantState, get_ptr
from packaging.version import Version

cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4

CUDA_STREAM: torch.cuda.Stream | None = None
HAS_CUDA_STREAM: bool = Version(bnb.__version__) > Version("0.43.3")


def dequantize_fp8(
    W: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 block-quantized weights: W_dequant = W_fp8 * scale_inv.

    Args:
        W: FP8 weight tensor [out_features, in_features] in float8_e4m3fn.
        scale_inv: Per-block inverse scale [ceil(out/block), ceil(in/block)]
            or per-tensor scalar.
        dtype: Output dtype (default bf16).

    Returns:
        Dequantized tensor in the specified dtype.
    """
    W_float = W.to(dtype)
    if scale_inv.numel() == 1:
        return W_float * scale_inv.to(dtype)
    if scale_inv.dim() == 2 and W.dim() == 2:
        sr, sc = scale_inv.shape
        br = W.shape[0] // sr
        bc = W.shape[1] // sc
        # If dimensions are exactly divisible, use fast reshape path
        if sr * br == W.shape[0] and sc * bc == W.shape[1]:
            return (
                W_float.reshape(sr, br, sc, bc) * scale_inv[:, None, :, None].to(dtype)
            ).reshape(W.shape)
        # Tail-block handling: compute actual block size (ceil division),
        # tile scale_inv to cover full shape, then crop to W's dimensions
        br_ceil = -(-W.shape[0] // sr)  # ceil(rows / scale_rows) = block_size
        bc_ceil = -(-W.shape[1] // sc)
        scale_expanded = (
            scale_inv.to(dtype)
            .repeat_interleave(br_ceil, dim=0)
            .repeat_interleave(bc_ceil, dim=1)
        )[: W.shape[0], : W.shape[1]]
        return W_float * scale_expanded
    return W_float * scale_inv.to(dtype)


def dequantize(
    W: torch.Tensor,
    quant_state: QuantState | list | torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fast NF4 dequantization using `bitsandbytes` CUDA kernels.

    Performs efficient dequantization of weights from NF4 format using `bitsandbytes`'
    optimized CUDA implementations. Supports both legacy list and new `QuantState`
    formats.

    Args:
        W: Quantized weight tensor to dequantize
        quant_state: Quantization state containing metadata needed for
            dequantization. Can be either a `QuantState` object or legacy list format.
            If None, returns `W` unchanged.
        out: Optional output tensor for storing dequantized results. Must match
            expected shape and dtype if provided.

    Returns:
        Dequantized tensor in the specified dtype (fp16 or bf16). Will be transposed if
        input `W` was transposed.

    Raises:
        AssertionError: If provided output tensor doesn't match expected shape / dtype.

    Note:
        Uses CUDA streams for better performance when available in newer `bitsandbytes`
        versions (>0.43.3).
    """
    if quant_state is None:
        return W

    # FP8 path: quant_state is actually scale_inv tensor
    if W.dtype == torch.float8_e4m3fn:
        scale_inv = quant_state
        # Caller may pass W.t() (non-contiguous) — dequantize in original
        # layout then transpose back so the result shape matches the input.
        if not W.is_contiguous() and W.dim() == 2:
            return dequantize_fp8(W.t(), scale_inv).t()
        return dequantize_fp8(W, scale_inv)

    # Get the target device from input tensor W
    target_device = W.device

    # Extract quantization state
    if not isinstance(quant_state, list):
        # New style quant_state class
        # Non-double-quantized models have offset=None and state2=None
        if quant_state.offset is None or quant_state.state2 is None:
            # Fall back to bitsandbytes standard dequantize
            return bnb.functional.dequantize_4bit(W, quant_state, quant_type="nf4")
        absmax = quant_state.absmax.to(target_device)
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset.to(target_device)
        state2 = quant_state.state2
        absmax2 = state2.absmax.to(target_device)
        code2 = state2.code.to(target_device)
        blocksize2 = state2.blocksize
    else:
        # Legacy list format
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        absmax = absmax.to(target_device)
        offset, state2 = compressed_stats
        offset = offset.to(target_device)
        absmax2, code2, blocksize2, _, _, _, _ = state2
        absmax2 = absmax2.to(target_device)
        code2 = code2.to(target_device)

    # Setup output tensor on the same device as input
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=target_device)
    else:
        assert out.shape == shape and out.dtype == dtype
        out = out.to(target_device)

    # Dequantize statistics on the target device
    n_elements_absmax: int = absmax.numel()
    out_absmax: torch.Tensor = torch.empty(
        n_elements_absmax, dtype=torch.float32, device=target_device
    )
    ptr_out_absmax: int = get_ptr(out_absmax)

    # Use CUDA stream if available
    if HAS_CUDA_STREAM:
        global CUDA_STREAM
        if CUDA_STREAM is None:
            CUDA_STREAM = torch.cuda.current_stream(target_device)

        cdequantize_blockwise_fp32(
            get_ptr(code2),
            get_ptr(absmax),
            get_ptr(absmax2),
            ptr_out_absmax,
            ctypes.c_int(blocksize2),
            ctypes.c_int(n_elements_absmax),
            CUDA_STREAM,
        )
    else:
        cdequantize_blockwise_fp32(
            get_ptr(code2),
            get_ptr(absmax),
            get_ptr(absmax2),
            ptr_out_absmax,
            ctypes.c_int(blocksize2),
            ctypes.c_int(n_elements_absmax),
        )

    out_absmax += offset

    # Choose appropriate dequantization function
    fx = (
        cdequantize_blockwise_fp16_nf4
        if dtype == torch.float16
        else cdequantize_blockwise_bf16_nf4
    )

    # Dequantize weights
    if HAS_CUDA_STREAM:
        fx(
            get_ptr(None),
            get_ptr(W),
            ptr_out_absmax,
            get_ptr(out),
            ctypes.c_int(blocksize),
            ctypes.c_int(out.numel()),
            CUDA_STREAM,
        )
    else:
        fx(
            get_ptr(None),
            get_ptr(W),
            ptr_out_absmax,
            get_ptr(out),
            ctypes.c_int(blocksize),
            ctypes.c_int(out.numel()),
        )

    # Handle transposed data
    is_transposed: bool = W.shape[0] == 1
    return out.t() if is_transposed else out
