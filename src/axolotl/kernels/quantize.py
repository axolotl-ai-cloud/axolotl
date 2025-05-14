"""Dequantization utilities for `bitsandbytes` integration."""

# pylint: disable=invalid-name,global-statement

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


def dequantize(
    W: torch.Tensor,
    quant_state: QuantState | list | None = None,
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

    # Get the target device from input tensor W
    target_device = W.device

    # Extract quantization state
    if not isinstance(quant_state, list):
        # New style quant_state class
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
