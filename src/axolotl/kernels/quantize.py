import ctypes

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import QuantState, get_ptr
from packaging.version import Version

cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4

CUDA_STREAM = None
HAS_CUDA_STREAM = Version(bnb.__version__) > Version("0.43.3")


def dequantize(
    W: torch.Tensor,
    quant_state: QuantState | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fast NF4 dequantization using bitsandbytes CUDA kernels"""
    if quant_state is None:
        return W

    # Extract quantization state
    if not isinstance(quant_state, list):
        # New style quant_state class
        absmax = quant_state.absmax
        shape = quant_state.shape
        dtype = quant_state.dtype
        blocksize = quant_state.blocksize
        offset = quant_state.offset
        state2 = quant_state.state2
        absmax2 = state2.absmax
        code2 = state2.code
        blocksize2 = state2.blocksize
    else:
        # Legacy list format
        absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
        offset, state2 = compressed_stats
        absmax2, code2, blocksize2, _, _, _, _ = state2

    # Setup output tensor
    if out is None:
        out = torch.empty(shape, dtype=dtype, device="cuda:0")
    else:
        assert out.shape == shape and out.dtype == dtype

    # Dequantize statistics
    n_elements_absmax = absmax.numel()
    out_absmax = torch.empty(n_elements_absmax, dtype=torch.float32, device="cuda:0")
    ptr_out_absmax = get_ptr(out_absmax)

    # Use CUDA stream if available
    if HAS_CUDA_STREAM:
        global CUDA_STREAM
        if CUDA_STREAM is None:
            CUDA_STREAM = torch.cuda.current_stream("cuda:0")

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
    is_transposed = W.shape[0] == 1
    return out.t() if is_transposed else out
