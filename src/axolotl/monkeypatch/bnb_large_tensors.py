"""Keep bitsandbytes' int32 C ABI while handling larger tensors in bounded calls."""

import math

_MAX_ELEMENTS = 2**31 - 1


def patch_bnb_large_tensors():
    import bitsandbytes.functional as F

    if getattr(F.quantize_4bit, "_axolotl_chunked", False):
        return
    from axolotl.utils.nf4 import dequantize_bnb_4bit, quantize_bnb_4bit

    original_quantize = F.quantize_4bit
    original_dequantize = F.dequantize_4bit

    def quantize(
        A,
        absmax=None,
        out=None,
        blocksize=None,
        compress_statistics=False,
        quant_type="fp4",
        quant_storage=None,
    ):
        import torch

        storage = quant_storage or torch.uint8
        if A.numel() <= _MAX_ELEMENTS:
            return original_quantize(
                A, absmax, out, blocksize, compress_statistics, quant_type, storage
            )
        data, state = quantize_bnb_4bit(
            A,
            blocksize=blocksize or 64,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=storage,
            chunk_size=min(2**26, _MAX_ELEMENTS),
        )
        if out is not None:
            data = out.copy_(data)
        if absmax is not None:
            state.absmax = absmax.copy_(state.absmax)
        return data, state

    def dequantize(
        A, quant_state=None, absmax=None, out=None, blocksize=None, quant_type="fp4"
    ):
        shape = (
            quant_state.shape if quant_state is not None else getattr(out, "shape", ())
        )
        if math.prod(shape) <= _MAX_ELEMENTS:
            return original_dequantize(
                A, quant_state, absmax, out, blocksize, quant_type
            )
        if quant_state is None:
            quant_state = F.QuantState(
                absmax=absmax,
                shape=out.shape,
                dtype=out.dtype,
                blocksize=blocksize or 64,
                quant_type=quant_type,
            )
        result = dequantize_bnb_4bit(
            A, quant_state, out=out, chunk_size=min(2**26, _MAX_ELEMENTS)
        )
        return result.t() if A.shape[0] == 1 else result

    quantize._axolotl_chunked = True
    F.quantize_4bit = quantize
    F.dequantize_4bit = dequantize
