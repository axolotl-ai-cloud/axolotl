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


def is_quant_tensor_subclass(W: torch.Tensor) -> bool:
    """True for torchao quant tensor subclasses (NF4Tensor,
    AffineQuantizedTensor, etc.) — anything that is not plain
    ``torch.Tensor`` or ``torch.nn.Parameter``. ``type(W) is not
    torch.Tensor`` alone is unsafe: ``Parameter`` is a *subclass* of
    Tensor, not the same type, so the bare check misclassifies every
    unquantized PEFT base weight.
    """
    return type(W) is not torch.Tensor and type(W) is not torch.nn.Parameter


_FP8_E4M3_MAX = 448.0


def _is_fp8_rowwise_weight(W: torch.Tensor) -> bool:
    """True for a torchao Float8 weight-only tensor with rowwise (per-output)
    e4m3 scales — the layout torchao's Float8WeightOnlyConfig produces and the
    one ``torch._scaled_mm`` can consume directly for the forward base GEMM."""
    qd = getattr(W, "qdata", None)
    sc = getattr(W, "scale", None)
    return (
        qd is not None
        and sc is not None
        and qd.dtype == torch.float8_e4m3fn
        and sc.numel() == qd.shape[0]  # one scale per output channel
    )


def fp8_base_forward(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor | None:
    """Forward base GEMM ``X @ W^T`` run natively in fp8 on the tensor cores.

    For a frozen torchao Float8 weight (``W.qdata`` e4m3 ``[out, in]``,
    ``W.scale`` ``[out, 1]``) the weight is already quantized, so only the
    activation is quantized at runtime (rowwise, dynamic) and the GEMM runs via
    ``torch._scaled_mm`` instead of dequantizing the weight to bf16. This is the
    forward half of fp8 LoRA: backward ``dX`` stays bf16 (its rowwise weight
    scale sits on the contraction axis), keeping a single fp8 weight copy and
    clean gradients.

    Returns the ``[*, out]`` result in ``X.dtype``, or ``None`` when the fast
    path does not apply (non-fp8 weight, unsupported hardware/shape) so the
    caller can fall back to the dequantize path.
    """
    if X.dtype not in (torch.bfloat16, torch.float16):
        return None
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
        return None  # fp8 tensor cores: Ada (8.9), Hopper (9.0), Blackwell (10.0/12.0)
    if not _is_fp8_rowwise_weight(W):
        return None
    qd, wsc = W.qdata, W.scale
    out_features, in_features = qd.shape
    X2 = X.reshape(-1, in_features) if X.dim() != 2 else X
    if X2.shape[1] % 16 or out_features % 16:  # _scaled_mm 16-element constraint
        return None
    amax = X2.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)
    x_inv_scale = amax / _FP8_E4M3_MAX
    xq = (X2 / x_inv_scale).to(torch.float8_e4m3fn)
    out = torch._scaled_mm(
        xq,
        qd.t(),  # [in, out], column-major — the layout _scaled_mm wants for mat2
        scale_a=x_inv_scale.to(torch.float32),
        scale_b=wsc.reshape(1, out_features).to(torch.float32),
        out_dtype=X.dtype,
    )
    return out.view(*X.shape[:-1], out_features) if X.dim() != 2 else out


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


def dequantize_weight(
    W: torch.Tensor,
    quant_state: QuantState | list | None = None,
    transpose: bool = False,
) -> torch.Tensor:
    """Unified dequantization for both torchao and bnb quantized weights.

    For torchao tensor subclasses (AffineQuantizedTensor, NF4Tensor), dequantizes
    using the appropriate instance method. For bnb Params4bit, delegates to the
    optimized CUDA kernel in ``dequantize``.

    Args:
        W: Quantized weight tensor ``[out_features, in_features]``.
        quant_state: bnb ``QuantState`` (None for torchao / unquantized).
        transpose: If True, return ``[in_features, out_features]``.

    Returns:
        Dequantized float tensor, optionally transposed.
    """
    # torchao path: tensor subclass with embedded quantization state
    if quant_state is None and is_quant_tensor_subclass(W):
        result = None
        # NF4Tensor (check first — NF4Tensor.dequantize is a static method)
        if hasattr(W, "get_original_weight"):
            result = W.get_original_weight()
        else:
            # AffineQuantizedTensor (INT4, etc.)
            try:
                result = W.dequantize()
            except (TypeError, RuntimeError):
                pass
        if result is not None:
            return result.t() if transpose else result

    # bnb path: transpose input before the CUDA kernel (existing convention)
    if transpose:
        return dequantize(W.t(), quant_state)
    return dequantize(W, quant_state)
