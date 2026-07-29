"""W2A8 int8 tensor-core forward for the λ == 1 regime.

At λ == 1 the effective weight is exactly `code · s16`, so a fake-quant linear is an
int8 GEMM plus a per-(token, tensor) fp32 rescale: integer accumulation is exact, and
the only float rounding left is the epilogue and the final cast. Backward stays dense
(STE), so gradients are identical to the eager path up to that same rounding.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .. import quant

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.libdevice import div_rn, rint
    except ModuleNotFoundError:  # pragma: no cover
        from triton.language.extra.cuda.libdevice import div_rn, rint

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

LOG = logging.getLogger(__name__)

# torch._int_mm wants more than 16 rows and K, N multiples of 8; cuBLASLt additionally
# rejects row counts that are not a multiple of 32 unless K is a multiple of 16
_MIN_ROWS: int = 32
_DEPTH_ALIGN: int = 16
_COL_ALIGN: int = 8

_MIN_CAPABILITY: tuple[int, int] = (8, 0)
_MIN_FEATURES: int = 512
_SUPPORTED_DTYPES: tuple[torch.dtype, ...] = (
    torch.bfloat16,
    torch.float16,
    torch.float32,
)

_RESCALE_BLOCK: int = 512
_ACT_QUANT_BLOCK: int = 1024
_CODES_BLOCK: int = 2048
_EVAL_CACHE_ATTR: str = "_ternary_int8_eval_cache"

_INT_MM_REJECTED: set[tuple[int, int, int]] = set()


# scale modes whose dequant grid is a trained parameter rather than a latent statistic
_TRAINED_SCALE_MODES: frozenset[str] = frozenset({"learnable", "learnable_row"})


@dataclass
class _EvalWeightCodes:
    """Int8 weight codes cached against the identity of the latent they came from."""

    version: int
    data_ptr: int
    codes: torch.Tensor
    scale: torch.Tensor
    stamp: tuple = ()


if _HAS_TRITON:

    @triton.jit
    def _rescale_kernel(
        acc_ptr,
        x_scale_ptr,
        w_scale_ptr,
        out_ptr,
        acc_row_stride,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fuse `int32 → fp32 · s_x · s16 → out_dtype` into one pass over the accumulator."""
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        acc = tl.load(acc_ptr + row * acc_row_stride + offsets, mask=mask, other=0)
        scale = tl.load(x_scale_ptr + row) * tl.load(w_scale_ptr)
        out = acc.to(tl.float32) * scale
        tl.store(
            out_ptr + row * out_row_stride + offsets,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _weight_codes_kernel(
        w_ptr,
        codes_ptr,
        scale_ptr,
        numel,
        EPS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Write `round_half_even(w / s)` clipped to `{-1, 0, +1}` as int8, in one pass."""
        pid = tl.program_id(0).to(tl.int64)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.load(scale_ptr).to(tl.float32), EPS)
        # div_rn, not `/`: only correctly-rounded division reproduces the oracle's ties
        code = rint(div_rn(w, scale))
        code = tl.minimum(tl.maximum(code, -1.0), 1.0)
        tl.store(codes_ptr + offsets, code.to(tl.int8), mask=mask)

    @triton.jit
    def _row_rescale_kernel(
        acc_ptr,
        x_scale_ptr,
        w_scale_ptr,
        out_ptr,
        acc_row_stride,
        out_row_stride,
        n_cols,
        BLOCK_SIZE: tl.constexpr,
    ):
        """`int32 -> fp32 · s_x · s16[n] -> out_dtype`, one weight scale per column.

        The per-tensor epilogue reads one scalar; a per-row weight scale is one value
        per *output* feature, so it is a vector load along the same axis the block
        already walks.
        """
        row = tl.program_id(0)
        offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        acc = tl.load(acc_ptr + row * acc_row_stride + offsets, mask=mask, other=0)
        w_scale = tl.load(w_scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        out = acc.to(tl.float32) * tl.load(x_scale_ptr + row) * w_scale
        tl.store(
            out_ptr + row * out_row_stride + offsets,
            out.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _act_quant_kernel(
        x_ptr,
        codes_ptr,
        scale_ptr,
        x_row_stride,
        n_cols,
        eps,
        qmax,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Per-token `s_x = max(|x|) / qmax` and `round_half_even(x / s_x)` in one pass."""
        row = tl.program_id(0)
        x_row = x_ptr + row * x_row_stride
        amax = 0.0
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            values = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(values)))
        scale = tl.maximum(amax / qmax, eps)
        tl.store(scale_ptr + row, scale)

        codes_row = codes_ptr + row * n_cols
        for start in tl.range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            values = tl.load(x_row + offsets, mask=mask, other=0.0).to(tl.float32)
            # div_rn + rint: torch's fp32 divide-then-round-half-even, tie-for-tie
            codes = rint(div_rn(values, scale))
            codes = tl.minimum(tl.maximum(codes, -qmax), qmax)
            tl.store(codes_row + offsets, codes.to(tl.int8), mask=mask)


def _align_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


@functools.lru_cache(maxsize=None)
def _capable_device(index: int) -> bool:
    return torch.cuda.get_device_capability(index) >= _MIN_CAPABILITY


def _float_matmul(x_codes: torch.Tensor, w_codes: torch.Tensor) -> torch.Tensor:
    """Exact int32 accumulation through a float GEMM, for shapes `_int_mm` rejects.

    Inputs of magnitude ≤ 127 survive even a TF32 mantissa unrounded and the fp32
    accumulator is exact below 2**24, so this stays bit-identical to the integer path.
    """
    return (x_codes.to(torch.float32) @ w_codes.to(torch.float32).t()).to(torch.int32)


def _int32_matmul(x_codes: torch.Tensor, w_codes: torch.Tensor) -> torch.Tensor:
    """Return the exact int32 `x_codes @ w_codes.T`, padding around `_int_mm` limits."""
    tokens, in_features = x_codes.shape
    out_features = w_codes.shape[0]
    x_codes = x_codes.contiguous()
    w_codes = w_codes.contiguous()

    if x_codes.is_cuda:
        rows = max(tokens, _MIN_ROWS)
        depth = _align_up(in_features, _DEPTH_ALIGN)
        cols = _align_up(out_features, _COL_ALIGN)
    else:
        rows, depth, cols = tokens, in_features, out_features

    # zero lanes contribute nothing to a dot product, so padding stays bit-exact
    if rows != tokens or depth != in_features:
        x_codes = F.pad(x_codes, (0, depth - in_features, 0, rows - tokens))
    if cols != out_features or depth != in_features:
        w_codes = F.pad(w_codes, (0, depth - in_features, 0, cols - out_features))

    key = (rows, depth, cols)
    if key not in _INT_MM_REJECTED:
        try:
            return torch._int_mm(x_codes, w_codes.t())[:tokens, :out_features]
        except (RuntimeError, NotImplementedError) as exc:
            _INT_MM_REJECTED.add(key)
            LOG.warning(
                "int8 GEMM %s unsupported here (%s); falling back to the float path",
                key,
                exc,
            )
    return _float_matmul(x_codes, w_codes)[:tokens, :out_features]


def _rescale(
    acc: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply `acc · (s_x · s16)` and cast — pinned multiply order, Triton or eager.

    `w_scale` is either a single value for the whole tensor or one per output
    feature; the per-row form takes a vector load along the axis the block already
    walks, so it costs the same pass.
    """
    tokens, out_features = acc.shape
    per_row = w_scale.numel() > 1
    if per_row and w_scale.numel() != out_features:
        raise ValueError(
            f"expected one weight scale per output feature, got {w_scale.numel()} "
            f"for {out_features}"
        )
    if not (_HAS_TRITON and acc.is_cuda):
        dense = w_scale.reshape(1, -1) if per_row else w_scale
        return (acc * (x_scale.reshape(tokens, 1).to(torch.float32) * dense)).to(
            out_dtype
        )

    out = torch.empty((tokens, out_features), dtype=out_dtype, device=acc.device)
    if tokens == 0 or out_features == 0:
        return out
    grid = (tokens, triton.cdiv(out_features, _RESCALE_BLOCK))
    kernel = _row_rescale_kernel if per_row else _rescale_kernel
    kernel[grid](
        acc,
        x_scale.reshape(-1).to(torch.float32).contiguous(),
        w_scale.reshape(-1).to(torch.float32).contiguous(),
        out,
        acc.stride(0),
        out.stride(0),
        out_features,
        BLOCK_SIZE=_RESCALE_BLOCK,
    )
    return out


def act_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flattened per-token int8 activation codes and fp32 scales for the GEMM.

    Bit-identical to `quant.act_quant_int8`, which is used verbatim whenever the
    fused kernel is unavailable.

    Args:
        x: Activation of shape `(..., in_features)`.

    Returns:
        `(codes, scale)` of shapes `(tokens, in_features)` int8 and `(tokens, 1)` fp32.
    """
    if not (_HAS_TRITON and x.is_cuda):
        codes, scale = quant.act_quant_int8(x)
        return codes.reshape(-1, codes.shape[-1]), scale.reshape(-1, 1)

    rows = x.reshape(-1, x.shape[-1]).contiguous()
    tokens, n_cols = rows.shape
    codes = torch.empty((tokens, n_cols), dtype=torch.int8, device=x.device)
    scale = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    if tokens:
        _act_quant_kernel[(tokens,)](
            rows,
            codes,
            scale,
            rows.stride(0),
            n_cols,
            quant.SCALE_EPS,
            float(quant.ACT_QMAX),
            BLOCK_SIZE=_ACT_QUANT_BLOCK,
            num_warps=8,
        )
    return codes, scale


def int8_gemm(
    x_codes: torch.Tensor,
    w_codes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute `(x_codes @ w_codes.T) * x_scale * w_scale` on int8 tensor cores.

    Args:
        x_codes: int8 activations of shape `(tokens, in_features)`.
        w_codes: int8 ternary weight codes of shape `(out_features, in_features)`.
        x_scale: fp32 per-token scale of shape `(tokens, 1)`.
        w_scale: fp32 weight scale — 0-dim per tensor, or `(out_features,)` per row.
        out_dtype: Output dtype.

    Returns:
        `(tokens, out_features)` tensor in `out_dtype`.

    Raises:
        ValueError: If the operands are not 2-D, the contraction dims disagree, or
            the scales do not have one entry per token and one per tensor.
    """
    if x_codes.dim() != 2 or w_codes.dim() != 2:
        raise ValueError(
            f"int8_gemm expects 2-D operands, got {tuple(x_codes.shape)} and "
            f"{tuple(w_codes.shape)}"
        )
    if x_codes.shape[1] != w_codes.shape[1]:
        raise ValueError(
            f"contraction dim mismatch: {x_codes.shape[1]} vs {w_codes.shape[1]}"
        )
    if x_scale.numel() != x_codes.shape[0]:
        raise ValueError(
            f"expected one activation scale per token, got {x_scale.numel()} for "
            f"{x_codes.shape[0]} tokens"
        )
    if w_scale.numel() not in (1, w_codes.shape[0]):
        raise ValueError(
            "int8_gemm expects a per-tensor weight scale or one per output feature, "
            f"got {w_scale.numel()} for {w_codes.shape[0]} outputs"
        )

    acc = _int32_matmul(x_codes, w_codes)
    return _rescale(acc, x_scale, w_scale, out_dtype)


def int8_forward_supported(
    in_features: int, out_features: int, dtype: torch.dtype
) -> bool:
    """Whether the int8 path is available and beneficial for this shape and device."""
    if dtype not in _SUPPORTED_DTYPES:
        return False
    if not torch.cuda.is_available():
        return False
    if not _capable_device(torch.cuda.current_device()):
        return False
    if in_features % _COL_ALIGN or out_features % _COL_ALIGN:
        return False
    return min(in_features, out_features) >= _MIN_FEATURES


def ternary_codes(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fused `quant.ternary_codes`: one pass from the latent weight to int8 codes.

    Args:
        weight: Latent weight on any device.
        scale: fp32 per-tensor scale, 0-dim.

    Returns:
        int8 codes shaped like `weight`.
    """
    if not (_HAS_TRITON and weight.is_cuda) or scale.numel() != 1:
        return quant.ternary_codes(weight, scale)

    weight = weight.contiguous()
    codes = torch.empty_like(weight, dtype=torch.int8)
    numel = weight.numel()
    if numel:
        _weight_codes_kernel[(triton.cdiv(numel, _CODES_BLOCK),)](
            weight,
            codes,
            scale.reshape(-1).to(torch.float32).contiguous(),
            numel,
            EPS=quant.SCALE_EPS,
            BLOCK=_CODES_BLOCK,
            num_warps=8,
        )
    return codes


def _derive_weight_codes(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    weight = module.weight.detach()
    trained = _trained_scale(module)
    if trained is not None:
        # a learnable mode stores its grid, so the codes come from the trained scale;
        # an absmean statistic re-derived from the latent is a different quantizer
        scale = trained.to(torch.float32)
        return ternary_codes(weight, scale), quant.f16_round_scale(scale).reshape(-1)
    is_baked = getattr(module, "is_baked", None)
    if is_baked() if callable(is_baked) else getattr(module, "baked", False):
        # a baked latent already holds exactly `code · s16`; absmean would shrink the scale
        scale = weight.abs().amax().to(torch.float32).clamp_min(quant.SCALE_EPS)
        return ternary_codes(weight, scale), scale
    scale = quant.absmean_scale(weight)
    codes = ternary_codes(weight, scale)
    # s16 rounded through the compute dtype: the weight side then matches the baked master
    scale_f16 = quant.f16_round_scale(scale).to(weight.dtype).to(torch.float32)
    return codes, scale_f16


def _trained_scale(module: torch.nn.Module) -> torch.Tensor | None:
    """Return the trained dequant scale of a learnable module, else `None`.

    `_scale()` undoes the log-space storage the parameter actually holds, and already
    declines the two-plane modes, whose grid no single scale can describe.
    """
    if getattr(module, "weight_scale", None) not in _TRAINED_SCALE_MODES:
        return None
    scale = getattr(module, "_scale", None)
    return None if scale is None else scale()


def _weight_codes(module: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    """Int8 codes for `module.weight`, cached only while the module is in eval mode."""
    weight = module.weight
    cached: _EvalWeightCodes | None = getattr(module, _EVAL_CACHE_ATTR, None)
    if module.training:
        # latents move every optimizer step; an int8 copy would only pin memory
        if cached is not None:
            delattr(module, _EVAL_CACHE_ATTR)
        return _derive_weight_codes(module)

    stamp = _cache_stamp(module)
    if (
        cached is not None
        and cached.version == weight._version
        and cached.data_ptr == weight.data_ptr()
        and cached.stamp == stamp
    ):
        return cached.codes, cached.scale

    codes, scale = _derive_weight_codes(module)
    setattr(
        module,
        _EVAL_CACHE_ATTR,
        _EvalWeightCodes(weight._version, weight.data_ptr(), codes, scale, stamp),
    )
    return codes, scale


def _cache_stamp(module: torch.nn.Module) -> tuple:
    """Identity of everything the cached codes were derived from beyond the weight.

    A learnable mode keeps its grid in a trained parameter, so the weight's version
    counter alone does not describe the codes — and a fused optimizer updates
    parameters without bumping that counter at all (the same blind spot that once let
    a fit-initialized module claim to be baked for a whole run). The scale's own
    identity and contents go into the key so a moved scale cannot be served stale.
    """
    scale = _trained_scale(module)
    if scale is None:
        return ()
    dense = scale.detach()
    return (dense.data_ptr(), dense._version, float(dense.sum()))


class Int8LinearSTE(torch.autograd.Function):
    """int8 forward with the dense straight-through backward of a fake-quant linear."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        weight: torch.Tensor,
        x: torch.Tensor,
        w_codes: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize `x` per token and run the int8 GEMM against `w_codes`."""
        x_codes, x_scale = act_quant_int8(x)
        out = int8_gemm(x_codes, w_codes, x_scale, w_scale, out_dtype=x.dtype)
        ctx.save_for_backward(x_codes, x_scale, w_codes, w_scale)
        ctx.x_shape = x.shape
        ctx.x_dtype = x.dtype
        return out.view(*x.shape[:-1], w_codes.shape[0])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Dense GEMMs against the dequantized operands — STE through both quantizers."""
        x_codes, x_scale, w_codes, w_scale = ctx.saved_tensors
        grad_out = grad_output.reshape(-1, w_codes.shape[0])
        grad_x = grad_w = None
        if ctx.needs_input_grad[1]:
            w_eff = (w_codes * w_scale).to(ctx.x_dtype)
            grad_x = (grad_out @ w_eff).view(ctx.x_shape)
        if ctx.needs_input_grad[0]:
            x_eff = (x_codes * x_scale).to(ctx.x_dtype)
            grad_w = grad_out.t() @ x_eff
        return grad_w, grad_x, None, None


def int8_linear_forward(
    module: torch.nn.Module, x: torch.Tensor
) -> torch.Tensor | None:
    """Run a `TernaryLinear` forward as an int8 GEMM, or `None` when it does not qualify.

    Args:
        module: Module holding a latent `weight` plus the `lambda_`, `group_size`,
            `activation_bits`, `baked` and `scale` attributes of `TernaryLinear`.
        x: Activation of shape `(..., in_features)`.

    Returns:
        `(..., out_features)` in `x`'s dtype, or `None` when the caller must fall back
        to the eager fake-quant path (λ < 1, weight-only QAT, group or learnable
        scales, unsupported shape or device).
    """
    weight = module.weight
    if getattr(module, "lambda_", 0.0) != 1.0:
        return None
    if getattr(module, "activation_bits", None) != 8:
        return None
    if getattr(module, "group_size", None) is not None:
        return None
    if getattr(module, "scale", None) is not None and _trained_scale(module) is None:
        return None
    if not (x.is_cuda and weight.is_cuda and x.device == weight.device):
        return None
    if x.dtype != weight.dtype:
        return None
    if not int8_forward_supported(weight.shape[1], weight.shape[0], weight.dtype):
        return None

    w_codes, w_scale = _weight_codes(module)
    # weight is passed only so the straight-through gradient reaches the latent
    return Int8LinearSTE.apply(weight, x, w_codes, w_scale)
