# mypy: disable-error-code="assignment, misc"
"""NVFP4-GEMM training: real FP4 forward + backward GEMMs on Blackwell.

This is a throughput feature, not a memory feature: master weights and
optimizer state stay bf16/fp32; only the GEMM operands are NVFP4. It mirrors
the FP8 training surface (module-swap on ``nn.Linear``) but is self-owned —
accelerate/torchao have no NVFP4 training path.

Foundations verified on sm_120 (RTX 5090, torchao 0.18):
  * the three GEMMs (fprop, dgrad, wgrad) run via ``torch._scaled_mm`` over
    NVFP4 operands and match dequant-matmul exactly;
  * two-level (per-tensor) scaling is REQUIRED — gradients otherwise underflow
    the per-block e4m3 scale floor (2^-9) and round to zero;
  * the quant->GEMM boundary compiles with zero graph breaks (the 2.9-3.6x
    speedup over bf16 only materializes under torch.compile).

Convergence recipe knobs (stochastic rounding on gradients, random Hadamard
transform on wgrad inputs) attach at the ``_quantize`` seam — see
``QuantPolicy``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import triton
from torch import nn
from triton import language as tl

LOG = logging.getLogger(__name__)

# Blackwell: B200/B300 = sm_100/sm_103, consumer (RTX 50xx, RTX PRO 6000) = sm_120.
# FP4 tensor cores exist on both families; do NOT blanket-gate on sm_100 only
# (the recurring hazard where the consumer SM is silently dropped).
_MIN_FP4_CAPABILITIES = ((10, 0), (12, 0))

_BLOCK_SIZE = 16  # NVFP4 is fixed at block_size 16
# torch._scaled_mm packs 2 FP4/byte and requires the packed contraction dim
# (= logical/2) to be divisible by 16, so logical contraction must be a
# multiple of 32. The token dimension (M) is padded to this alignment.
_GEMM_ALIGN = 32

# Paper ablation sweet spot: 16x16 random Hadamard (4x4 worse, 128x128 marginal).
# 16 | 32 (_GEMM_ALIGN) so the block tiles the padded contraction dim cleanly.
_HAD_DIM = 16
# FP4 E2M1 normalized-mantissa exponent range: levels are 2^e * {1, 1.5} for
# e in {0,1,2} (above the denormal step), so the half-step granularity at which
# SR dithers lives in this clamp.
_FP4_EXP_LO, _FP4_EXP_HI = 0, 2


def nvfp4_supported() -> tuple[bool, str]:
    """Return (ok, reason). ok=False means refuse with `reason`."""
    if not torch.cuda.is_available():
        return False, "NVFP4 training requires CUDA"
    cap = torch.cuda.get_device_capability()
    major = cap[0]
    # sm_100/103 (Blackwell datacenter) and sm_120 (Blackwell consumer) both
    # carry 5th-gen FP4 tensor cores. Gate on major>=10 with the known minors.
    if major < 10:
        return (
            False,
            f"NVFP4 training requires Blackwell FP4 tensor cores "
            f"(compute capability >= 10.0), got {cap[0]}.{cap[1]}",
        )
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import (  # noqa: F401
            NVFP4Tensor,
            _addmm_nvfp4_dispatch,
            per_tensor_amax_to_scale,
        )
    except ImportError as exc:
        return False, f"NVFP4 training requires torchao >= 0.18 ({exc})"
    from packaging import version as _v

    # parse() not string compare: "2.12" < "2.8" lexicographically (wrong for >=2.10)
    if _v.parse(torch.__version__.split("+")[0]) < _v.parse("2.8"):
        return False, "NVFP4 training requires torch >= 2.8"
    return True, ""


@dataclass
class QuantPolicy:
    """How a tensor is quantized for one operand of one GEMM.

    The convergence-recipe agent extends ``stochastic`` (gradient SR) and
    ``hadamard`` (RHT on wgrad inputs); the base path implements round-to-
    nearest with mandatory two-level per-tensor scaling.
    """

    stochastic: bool = False
    hadamard: bool = False


def _build_base_dh() -> torch.Tensor:
    """Orthonormal 16x16 Hadamard (H_16/4) times a fixed random ±1 diagonal.

    (D H)^T (D H) = H^T D^T D H = H^T H = 16 I, so dividing by sqrt(16)=4 gives
    an orthonormal rotation: applied to the contraction dim of both wgrad
    operands it cancels, while the FP4 quant in between sees Gaussian-ized values.
    """
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < _HAD_DIM:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / (_HAD_DIM**0.5)
    gen = torch.Generator().manual_seed(0)
    sign = torch.randint(0, 2, (_HAD_DIM,), generator=gen).to(torch.float64) * 2 - 1
    return sign.unsqueeze(1) * h  # D @ H, orthonormal


# Built once at import (the Generator call must stay out of the traced region —
# a torch.Generator() inside the backward graph forces a dynamo break under
# fullgraph). The hot path only casts this constant to device/dtype, which is
# fully traceable.
_BASE_DH = _build_base_dh()


def _apply_rht(t: torch.Tensor) -> torch.Tensor:
    """Block-Hadamard rotation along the last (contraction) dim of ``t``.

    Last dim is the contraction axis for both wgrad operands as fed to
    ``_quantize`` (``gt`` directly, ``xp`` via ``b.t()``), so the same rotation
    on both cancels the product.
    """
    dh = _BASE_DH.to(device=t.device, dtype=t.dtype)
    lead = t.shape[:-1]
    blocks = t.shape[-1] // _HAD_DIM
    return (t.reshape(*lead, blocks, _HAD_DIM) @ dh.t()).reshape(*lead, t.shape[-1])


def _sr_dither(t: torch.Tensor, per_tensor_scale: torch.Tensor) -> torch.Tensor:
    """Uniform dither over one FP4 step so the subsequent RTN realizes SR.

    Adding uniform noise of width = one quantization step before RTN is exactly
    unbiased stochastic rounding. The step is computed in the original domain by
    mirroring torchao's two-level scaling: rotate to the per-block-scaled value
    ``v``, take the FP4 half-step ``2^floor(log2|v|)`` (mantissa is 1 bit, levels
    are 2^e*{1,1.5}; clamp e to the E2M1 normal range), then de-scale.
    """
    from torchao.prototype.mx_formats.constants import (
        F4_E2M1_MAX,
        F8E4M3_MAX,
    )

    eps = torch.finfo(torch.float8_e4m3fn).tiny
    x = t.float().reshape(t.shape[0], -1, _BLOCK_SIZE)
    block_scale = torch.amax(torch.abs(x), dim=-1) / F4_E2M1_MAX
    scaled_block = block_scale / per_tensor_scale
    sbf8 = (
        torch.clamp(scaled_block, min=eps, max=F8E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    recip = (1.0 / per_tensor_scale) / sbf8  # original -> fp4-grid multiplier
    v = x * recip.unsqueeze(-1)
    e = torch.floor(torch.log2(v.abs().clamp(min=eps))).clamp(_FP4_EXP_LO, _FP4_EXP_HI)
    step_orig = (2.0**e) / recip.unsqueeze(-1)  # one FP4 step, original domain
    u = (torch.rand_like(v) - 0.5) * step_orig
    return (x + u).reshape(t.shape).to(t.dtype)


# Row-chunk size for the load-time quant of large frozen weights. torchao's
# to_nvfp4 upcasts the WHOLE tensor to f32 at once (~2x the bf16 size of scratch),
# which OOMs lm_head/embedding-sized weights on a near-full card. Quantizing in
# row blocks (with a globally-fixed per-tensor scale so the result is bit-
# identical) bounds that scratch to one block. 128 = torchao's swizzle row tile,
# so block-row qdata/scale concatenate exactly to the whole-tensor layout.
_QUANT_CHUNK_ROWS = 8192
_QUANT_CHUNK_MIN_ROWS = 2 * _QUANT_CHUNK_ROWS  # below this the scratch is small


def _to_nvfp4_chunked(t, per_tensor_scale, act_quant_kwargs):
    """``NVFP4Tensor.to_nvfp4`` over a 2D weight, row-block by row-block.

    The f32 quant scratch is bounded to ``_QUANT_CHUNK_ROWS`` rows instead of the
    whole tensor. Bit-identical to the single-shot quant: NVFP4 blocks lie along
    the last dim (rows are independent) and the per-tensor scale is fixed across
    blocks, so concatenating the per-block qdata/scale reproduces the full layout
    (block rows align to the 128-row swizzle tile). Falls back to a single call
    for small tensors / non-2D inputs.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    def _one(x):
        return NVFP4Tensor.to_nvfp4(
            x.contiguous(),
            block_size=_BLOCK_SIZE,
            per_tensor_scale=per_tensor_scale,
            act_quant_kwargs=act_quant_kwargs,
        )

    if t.dim() != 2 or t.shape[0] < _QUANT_CHUNK_MIN_ROWS:
        return _one(t)

    qparts, sparts, ctx = [], [], None
    for i in range(0, t.shape[0], _QUANT_CHUNK_ROWS):
        c = _one(t[i : i + _QUANT_CHUNK_ROWS])
        if ctx is None:
            ctx = c.__tensor_flatten__()[1]
        qparts.append(c.qdata)
        sparts.append(c.scale)
        del c
    inner = {
        "qdata": torch.cat(qparts, dim=0),
        "scale": torch.cat(sparts, dim=0),
        "per_tensor_scale": per_tensor_scale,
    }
    return NVFP4Tensor.__tensor_unflatten__(inner, ctx, None, None)


def _abs_amax(t: torch.Tensor) -> torch.Tensor:
    """Global max(|t|) as fp32 in ONE fused reduction (no materialized ``|t|``).

    Bit-identical to ``torch.amax(torch.abs(t))`` but the inf-norm reduction folds
    the abs into the reduce kernel, dropping the separate ``AbsFunctor`` elementwise
    pass that dominated the NVFP4 quant prologue.
    """
    return torch.linalg.vector_norm(t, ord=float("inf")).to(torch.float32)


def _quantize(t: torch.Tensor, policy: QuantPolicy):
    """Quantize a high-precision tensor to an NVFP4Tensor (along its last dim).

    Two-level scaling (per-tensor fp32 scale + per-block e4m3 scale) is always
    applied: without it small-magnitude tensors (gradients) underflow to zero.
    ``policy.hadamard`` rotates the contraction dim (RHT, wgrad inputs);
    ``policy.stochastic`` dithers for stochastic rounding (gradient operands).
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    t = t.contiguous()
    if (
        (policy.hadamard or policy.stochastic)
        and t.is_cuda
        and t.dim() == 2
        and t.shape[-1] % _BLOCK_SIZE == 0
        and _recipe_fusion_available(t)
    ):
        q, s, inv_gs = _mslk_quantize_recipe(t, policy)
        return NVFP4Tensor(
            q,
            s,
            _BLOCK_SIZE,
            t.dtype,
            per_tensor_scale=inv_gs.to(torch.float32),
            is_swizzled_scales=True,
        )
    if policy.hadamard:
        t = _apply_rht(t).contiguous()
    # Clamp amax like the mslk fast path (_mslk_quantize_recipe_op): an all-zero
    # tile (e.g. a frozen-base dgrad with no contribution for a packed micro-batch
    # under FSDP) gives per_tensor_scale=0, and the SR dither's 1/per_tensor_scale
    # then makes 0*inf=NaN. A positive floor keeps it 0*finite=0.
    per_tensor_scale = per_tensor_amax_to_scale(_abs_amax(t).clamp(min=1e-12))
    if policy.stochastic:
        t = _sr_dither(t, per_tensor_scale).contiguous()
    # RHT/SR rewrite the whole tensor up front (no per-block-row independence), so
    # only the plain frozen-weight quant takes the memory-bounded chunked path.
    if policy.hadamard or policy.stochastic:
        return NVFP4Tensor.to_nvfp4(
            t, block_size=_BLOCK_SIZE, per_tensor_scale=per_tensor_scale
        )
    return _to_nvfp4_chunked(t, per_tensor_scale, None)


def _fp4_mm(a_hp: torch.Tensor, b_hp: torch.Tensor, a_pol, b_pol) -> torch.Tensor:
    """C[M,N] = a_hp[M,K] @ b_hp[K,N] with both operands quantized to NVFP4.

    ``_scaled_mm`` wants TN layout: ``a`` row-major, ``b`` such that
    ``b.qdata.t()`` is contiguous. The latter is produced by quantizing the
    [N,K] form (contiguous, along K) and transposing.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    a_q = _quantize(a_hp, a_pol)
    b_q = _quantize(b_hp.t().contiguous(), b_pol).t()
    return _addmm_nvfp4_dispatch(a_q, b_q, torch.ops.aten.mm.default)


def _pad_to_block(t: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    """Zero-pad ``dim`` up to a multiple of 32; return (padded, original_size).

    The token dimension (M) is the wgrad contraction axis and is rarely a
    multiple of 32. Zero rows contribute zero to the wgrad accumulation and are
    sliced back off the forward output / dgrad.
    """
    n = t.shape[dim]
    rem = n % _GEMM_ALIGN
    if rem == 0:
        return t, n
    pad = _GEMM_ALIGN - rem
    pad_spec = [0, 0] * (t.dim() - dim - 1) + [0, pad]
    return torch.nn.functional.pad(t, pad_spec), n


class NVFP4LinearFunction(torch.autograd.Function):
    """Linear with FP4 GEMMs in forward (fprop) and backward (dgrad + wgrad).

    Master weight stays high-precision (the differentiable ``weight``); only GEMM
    operands are quantized. The weight changes only once per optimizer step, so
    its two FP4 b-operand layouts (``w_fprop``, ``w_dgrad``) are pre-quantized by
    the caller (:class:`NVFP4Linear`) and passed as non-differentiable side
    inputs — wgrad still flows to the master ``weight`` (the RTN-quantized
    layouts carry no gradient). Per the NVFP4 recipe, gradient operands get
    stochastic rounding and wgrad inputs get RHT (via ``QuantPolicy``).
    """

    @staticmethod
    def forward(ctx, x, weight, w_fprop, w_dgrad, bias, recipe):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        # x:[*, K]  weight:[N, K]  w_fprop: pre-quantized W.T b-operand [K,N].
        # w_fprop/w_dgrad are None under torch.compile (the version cache is a
        # data-dependent Python branch that can't be traced) — then quantize the
        # weight inline exactly as the pre-cache path did, bit-identical.
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)

        # fprop: x[M,K] @ W.T[K,N]. w_fprop == _quantize(W, QuantPolicy()).t(),
        # bit-identical to _fp4_mm(x, weight.t(), ..)'s internal weight quant.
        if w_fprop is None:
            w_fprop = _quantize(weight.contiguous(), QuantPolicy()).t()
        a_q = _quantize(x2d_p, QuantPolicy())
        out = _addmm_nvfp4_dispatch(a_q, w_fprop, torch.ops.aten.mm.default)[:m]
        if bias is not None:
            out = out + bias

        ctx.save_for_backward(x2d, weight)
        ctx.w_dgrad = w_dgrad
        ctx.recipe = recipe
        ctx.has_bias = bias is not None
        ctx.x_shape = orig_shape
        return out.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        x2d, weight = ctx.saved_tensors  # x2d:[M,K] weight:[N,K]
        w_dgrad = ctx.w_dgrad
        recipe = ctx.recipe
        g = grad_out.reshape(-1, weight.shape[0])  # [M, N]

        grad_x = grad_w = grad_bias = None

        # gradient operands use stochastic rounding; wgrad inputs use RHT
        g_pol = QuantPolicy(stochastic=recipe.stochastic_rounding)
        rht_pol = QuantPolicy(
            stochastic=recipe.stochastic_rounding, hadamard=recipe.hadamard
        )

        if ctx.needs_input_grad[0]:
            # dgrad: grad_x[M,K] = g[M,N] @ weight[N,K]   (contraction N).
            # w_dgrad == _quantize(W.t().contiguous(), QuantPolicy()).t(),
            # bit-identical to _fp4_mm(g, weight, ..)'s internal weight quant.
            g_p, m = _pad_to_block(g, 0)
            if w_dgrad is None:
                w_dgrad = _quantize(weight.t().contiguous(), QuantPolicy()).t()
            g_q = _quantize(g_p, g_pol)
            grad_x = _addmm_nvfp4_dispatch(g_q, w_dgrad, torch.ops.aten.mm.default)[:m]
            grad_x = grad_x.reshape(ctx.x_shape)

        if ctx.needs_input_grad[1]:
            # wgrad: grad_w[N,K] = g.t()[N,M] @ x[M,K]    (contraction M, RHT).
            # Unchanged — uses only g and the saved x, never the weight value, so
            # the master-weight gradient is bit-identical to the requant path.
            gt, _ = _pad_to_block(g.t().contiguous(), 1)  # [N, M_pad]
            xp, _ = _pad_to_block(x2d, 0)  # [M_pad, K]
            grad_w = _fp4_mm(gt, xp, rht_pol, rht_pol)

        # input order: (x, weight, w_fprop, w_dgrad, bias, recipe) -> bias is [4]
        if ctx.has_bias and ctx.needs_input_grad[4]:
            grad_bias = g.sum(dim=0)

        return grad_x, grad_w, None, None, grad_bias, None


@dataclass
class NVFP4Recipe:
    """Training-precision recipe (distinct from the QAT/PTQ quantization block)."""

    stochastic_rounding: bool = True
    hadamard: bool = True


class NVFP4Linear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` whose GEMMs run in NVFP4.

    Weight/bias remain ``nn.Parameter`` in their original dtype; quantization
    happens inside each GEMM. Requires ``in_features`` and ``out_features``
    divisible by 16 — callers must exclude layers that violate this (lm_head,
    odd-vocab embeddings) from the swap.
    """

    def __init__(self, weight, bias, recipe: NVFP4Recipe):
        super().__init__()
        self.weight = weight
        self.bias = bias
        self.recipe = recipe
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        # Per-step cache of the two FP4 b-operand layouts, invalidated when the
        # optimizer mutates the master weight (bumps weight._version). The weight
        # quant is deterministic RTN, so caching is bit-exact, not approximate.
        self._wq_version = None
        self._wq_fprop = None
        self._wq_dgrad = None

    def _quantized_weights(self):
        """(w_fprop, w_dgrad) for the current weight, recomputed only on update.

        Mirrors NVFP4ComputeBaseLinear.from_linear's two layouts but from the
        live master weight, refreshed when weight._version changes (one optimizer
        step). detach() so no autograd tracks through the cached FP4 operands —
        the differentiable path to the master weight is the wgrad in backward.
        """
        version = self.weight._version
        if self._wq_version != version:
            w = self.weight.detach()
            # fprop b-operand represents W.T ([K,N]): quantize W then transpose.
            self._wq_fprop = _quantize(w, QuantPolicy()).t()
            # dgrad b-operand represents W ([N,K]) blocked along N: quantize W.T.
            self._wq_dgrad = _quantize(w.t().contiguous(), QuantPolicy()).t()
            self._wq_version = version
        return self._wq_fprop, self._wq_dgrad

    def forward(self, x):
        # Under torch.compile the cache lookup (a data-dependent branch on
        # weight._version) can't be traced; pass None so the autograd Function
        # quantizes the weight inline, which compiles with zero graph breaks and
        # is what the graph already fuses.
        if torch.compiler.is_compiling():
            w_fprop = w_dgrad = None
        else:
            w_fprop, w_dgrad = self._quantized_weights()
        return NVFP4LinearFunction.apply(
            x, self.weight, w_fprop, w_dgrad, self.bias, self.recipe
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, recipe: NVFP4Recipe) -> "NVFP4Linear":
        return cls(linear.weight, linear.bias, recipe)


def _clone_nvfp4_data(w_q):
    """Plain NVFP4Tensor with cloned packed storage, independent of FSDP's
    all-gather buffer, for the frozen-base backward dgrad."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    pts = w_q.per_tensor_scale
    inner = {
        "qdata": w_q.qdata.clone(),
        "scale": w_q.scale.clone(),
        "per_tensor_scale": None if pts is None else pts.clone(),
    }
    return NVFP4Tensor.__tensor_unflatten__(
        inner, w_q.__tensor_flatten__()[1], None, None
    )


class NVFP4FrozenBaseFunction(torch.autograd.Function):
    """Forward GEMM against a pre-quantized FROZEN weight; dgrad only.

    For the NVFP4-QLoRA path: the base weight is stored packed in FP4 (no
    high-precision master copy → ~3.5x weight memory savings) and is frozen, so
    there is no weight gradient — only dgrad to propagate into the trainable
    LoRA adapters and earlier layers.
    """

    @staticmethod
    def forward(ctx, x, w_q, recipe):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)
        # w_q is the stored FP4 weight ([N,K], blocked along K); w_q.t() is the
        # [K,N] B operand. Route through _addmm (NOT F.linear): torchao's
        # dynamic-act F.linear path can't carry the two-level per-tensor scale
        # (it asserts per_tensor_scale is None on both operands).
        out = _addmm_nvfp4_dispatch(
            _quantize(x2d_p, QuantPolicy()), w_q.t(), torch.ops.aten.mm.default
        )[:m]
        # FSDP2 frees the all-gathered weight storage after forward and does not
        # re-gather the FROZEN base for backward, so the saved reference would
        # dequantize freed memory (-> NaN grads). Under FSDP, snapshot the packed
        # FP4 data (small; ~3.5x under bf16) into independent storage; the plain
        # single-GPU path keeps the zero-copy reference.
        if hasattr(w_q, "fsdp_pre_all_gather"):
            ctx.w_q = _clone_nvfp4_data(w_q)
        else:
            ctx.w_q = w_q
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        return out.reshape(*orig_shape[:-1], w_q.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        w_q = ctx.w_q
        grad_x = None
        if ctx.needs_input_grad[0]:
            g = grad_out.reshape(-1, w_q.shape[0])
            g_p, m = _pad_to_block(g, 0)
            # dgrad = g @ W; dequant the stored weight for the contraction-along-N
            # GEMM (the stored layout is quantized along K for the forward).
            w_hp = w_q.dequantize(torch.bfloat16)
            grad_x = _fp4_mm(
                g_p,
                w_hp,
                QuantPolicy(stochastic=ctx.recipe.stochastic_rounding),
                QuantPolicy(),
            )[:m]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None


_FSDP_NVFP4_CLS = None


def _fsdp_nvfp4_class():
    """torchao's NVFP4Tensor subclassed with FSDP2 all-gather hooks (cached).

    Built lazily so importing this module never needs the torchao.prototype
    NVFP4Tensor symbol at top level.
    """
    global _FSDP_NVFP4_CLS
    if _FSDP_NVFP4_CLS is not None:
        return _FSDP_NVFP4_CLS
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    aten = torch.ops.aten

    class FSDPNVFP4Tensor(NVFP4Tensor):
        # The frozen FP4 base shards along dim 0: qdata and scale both split by
        # row; the global per_tensor_scale (computed once over the whole weight,
        # identical on every rank) is replicated. Reconstruction concatenates the
        # row-shards — verified bit-exact against the unsharded tensor.
        def fsdp_pre_all_gather(self, mesh):
            return (self.qdata, self.scale), (
                self.__tensor_flatten__()[1],
                self.per_tensor_scale,
            )

        def fsdp_post_all_gather(
            self, all_gather_outputs, metadata, param_dtype, *, out=None
        ):
            qdata, scale = all_gather_outputs
            ctx, per_tensor_scale = metadata
            if out is not None:
                return
            inner = {
                "qdata": qdata,
                "scale": scale,
                "per_tensor_scale": per_tensor_scale,
            }
            rebuilt = type(self).__tensor_unflatten__(inner, ctx, None, None)
            return rebuilt, (qdata, scale)

        @classmethod
        def _rewrap(cls, t):
            # torchao's NVFP4 ops hardcode the NVFP4Tensor type for their outputs,
            # dropping this subclass (and its all-gather hooks). FSDP2 sharding
            # slices/copies the param, so re-assert the subclass on any NVFP4Tensor
            # result to keep the hooks alive on the sharded parameter.
            if type(t) is NVFP4Tensor:
                return cls(
                    t.qdata,
                    t.scale,
                    t.block_size,
                    t.orig_dtype,
                    t.per_tensor_scale,
                    t.act_per_tensor_scale,
                    t.is_swizzled_scales,
                    t.use_triton_kernel,
                    t.act_quant_kwargs,
                )
            return t

        @classmethod
        def __torch_dispatch__(cls, func, types, args, kwargs=None):
            kwargs = kwargs or {}

            # cpu_ram_efficient_loading broadcasts rank-0 params at init. The base
            # NVFP4Tensor op table has no c10d.broadcast_, so broadcast each inner
            # component (qdata/scale/per_tensor_scale) with the same collective args
            # — the op is in-place, so the wrapper's storage updates. Schema returns
            # (Tensor[], Work); hand back the original wrappers + the last Work for
            # the caller (dist.broadcast, async_op=False) to wait on.
            if func is torch.ops.c10d.broadcast_.default:
                tensors = args[0]
                rest = args[1:]
                last_work = None
                for t in tensors:
                    if isinstance(t, NVFP4Tensor):
                        comps = (t.qdata, t.scale, t.per_tensor_scale)
                    else:
                        comps = (t,)
                    for inner in comps:
                        if inner is None:
                            continue
                        # Inner qdata/scale are contiguous for both base modes
                        # (storage and compute store non-transposed NVFP4Tensors),
                        # so the broadcast is a plain contiguous collective.
                        _out, last_work = torch.ops.c10d.broadcast_.default(
                            [inner], *rest
                        )
                return tensors, last_work

            # FSDP2 allocates the sharded param storage with empty_like, then
            # populates it with copy_ — neither is in torchao's op table.
            if func is aten.empty_like.default:
                src = args[0]
                dev = kwargs.get("device", None)

                def _empty(t):
                    return (
                        torch.empty_like(t, device=dev)
                        if dev is not None
                        else torch.empty_like(t)
                    )

                pts = src.per_tensor_scale
                return cls(
                    _empty(src.qdata),
                    _empty(src.scale),
                    src.block_size,
                    src.orig_dtype,
                    None if pts is None else _empty(pts),
                    src.act_per_tensor_scale,
                    src.is_swizzled_scales,
                    src.use_triton_kernel,
                    src.act_quant_kwargs,
                )
            if func is aten.copy_.default:
                dst, src = args[0], args[1]
                dst.qdata.copy_(src.qdata)
                dst.scale.copy_(src.scale)
                if (
                    dst.per_tensor_scale is not None
                    and getattr(src, "per_tensor_scale", None) is not None
                ):
                    dst.per_tensor_scale.copy_(src.per_tensor_scale)
                return dst

            out = super().__torch_dispatch__(func, types, args, kwargs)
            if isinstance(out, NVFP4Tensor):
                return cls._rewrap(out)
            if isinstance(out, (tuple, list)):
                rewrapped = [
                    cls._rewrap(o) if isinstance(o, NVFP4Tensor) else o for o in out
                ]
                return type(out)(rewrapped)
            return out

    # The class is built lazily inside this function so the module never imports
    # torchao's NVFP4Tensor at top level — but a ``<locals>`` class is not
    # picklable, and FSDP2's FULL_STATE_DICT save pickles the frozen NVFP4 params.
    # Expose it at module scope with a stable qualname so pickle can round-trip it.
    FSDPNVFP4Tensor.__module__ = __name__
    FSDPNVFP4Tensor.__qualname__ = "FSDPNVFP4Tensor"
    globals()["FSDPNVFP4Tensor"] = FSDPNVFP4Tensor

    _FSDP_NVFP4_CLS = FSDPNVFP4Tensor
    return _FSDP_NVFP4_CLS


def _to_fsdp_nvfp4(w_q):
    """Re-wrap an NVFP4Tensor as the FSDP-hooked subclass (same inner data)."""
    sub = _fsdp_nvfp4_class()
    ctx = w_q.__tensor_flatten__()[1]
    inner = {
        "qdata": w_q.qdata,
        "scale": w_q.scale,
        "per_tensor_scale": w_q.per_tensor_scale,
    }
    return sub.__tensor_unflatten__(inner, ctx, None, None)


class NVFP4FrozenBaseLinear(nn.Module):
    """Frozen base linear whose weight is stored packed in FP4 (QLoRA base).

    Unlike ``NVFP4Linear`` (which keeps a high-precision trainable master weight
    — throughput only), this drops the master copy for ~3.5x weight memory
    savings. It is FROZEN (no weight gradient) and is meant to sit under LoRA
    adapters. Bias, if any, stays high-precision.
    """

    def __init__(self, w_q, bias, recipe: NVFP4Recipe):
        super().__init__()
        # Buffer (not Parameter): frozen/no-grad, but still enters state_dict so
        # the FP4-packed base survives save/load — otherwise resume silently
        # reinitializes the base. NVFP4Tensor round-trips through torch.save.
        self.register_buffer("w_q", w_q)
        self.bias = bias  # already an nn.Parameter from the source linear
        self.recipe = recipe
        self.in_features = w_q.shape[1]
        self.out_features = w_q.shape[0]

    def forward(self, x):
        out = NVFP4FrozenBaseFunction.apply(x, self.w_q, self.recipe)
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] view for PEFT (DoRA weight-norm, delta
        # compute, base_layer.weight reads). Writes to it (in-process merge:
        # base_layer.weight.data += delta) do NOT persist into the FP4 store —
        # NVFP4 LoRA bases must merge via the offline CLI (axolotl merge-lora),
        # which the patch_manager enforces by skipping the FP4 swap under merge.
        return self.w_q.dequantize(torch.bfloat16)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe, *, fsdp: bool = False
    ) -> "NVFP4FrozenBaseLinear":
        from torchao.prototype.mx_formats.nvfp4_tensor import (
            QuantizeTensorToNVFP4Kwargs,
            per_tensor_amax_to_scale,
        )

        w = linear.weight.detach()
        pts = per_tensor_amax_to_scale(_abs_amax(w))
        w_q = _to_nvfp4_chunked(
            w.contiguous(),
            pts,
            QuantizeTensorToNVFP4Kwargs(block_size=_BLOCK_SIZE),
        )
        # FSDP2 needs the all-gather hooks to shard the FP4 base by row.
        if fsdp:
            w_q = _to_fsdp_nvfp4(w_q)
        return cls(w_q, linear.bias, recipe)


def _embedding_to_nvfp4(weight: torch.Tensor):
    """Quantize an [vocab, hidden] embedding weight to a stored NVFP4Tensor.

    Blocked along the hidden dim (last), two-level scaled — same packing as the
    frozen base linear. The result is the SHARED store routed to both the
    embedding lookup and (when tied) the lm_head GEMM.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        QuantizeTensorToNVFP4Kwargs,
        per_tensor_amax_to_scale,
    )

    w = weight.detach().contiguous()
    pts = per_tensor_amax_to_scale(_abs_amax(w))
    return _to_nvfp4_chunked(
        w, pts, QuantizeTensorToNVFP4Kwargs(block_size=_BLOCK_SIZE)
    )


def _nvfp4_embedding_gather(w_q, input):
    """Embedding lookup that dequantizes only the gathered rows of ``w_q``.

    Avoids materializing the full [vocab, hidden] bf16 table (lm_head-sized) just
    to gather a handful of rows. NVFP4 blocks lie along the hidden dim, so each
    vocab row is a self-contained slice of qdata/scale; gathering the rows of the
    packed buffers and dequantizing that subset is bit-identical to dequantizing
    the whole table and gathering. Frozen weight, so padding_idx (gradient-only)
    is a no-op in forward. Returns None (caller falls back) when the layout isn't
    safe to row-slice (swizzled scales) or anything is unexpected.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    # Under torch.compile keep the plain full dequant: the subtensor rebuild and
    # row gather graph-break, and the table dequant is what the compiled graph
    # already fuses. The memory win matters at load/eager, not in the hot graph.
    if torch.compiler.is_compiling():
        return None

    try:
        names, ctx = w_q.__tensor_flatten__()
        if ctx.get("is_swizzled_scales"):
            return None
        flat = input.reshape(-1)
        sub = NVFP4Tensor.__tensor_unflatten__(
            {
                "qdata": w_q.qdata[flat],
                "scale": w_q.scale[flat],
                "per_tensor_scale": w_q.per_tensor_scale,
            },
            ctx,
            None,
            None,
        )
        rows = sub.dequantize(torch.bfloat16)
        return rows.reshape(*input.shape, rows.shape[-1])
    except Exception:  # any layout/version surprise -> full-dequant fallback
        return None


class NVFP4Embedding(nn.Module):
    """Input embedding whose weight is stored packed in FP4 (W4A16 lookup).

    The lookup gathers rows by integer index — no activation quant — so forward
    is ``F.embedding`` over the dequantized weight. FROZEN only: an FP4-stored
    weight has no high-precision master to receive gradients (use QAT for a
    trainable FP4 embedding). Hidden dim must be divisible by 16.
    """

    def __init__(self, w_q, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        # Buffer (not Parameter): frozen, but must enter the state_dict so the
        # FP4-packed embedding survives save/load (NVFP4Tensor round-trips).
        self.register_buffer("w_q", w_q)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

    def forward(self, input):
        # Dequantize ONLY the gathered rows, not the whole [vocab, hidden] table:
        # the full bf16 dequant (+ its f32 scratch) is lm_head-sized and OOMs on a
        # near-full card. NVFP4 blocks lie along the hidden dim so each row is
        # self-contained — slicing qdata/scale by the unique looked-up rows and
        # dequantizing that subset is bit-identical to gathering after a full
        # dequant. Falls back to the full path if the scales are swizzled (row
        # slicing would break the tile) or anything is unexpected.
        gathered = _nvfp4_embedding_gather(self.w_q, input)
        if gathered is not None:
            return gathered
        w = self.w_q.dequantize(torch.bfloat16)
        return torch.nn.functional.embedding(input, w, self.padding_idx)

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [vocab, hidden] view; writes don't persist.
        return self.w_q.dequantize(torch.bfloat16)

    @classmethod
    def from_embedding(cls, emb: nn.Embedding) -> "NVFP4Embedding":
        return cls.from_weight(
            emb.weight, emb.num_embeddings, emb.embedding_dim, emb.padding_idx
        )

    @classmethod
    def from_weight(
        cls, weight, num_embeddings, embedding_dim, padding_idx=None
    ) -> "NVFP4Embedding":
        return cls(
            _embedding_to_nvfp4(weight), num_embeddings, embedding_dim, padding_idx
        )


class NVFP4TiedLMHead(nn.Module):
    """lm_head GEMM over a SHARED FP4 store (tied-embedding case).

    Holds no weight of its own: it reads the SAME ``NVFP4Tensor`` that backs the
    input :class:`NVFP4Embedding`, so the dequantized weight is bit-identical for
    the lookup and the GEMM. Frozen (no wgrad); dgrad flows via the storage-mode
    frozen-base function.
    """

    def __init__(self, embedding: NVFP4Embedding, bias, recipe: NVFP4Recipe):
        super().__init__()
        # Reference the embedding's buffer directly so the two roles always read
        # the same FP4 store (the embedding owns it in the state_dict).
        self._embedding = embedding
        self.bias = bias
        self.recipe = recipe
        self.in_features = embedding.w_q.shape[1]
        self.out_features = embedding.w_q.shape[0]

    @property
    def w_q(self):
        return self._embedding.w_q

    def forward(self, x):
        out = NVFP4FrozenBaseFunction.apply(x, self.w_q, self.recipe)
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        return self.w_q.dequantize(torch.bfloat16)


class NVFP4ComputeBaseFunction(torch.autograd.Function):
    """Frozen-base linear with FP4 compute on BOTH base GEMMs, zero per-step
    base-quant prologue.

    The base is frozen, so its NVFP4 operands are quantized ONCE at load (not
    per step like FFT). fprop needs the base blocked along K (contraction) and
    dgrad needs it blocked along N (contraction), so two pre-quantized layouts
    are stored. Each is a real ``torch._scaled_mm`` FP4 GEMM; only the activation
    / incoming-gradient operand is quantized per step. No wgrad (frozen).
    """

    @staticmethod
    def forward(ctx, x, w_fprop, w_dgrad, recipe):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d_p, m = _pad_to_block(x2d, 0)
        # w_fprop is stored CONTIGUOUS ([N,K], blocked along K); .t() gives the
        # [K,N] b-operand (addmm wants b.qdata.t() contiguous). Transposing at
        # GEMM time (not at store) keeps the buffer's inner qdata/scale contiguous
        # so FSDP2 rank-0 broadcast (cpu_ram_efficient_loading) can broadcast them.
        out = _addmm_nvfp4_dispatch(
            _quantize(x2d_p, QuantPolicy()), w_fprop.t(), torch.ops.aten.mm.default
        )[:m]
        # FSDP2 frees the all-gathered weight after forward and does not re-gather
        # the FROZEN base for backward, so a saved reference would read freed
        # storage (-> NaN dgrad). Under FSDP, snapshot the packed FP4 dgrad layout
        # into independent storage; the plain path keeps the zero-copy reference.
        ctx.w_dgrad = (
            _clone_nvfp4_data(w_dgrad)
            if hasattr(w_dgrad, "fsdp_pre_all_gather")
            else w_dgrad
        )
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        ctx.out_features = w_fprop.shape[0]
        return out.reshape(*orig_shape[:-1], ctx.out_features)

    @staticmethod
    def backward(ctx, grad_out):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        grad_x = None
        if ctx.needs_input_grad[0]:
            g = grad_out.reshape(-1, ctx.out_features)
            g_p, m = _pad_to_block(g, 0)
            # dgrad: gx[M,K] = g[M,N] @ W[N,K]; W pre-quantized along N (contraction)
            g_q = _quantize(g_p, QuantPolicy(stochastic=ctx.recipe.stochastic_rounding))
            # w_dgrad is stored CONTIGUOUS ([K,N], blocked along N); .t() gives the
            # [N,K] b-operand (W blocked along N, the dgrad contraction axis).
            grad_x = _addmm_nvfp4_dispatch(
                g_q, ctx.w_dgrad.t(), torch.ops.aten.mm.default
            )[:m]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None


class NVFP4ComputeBaseLinear(nn.Module):
    """Frozen LoRA base with FP4 compute on fprop+dgrad and no per-step base quant.

    Stores two pre-quantized NVFP4 layouts of the frozen weight (fprop: blocked
    along K; dgrad: blocked along N) as buffers. ~1.75x weight memory vs bf16
    (two FP4 copies + scales) but the base GEMMs run pure FP4 with the quant
    prologue paid once at load — faster than re-quantizing the base every step.
    Adapters (added by PEFT around this base_layer) stay high-precision.
    """

    def __init__(self, w_fprop, w_dgrad, bias, recipe: NVFP4Recipe):
        super().__init__()
        # Both buffers are stored CONTIGUOUS (non-transposed) NVFP4Tensors:
        # w_fprop is [N,K] (blocked along K), w_dgrad is [K,N] (blocked along N).
        # The forward/backward apply .t() at GEMM time. Contiguous inner storage
        # is what lets FSDP2 rank-0 broadcast (cpu_ram_efficient_loading) work.
        self.register_buffer("w_fprop", w_fprop)
        self.register_buffer("w_dgrad", w_dgrad)
        self.bias = bias
        self.recipe = recipe
        self.in_features = w_fprop.shape[1]
        self.out_features = w_fprop.shape[0]

    def forward(self, x):
        out = NVFP4ComputeBaseFunction.apply(x, self.w_fprop, self.w_dgrad, self.recipe)
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight). w_fprop is the contiguous _quantize(W)
        # ([N,K]), so dequantize() directly recovers [N,K].
        return self.w_fprop.dequantize(torch.bfloat16)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe, *, fsdp: bool = False
    ) -> "NVFP4ComputeBaseLinear":
        w = linear.weight.detach()  # [N, K]
        # Store CONTIGUOUS (non-transposed) NVFP4 layouts; forward/backward apply
        # .t() at GEMM time. This keeps each buffer's inner qdata/scale contiguous
        # (vs the old .t() views), which is required for FSDP2 rank-0 broadcast
        # (cpu_ram_efficient_loading). The logical b-operands at the GEMM are
        # bit-identical to the old stored .t() views.
        # fprop: stored W ([N,K]) blocked along K; GEMM uses w_fprop.t() ([K,N]).
        w_fprop = _quantize(w, QuantPolicy())
        # dgrad: stored W.T ([K,N]) blocked along N; GEMM uses w_dgrad.t() ([N,K]).
        w_dgrad = _quantize(w.t().contiguous(), QuantPolicy())
        # FSDP2 shards each FP4 layout by row; the all-gather hooks live on the
        # FSDP subclass. Wrap both buffers independently (they shard on different
        # axes — fprop rows are N, dgrad rows are K — but each reassembles on its own).
        if fsdp:
            w_fprop = _to_fsdp_nvfp4(w_fprop)
            w_dgrad = _to_fsdp_nvfp4(w_dgrad)
        return cls(w_fprop, w_dgrad, linear.bias, recipe)


# NVFP4 two-level global scale target: map a tensor's amax onto the product of
# the FP4 and FP8 maxima so the per-block e4m3 scales use their full range.
_NVFP4_GLOBAL_AMAX = 448.0 * 6.0  # F8E4M3_MAX * F4_E2M1_MAX
_MSLK_AVAILABLE: bool | None = None


def _mslk_available() -> bool:
    """Whether the MSLK fused NVFP4 quant kernel is importable (cached).

    MSLK lives only in the TE/perf venv, not the base experimental venv, so the
    fast path is strictly optional — callers fall back to the torchao quantizer.
    """
    global _MSLK_AVAILABLE
    import os

    if os.environ.get("AXOLOTL_NVFP4_NO_MSLK") == "1":
        return False
    if _MSLK_AVAILABLE is None:
        try:
            from mslk.quantize.triton.fp4_quantize import (  # noqa: F401
                triton_quantize_nvfp4,
            )

            _MSLK_AVAILABLE = True
        except Exception:  # ImportError or a triton/runtime probe failure
            _MSLK_AVAILABLE = False
    return _MSLK_AVAILABLE


def _recipe_fusion_available(t: torch.Tensor) -> bool:
    if (
        not t.is_cuda
        or t.dim() != 2
        or t.shape[-1] % _BLOCK_SIZE != 0
        or not _mslk_available()
    ):
        return False
    cap = torch.cuda.get_device_capability(t.device)
    return cap[0] == 10


def _swizzled_scale_shape(m: int, k: int) -> tuple[int, int]:
    """Shape of MSLK's swizzled e4m3 block-scale tensor for a [m, k] input.

    Rows pad to 128, columns (= k/16 block scales) pad to 4 — the tcgen05 MMA
    scale-factor tile. Needed by the custom-op fake impls (the scale tensor's
    size can't be derived from the qdata under tracing).
    """
    rounded_m = (m + 127) // 128 * 128
    n_blocks = k // _BLOCK_SIZE
    rounded_k = (n_blocks + 3) // 4 * 4
    return rounded_m, rounded_k


@triton.jit
def _recipe_scale_swizzle(offs_m):
    sub_layout_idx = offs_m % 32
    sub_layout_off = sub_layout_idx * 16
    sub_layout_row = offs_m // 32
    elems = tl.arange(0, 4)[None, :]
    return sub_layout_off + sub_layout_row * 4 + elems


@triton.jit
def _recipe_fp32_to_fp4_packed(x_pairs):
    x_fp4x2 = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=x_pairs,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
    return x_fp4x2


@triton.jit
def _recipe_load_lane(
    x_ptr,
    offs_m,
    group,
    pid_n,
    stride_xm,
    stride_xn,
    M,
    N,
    lane: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    offs_n = pid_n * 64 + group * 16 + lane
    if USE_MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None
    return tl.load(
        x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=other
    ).to(tl.float32)


@triton.jit
def _recipe_hadamard16(
    x0,
    x1,
    x2,
    x3,
    x4,
    x5,
    x6,
    x7,
    x8,
    x9,
    x10,
    x11,
    x12,
    x13,
    x14,
    x15,
):
    a0 = x0 + x1
    a1 = x0 - x1
    a2 = x2 + x3
    a3 = x2 - x3
    a4 = x4 + x5
    a5 = x4 - x5
    a6 = x6 + x7
    a7 = x6 - x7
    a8 = x8 + x9
    a9 = x8 - x9
    a10 = x10 + x11
    a11 = x10 - x11
    a12 = x12 + x13
    a13 = x12 - x13
    a14 = x14 + x15
    a15 = x14 - x15

    b0 = a0 + a2
    b1 = a1 + a3
    b2 = a0 - a2
    b3 = a1 - a3
    b4 = a4 + a6
    b5 = a5 + a7
    b6 = a4 - a6
    b7 = a5 - a7
    b8 = a8 + a10
    b9 = a9 + a11
    b10 = a8 - a10
    b11 = a9 - a11
    b12 = a12 + a14
    b13 = a13 + a15
    b14 = a12 - a14
    b15 = a13 - a15

    c0 = b0 + b4
    c1 = b1 + b5
    c2 = b2 + b6
    c3 = b3 + b7
    c4 = b0 - b4
    c5 = b1 - b5
    c6 = b2 - b6
    c7 = b3 - b7
    c8 = b8 + b12
    c9 = b9 + b13
    c10 = b10 + b14
    c11 = b11 + b15
    c12 = b8 - b12
    c13 = b9 - b13
    c14 = b10 - b14
    c15 = b11 - b15

    s = 0.25
    y0 = -(c0 + c8) * s
    y1 = (c1 + c9) * s
    y2 = (c2 + c10) * s
    y3 = -(c3 + c11) * s
    y4 = (c4 + c12) * s
    y5 = (c5 + c13) * s
    y6 = (c6 + c14) * s
    y7 = (c7 + c15) * s
    y8 = (c0 - c8) * s
    y9 = (c1 - c9) * s
    y10 = (c2 - c10) * s
    y11 = -(c3 - c11) * s
    y12 = -(c4 - c12) * s
    y13 = (c5 - c13) * s
    y14 = -(c6 - c14) * s
    y15 = -(c7 - c15) * s
    return (
        y0.to(tl.bfloat16).to(tl.float32),
        y1.to(tl.bfloat16).to(tl.float32),
        y2.to(tl.bfloat16).to(tl.float32),
        y3.to(tl.bfloat16).to(tl.float32),
        y4.to(tl.bfloat16).to(tl.float32),
        y5.to(tl.bfloat16).to(tl.float32),
        y6.to(tl.bfloat16).to(tl.float32),
        y7.to(tl.bfloat16).to(tl.float32),
        y8.to(tl.bfloat16).to(tl.float32),
        y9.to(tl.bfloat16).to(tl.float32),
        y10.to(tl.bfloat16).to(tl.float32),
        y11.to(tl.bfloat16).to(tl.float32),
        y12.to(tl.bfloat16).to(tl.float32),
        y13.to(tl.bfloat16).to(tl.float32),
        y14.to(tl.bfloat16).to(tl.float32),
        y15.to(tl.bfloat16).to(tl.float32),
    )


@triton.jit
def _recipe_lane_amax(
    y0,
    y1,
    y2,
    y3,
    y4,
    y5,
    y6,
    y7,
    y8,
    y9,
    y10,
    y11,
    y12,
    y13,
    y14,
    y15,
):
    a = tl.maximum(tl.abs(y0), tl.abs(y1))
    a = tl.maximum(a, tl.abs(y2))
    a = tl.maximum(a, tl.abs(y3))
    a = tl.maximum(a, tl.abs(y4))
    a = tl.maximum(a, tl.abs(y5))
    a = tl.maximum(a, tl.abs(y6))
    a = tl.maximum(a, tl.abs(y7))
    a = tl.maximum(a, tl.abs(y8))
    a = tl.maximum(a, tl.abs(y9))
    a = tl.maximum(a, tl.abs(y10))
    a = tl.maximum(a, tl.abs(y11))
    a = tl.maximum(a, tl.abs(y12))
    a = tl.maximum(a, tl.abs(y13))
    a = tl.maximum(a, tl.abs(y14))
    return tl.maximum(a, tl.abs(y15))


@triton.jit
def _recipe_rht_amax_kernel(
    x_ptr,
    partial_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
    HADAMARD: tl.constexpr,
    USE_INT64_INDEXING: tl.constexpr,
):
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)
    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    group = tl.arange(0, 4)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)

    x0 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 0, USE_MASK
    )
    x1 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 1, USE_MASK
    )
    x2 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 2, USE_MASK
    )
    x3 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 3, USE_MASK
    )
    x4 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 4, USE_MASK
    )
    x5 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 5, USE_MASK
    )
    x6 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 6, USE_MASK
    )
    x7 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 7, USE_MASK
    )
    x8 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 8, USE_MASK
    )
    x9 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 9, USE_MASK
    )
    x10 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 10, USE_MASK
    )
    x11 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 11, USE_MASK
    )
    x12 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 12, USE_MASK
    )
    x13 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 13, USE_MASK
    )
    x14 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 14, USE_MASK
    )
    x15 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 15, USE_MASK
    )

    if HADAMARD:
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15 = (
            _recipe_hadamard16(
                x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
            )
        )
    else:
        y0, y1, y2, y3, y4, y5, y6, y7 = x0, x1, x2, x3, x4, x5, x6, x7
        y8, y9, y10, y11, y12, y13, y14, y15 = x8, x9, x10, x11, x12, x13, x14, x15

    a = _recipe_lane_amax(
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15
    )
    tl.store(partial_ptr + pid_m * tl.num_programs(0) + pid_n, tl.max(a))


@triton.jit
def _recipe_norm_lane(
    y, scales, global_scale, seed, base_off, lane_off, STOCHASTIC: tl.constexpr
):
    yn = y * (global_scale / scales.to(tl.float32))
    if STOCHASTIC:
        ax = tl.abs(yn)
        step = tl.where(ax < 2.0, 1.0, tl.where(ax < 4.0, 2.0, 4.0))
        yn = yn + (tl.rand(seed, base_off + lane_off) - 0.5) * step
    return tl.clamp(yn, -6.0, 6.0)


@triton.jit
def _recipe_quantize_kernel(
    x_ptr,
    global_scale_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    seed,
    M_PER_BLOCK: tl.constexpr,
    USE_MASK: tl.constexpr,
    HADAMARD: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    USE_INT64_INDEXING: tl.constexpr,
):
    E4M3_EPS = 1.5258789e-05
    FP8_E4M3_MAX = 448.0
    FP4_E2M1_MAX = 6.0
    NUM_ELEM_PER_LAYOUT: tl.constexpr = 128 * 4
    NUM_N_BLOCKS = tl.cdiv(N, 64)

    pid_m = tl.program_id(1)
    pid_n = tl.program_id(0)

    if M_PER_BLOCK != 128 and pid_m * M_PER_BLOCK >= M:
        layout_off = pid_n * NUM_ELEM_PER_LAYOUT
        offs_m_zero = tl.arange(0, 128)[:, None]
        scale_offs = layout_off + _recipe_scale_swizzle(offs_m_zero)
        zero_scales = tl.full([128, 4], 0, dtype=tl.float8e4nv)
        oob_mask = (offs_m_zero >= M) & tl.full((4,), True, dtype=tl.int1)[None, :]
        tl.store(s_ptr + scale_offs, zero_scales, mask=oob_mask)
        return

    offs_m = pid_m * M_PER_BLOCK + tl.arange(0, M_PER_BLOCK)[:, None]
    group = tl.arange(0, 4)[None, :]
    if USE_INT64_INDEXING:
        offs_m = offs_m.to(tl.int64)

    x0 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 0, USE_MASK
    )
    x1 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 1, USE_MASK
    )
    x2 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 2, USE_MASK
    )
    x3 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 3, USE_MASK
    )
    x4 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 4, USE_MASK
    )
    x5 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 5, USE_MASK
    )
    x6 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 6, USE_MASK
    )
    x7 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 7, USE_MASK
    )
    x8 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 8, USE_MASK
    )
    x9 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 9, USE_MASK
    )
    x10 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 10, USE_MASK
    )
    x11 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 11, USE_MASK
    )
    x12 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 12, USE_MASK
    )
    x13 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 13, USE_MASK
    )
    x14 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 14, USE_MASK
    )
    x15 = _recipe_load_lane(
        x_ptr, offs_m, group, pid_n, stride_xm, stride_xn, M, N, 15, USE_MASK
    )

    if HADAMARD:
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15 = (
            _recipe_hadamard16(
                x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15
            )
        )
    else:
        y0, y1, y2, y3, y4, y5, y6, y7 = x0, x1, x2, x3, x4, x5, x6, x7
        y8, y9, y10, y11, y12, y13, y14, y15 = x8, x9, x10, x11, x12, x13, x14, x15

    block_amax = _recipe_lane_amax(
        y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14, y15
    )
    global_scale = tl.load(global_scale_ptr)
    scales = tl.clamp(block_amax / FP4_E2M1_MAX * global_scale, E4M3_EPS, FP8_E4M3_MAX)
    scales = scales.to(tl.float8e4nv)

    if USE_MASK:
        scale_offs_n = pid_n * 4 + group
        scale_mask = (offs_m < M) & (scale_offs_n < (N // 16))
        scales = tl.where(scale_mask, scales, 0.0)

    offs_m_in_layout = (pid_m * M_PER_BLOCK % 128) + tl.arange(0, M_PER_BLOCK)[:, None]
    layout_off = (
        (pid_m * M_PER_BLOCK) // 128
    ) * NUM_N_BLOCKS * NUM_ELEM_PER_LAYOUT + pid_n * NUM_ELEM_PER_LAYOUT
    scale_offs = layout_off + _recipe_scale_swizzle(offs_m_in_layout)
    tl.store(s_ptr + scale_offs, scales)

    row_base = offs_m * N
    col_base = pid_n * 64 + group * 16
    n0 = _recipe_norm_lane(
        y0, scales, global_scale, seed, row_base + col_base, 0, STOCHASTIC
    )
    n1 = _recipe_norm_lane(
        y1, scales, global_scale, seed, row_base + col_base, 1, STOCHASTIC
    )
    n2 = _recipe_norm_lane(
        y2, scales, global_scale, seed, row_base + col_base, 2, STOCHASTIC
    )
    n3 = _recipe_norm_lane(
        y3, scales, global_scale, seed, row_base + col_base, 3, STOCHASTIC
    )
    n4 = _recipe_norm_lane(
        y4, scales, global_scale, seed, row_base + col_base, 4, STOCHASTIC
    )
    n5 = _recipe_norm_lane(
        y5, scales, global_scale, seed, row_base + col_base, 5, STOCHASTIC
    )
    n6 = _recipe_norm_lane(
        y6, scales, global_scale, seed, row_base + col_base, 6, STOCHASTIC
    )
    n7 = _recipe_norm_lane(
        y7, scales, global_scale, seed, row_base + col_base, 7, STOCHASTIC
    )
    n8 = _recipe_norm_lane(
        y8, scales, global_scale, seed, row_base + col_base, 8, STOCHASTIC
    )
    n9 = _recipe_norm_lane(
        y9, scales, global_scale, seed, row_base + col_base, 9, STOCHASTIC
    )
    n10 = _recipe_norm_lane(
        y10, scales, global_scale, seed, row_base + col_base, 10, STOCHASTIC
    )
    n11 = _recipe_norm_lane(
        y11, scales, global_scale, seed, row_base + col_base, 11, STOCHASTIC
    )
    n12 = _recipe_norm_lane(
        y12, scales, global_scale, seed, row_base + col_base, 12, STOCHASTIC
    )
    n13 = _recipe_norm_lane(
        y13, scales, global_scale, seed, row_base + col_base, 13, STOCHASTIC
    )
    n14 = _recipe_norm_lane(
        y14, scales, global_scale, seed, row_base + col_base, 14, STOCHASTIC
    )
    n15 = _recipe_norm_lane(
        y15, scales, global_scale, seed, row_base + col_base, 15, STOCHASTIC
    )

    q0 = _recipe_fp32_to_fp4_packed((n0, n1))
    q1 = _recipe_fp32_to_fp4_packed((n2, n3))
    q2 = _recipe_fp32_to_fp4_packed((n4, n5))
    q3 = _recipe_fp32_to_fp4_packed((n6, n7))
    q4 = _recipe_fp32_to_fp4_packed((n8, n9))
    q5 = _recipe_fp32_to_fp4_packed((n10, n11))
    q6 = _recipe_fp32_to_fp4_packed((n12, n13))
    q7 = _recipe_fp32_to_fp4_packed((n14, n15))

    q_col = pid_n * 32 + group * 8
    q_mask_base = offs_m < M
    if USE_INT64_INDEXING:
        q_col = q_col.to(tl.int64)
    if USE_MASK:
        mask0 = q_mask_base & ((q_col + 0) < (N // 2))
        mask1 = q_mask_base & ((q_col + 1) < (N // 2))
        mask2 = q_mask_base & ((q_col + 2) < (N // 2))
        mask3 = q_mask_base & ((q_col + 3) < (N // 2))
        mask4 = q_mask_base & ((q_col + 4) < (N // 2))
        mask5 = q_mask_base & ((q_col + 5) < (N // 2))
        mask6 = q_mask_base & ((q_col + 6) < (N // 2))
        mask7 = q_mask_base & ((q_col + 7) < (N // 2))
    else:
        mask0 = None
        mask1 = None
        mask2 = None
        mask3 = None
        mask4 = None
        mask5 = None
        mask6 = None
        mask7 = None
    q_base = offs_m * (N // 2) + q_col
    tl.store(q_ptr + q_base + 0, q0, mask=mask0)
    tl.store(q_ptr + q_base + 1, q1, mask=mask1)
    tl.store(q_ptr + q_base + 2, q2, mask=mask2)
    tl.store(q_ptr + q_base + 3, q3, mask=mask3)
    tl.store(q_ptr + q_base + 4, q4, mask=mask4)
    tl.store(q_ptr + q_base + 5, q5, mask=mask5)
    tl.store(q_ptr + q_base + 6, q6, mask=mask6)
    tl.store(q_ptr + q_base + 7, q7, mask=mask7)


def _recipe_m_per_block(m: int) -> int:
    return min(triton.next_power_of_2(m), 128)


def _recipe_rht_amax(t: torch.Tensor, hadamard: bool) -> torch.Tensor:
    if not hadamard:
        return _abs_amax(t)
    m, n = t.shape
    m_per_block = _recipe_m_per_block(m)
    grid = (triton.cdiv(n, 64), triton.cdiv(m, m_per_block))
    partial = torch.empty((grid[1], grid[0]), device=t.device, dtype=torch.float32)
    _recipe_rht_amax_kernel[grid](
        t,
        partial,
        t.stride(0),
        t.stride(1),
        m,
        n,
        M_PER_BLOCK=m_per_block,
        USE_MASK=m % m_per_block != 0 or n % 64 != 0,
        HADAMARD=hadamard,
        USE_INT64_INDEXING=m * n > 2**31 - 1,
    )
    return torch.amax(partial).to(torch.float32)


@torch.library.custom_op("axolotl_nvfp4::quantize_two_level_recipe", mutates_args=())
def _mslk_quantize_recipe_op(
    t: torch.Tensor,
    hadamard: bool,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = t.contiguous()
    m, n = t.shape
    amax = _recipe_rht_amax(t, bool(hadamard))
    global_scale = _NVFP4_GLOBAL_AMAX / torch.clamp(amax, min=1e-12)
    q = t.new_empty(m, n // 2, dtype=torch.uint8)
    rm, rk = _swizzled_scale_shape(m, n)
    s = t.new_empty(rm, rk, dtype=torch.float8_e4m3fn)
    seed = torch.randint(0, 2**31 - 1, (1,), device="cpu").item() if stochastic else 0
    m_per_block = _recipe_m_per_block(m)
    grid = (triton.cdiv(n, 64), triton.cdiv(m, m_per_block))
    if m_per_block != 128:
        grid = (grid[0], grid[1] + 1)
    _recipe_quantize_kernel[grid](
        t,
        global_scale,
        q,
        s,
        t.stride(0),
        t.stride(1),
        m,
        n,
        seed,
        M_PER_BLOCK=m_per_block,
        USE_MASK=m % m_per_block != 0 or n % 64 != 0,
        HADAMARD=hadamard,
        STOCHASTIC=stochastic,
        USE_INT64_INDEXING=m * n > 2**31 - 1,
    )
    return (
        q.view(torch.float4_e2m1fn_x2),
        s.view(torch.float8_e4m3fn),
        (1.0 / global_scale).to(t.dtype),
    )


@_mslk_quantize_recipe_op.register_fake
def _(t, hadamard: bool, stochastic: bool):
    del hadamard, stochastic
    m, k = t.shape
    rm, rk = _swizzled_scale_shape(m, k)
    return (
        t.new_empty(m, k // 2, dtype=torch.float4_e2m1fn_x2),
        t.new_empty(rm, rk, dtype=torch.float8_e4m3fn),
        t.new_empty((), dtype=t.dtype),
    )


# MSLK's Triton quant kernels registered as opaque custom ops. Inductor's
# decompose_triton_kernel_wrapper_functional pass crashes if it traces INTO a raw
# Triton kernel in the dynamo graph; a registered op with a fake impl is a black
# box it compiles AROUND (no decompose, no graph break, no eager fallback). The
# concrete impl runs the kernel eagerly under compile — only the GEMM and the
# surrounding pointwise ops fuse, which is the win (the quant is already ~0.03ms).
# Profiled (sm_120, compiled compute-base LoRA step): the fprop activation quant
# is already 100% fused by Inductor into the preceding SiLU/RMSNorm epilogue, so a
# fused quant+GEMM prologue would only move that work out of mandatory pointwise
# ops it currently overlaps — net wash, not worth a custom MMA kernel. The only
# separable quant launch is the two-level dgrad quant (~30% of quant, <5% of step),
# which a fprop quant-GEMM does not touch.
@torch.library.custom_op("axolotl_nvfp4::quantize_two_level", mutates_args=())
def _mslk_quantize_op(
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4

    t = t.contiguous()
    amax = _abs_amax(t)
    global_scale = _NVFP4_GLOBAL_AMAX / torch.clamp(amax, min=1e-12)
    q, s = triton_quantize_nvfp4(t, global_scale)
    return (
        q.view(torch.float4_e2m1fn_x2),
        s.view(torch.float8_e4m3fn),
        (1.0 / global_scale).to(t.dtype),
    )


@_mslk_quantize_op.register_fake
def _(t):
    m, k = t.shape
    rm, rk = _swizzled_scale_shape(m, k)
    return (
        t.new_empty(m, k // 2, dtype=torch.float4_e2m1fn_x2),
        t.new_empty(rm, rk, dtype=torch.float8_e4m3fn),
        t.new_empty((), dtype=t.dtype),
    )


@torch.library.custom_op("axolotl_nvfp4::quantize_single_level", mutates_args=())
def _mslk_quantize_sl_op(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from mslk.quantize.triton.fp4_quantize import triton_quantize_nvfp4

    q, s = triton_quantize_nvfp4(t.contiguous(), None)
    return q.view(torch.float4_e2m1fn_x2), s.view(torch.float8_e4m3fn)


@_mslk_quantize_sl_op.register_fake
def _(t):
    m, k = t.shape
    rm, rk = _swizzled_scale_shape(m, k)
    return (
        t.new_empty(m, k // 2, dtype=torch.float4_e2m1fn_x2),
        t.new_empty(rm, rk, dtype=torch.float8_e4m3fn),
    )


def _mslk_quantize(t: torch.Tensor):
    """Quantize ``t`` (along its last dim) to NVFP4 via MSLK's fused Triton kernel.

    Returns ``(qdata, scale, inv_global_scale)`` ready for ``torch._scaled_mm``:
    ``qdata`` is ``float4_e2m1fn_x2`` packed, ``scale`` is the swizzled e4m3 block
    scale, and ``inv_global_scale = 1/global_scale`` is folded back into the GEMM
    output (two-level scaling). Routed through a registered custom op so it stays
    opaque-in-graph under torch.compile (see ``_mslk_quantize_op``).
    """
    return _mslk_quantize_op(t)


def _mslk_quantize_recipe(t: torch.Tensor, policy: QuantPolicy):
    if not (policy.hadamard or policy.stochastic) or not _recipe_fusion_available(t):
        return _mslk_quantize(t)
    return _mslk_quantize_recipe_op(t, bool(policy.hadamard), bool(policy.stochastic))


def _mslk_dequant(qdata, scale, inv_gs, shape, dtype=torch.bfloat16) -> torch.Tensor:
    """Dequantize MSLK-packed FP4 buffers back to a [*shape] hp tensor.

    MSLK emits swizzled e4m3 block scales (same layout as torchao's triton
    quant) and folds the two-level rescale into ``inv_gs = 1/global_scale``, so
    wrap into an NVFP4Tensor with ``per_tensor_scale=inv_gs`` and reuse torchao's
    dequant. ``shape`` is the logical [M,K] of the original hp tensor (qdata is
    [M,K/2] packed); used only to sanity-check the unpacked size.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    t = NVFP4Tensor(
        qdata,
        scale,
        _BLOCK_SIZE,
        dtype,
        per_tensor_scale=inv_gs.to(torch.float32),
        is_swizzled_scales=True,
    )
    out = t.dequantize(dtype)
    return out.reshape(shape)


def _mslk_scaled_mm(aq, a_scale, a_inv_gs, bq, b_scale, b_inv_gs, out_dtype):
    """``A @ B`` in FP4 from pre-quantized MSLK operands, two-level rescaled.

    ``aq`` is [M, K/2], ``bq`` is the B operand quantized as [N, K/2] so ``bq.t()``
    gives the [K, N] contraction layout ``_scaled_mm`` wants (TN).
    """
    out = torch._scaled_mm(
        aq,
        bq.t(),
        bias=None,
        out_dtype=out_dtype,
        scale_a=a_scale,
        scale_b=b_scale,
    )
    return out * (a_inv_gs * b_inv_gs)


def _prequant_sl(x_orig: torch.Tensor, x2d: torch.Tensor):
    """Single-level activation quant: reuse a fused-norm's pre-quantized result
    when ``x_orig`` is that norm's output (identity cache hit), else quantize
    ``x2d`` now. The cache stores the 2D ([M, K/2]) quant, drop-in for the
    ``_mslk_quantize_sl`` output.

    The cache is an identity-keyed WeakTensorKeyDictionary (untraceable) and only
    populated by the fuse_rmsnorm path (default OFF). Under torch.compile skip the
    lookup and quantize directly so the quant stays an opaque op in-graph; eager
    keeps the cache fast-path.
    """
    if torch.compiler.is_compiling():
        return _mslk_quantize_sl(x2d)
    from axolotl.kernels.nvfp4_rmsnorm import get_prequant

    cached = get_prequant(x_orig)
    if cached is not None:
        return cached
    return _mslk_quantize_sl(x2d)


def _mslk_quantize_sl(t: torch.Tensor):
    """Single-level NVFP4 quant (``global_scale=None``): MSLK fuses the amax into
    the kernel and bakes it into the block scales, so there is no separate
    per-tensor scale to compute (no amax reduction) or fold. ~3.6x faster than
    the two-level path. Safe ONLY for the forward ACTIVATION — its magnitudes are
    large enough that the e4m3 block scales don't underflow. Small weights and
    gradients DO underflow single-level, so they keep two-level. Routed through a
    registered custom op so it stays opaque-in-graph under torch.compile."""
    return _mslk_quantize_sl_op(t)


def _mslk_fprop_mm(aq, a_scale, bq, b_scale, b_inv_gs, out_dtype):
    """fprop GEMM: single-level activation (``aq``/``a_scale``, no per-tensor
    scale) @ two-level weight (``bq``/``b_scale``/``b_inv_gs``). Only the weight's
    per-tensor scale is folded post-GEMM (the activation has none)."""
    out = torch._scaled_mm(
        aq, bq.t(), bias=None, out_dtype=out_dtype, scale_a=a_scale, scale_b=b_scale
    )
    return out * b_inv_gs


class NVFP4FastComputeBaseFunction(torch.autograd.Function):
    """Compute-base fprop/dgrad using MSLK-quantized FP4 operands.

    Same math as :class:`NVFP4ComputeBaseFunction` (frozen base, two pre-quantized
    layouts, fprop + dgrad as ``_scaled_mm`` FP4 GEMMs, no wgrad) but the per-step
    activation/gradient quant uses MSLK's fused Triton kernel instead of the
    torchao quantizer — the prologue that otherwise dominates an unfused (whole-
    model-compiled) step.
    """

    @staticmethod
    def forward(
        ctx, x, wq_f, wsc_f, w_inv_f, wq_d, wsc_d, w_inv_d, out_features, recipe
    ):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        # fprop: single-level activation (no per-step amax) @ two-level weight.
        xq, xsc = _prequant_sl(x, x2d)
        out = _mslk_fprop_mm(xq, xsc, wq_f, wsc_f, w_inv_f, x.dtype)
        ctx.dgrad = (wq_d, wsc_d, w_inv_d)
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        ctx.out_features = out_features
        return out.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            wq_d, wsc_d, w_inv_d = ctx.dgrad
            g = grad_out.reshape(-1, ctx.out_features)
            gq, gsc, g_inv = _mslk_quantize_recipe(
                g, QuantPolicy(stochastic=ctx.recipe.stochastic_rounding)
            )
            grad_x = _mslk_scaled_mm(
                gq, gsc, g_inv, wq_d, wsc_d, w_inv_d, grad_out.dtype
            )
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None, None, None, None, None, None


class NVFP4FastComputeBaseLinear(nn.Module):
    """Frozen LoRA base with MSLK-fused FP4 compute on fprop + dgrad.

    Drop-in for :class:`NVFP4ComputeBaseLinear` (same memory: two FP4 weight
    layouts + scales) chosen when MSLK is available. Gradient-side SR is applied
    by the recipe-aware quantizer; RHT is irrelevant because the base is frozen
    and has no wgrad.
    """

    def __init__(self, w_f, w_d, bias, out_features, recipe: NVFP4Recipe):
        super().__init__()
        wq_f, wsc_f, w_inv_f = w_f  # fprop weight: two-level
        wq_d, wsc_d, w_inv_d = w_d  # dgrad weight: two-level
        self.register_buffer("wq_f", wq_f)
        self.register_buffer("wsc_f", wsc_f)
        self.register_buffer("w_inv_f", w_inv_f)
        self.register_buffer("wq_d", wq_d)
        self.register_buffer("wsc_d", wsc_d)
        self.register_buffer("w_inv_d", w_inv_d)
        self.bias = bias
        self.recipe = recipe
        self.out_features = out_features
        # wq_f packs the fprop layout [N, K/2]; K = in_features = 2 * packed cols.
        self.in_features = wq_f.shape[-1] * 2

    def forward(self, x):
        out = NVFP4FastComputeBaseFunction.apply(
            x,
            self.wq_f,
            self.wsc_f,
            self.w_inv_f,
            self.wq_d,
            self.wsc_d,
            self.w_inv_d,
            self.out_features,
            self.recipe,
        )
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight). Rebuilt from the fprop MSLK buffers.
        return _mslk_dequant(
            self.wq_f, self.wsc_f, self.w_inv_f, (self.out_features, self.in_features)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe
    ) -> "NVFP4FastComputeBaseLinear":
        w = linear.weight.detach()  # [N, K]
        # Both layouts two-level (small weights underflow single-level). fprop
        # B = wq_f.t(): quant W ([N,K]); dgrad B = wq_d.t(): quant W.T.
        w_f = _mslk_quantize(w)
        w_d = _mslk_quantize(w.t().contiguous())
        return cls(w_f, w_d, linear.bias, w.shape[0], recipe)


class NVFP4FastFrozenBaseFunction(torch.autograd.Function):
    """Storage-mode fprop/dgrad using MSLK-quantized FP4 operands.

    Same math as :class:`NVFP4FrozenBaseFunction` (single FP4 weight layout, ~3.5x
    weight memory; dgrad dequantizes the weight) but the per-step activation /
    gradient quant uses MSLK's fused Triton kernel instead of the torchao
    quantizer. The weight is stored ONLY in the fprop layout; dgrad dequantizes it
    to bf16 then re-quantizes both operands — so this keeps the single-layout
    memory win of storage mode, trading dgrad FLOPs for memory (vs the two-layout
    compute mode which stores a second FP4 dgrad layout to skip the dequant).
    """

    @staticmethod
    def forward(ctx, x, wq, wsc, w_inv, out_features, in_features, recipe):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        xq, xsc, x_inv = _mslk_quantize(x2d)
        out = _mslk_scaled_mm(xq, xsc, x_inv, wq, wsc, w_inv, x.dtype)
        ctx.wstore = (wq, wsc, w_inv)
        ctx.recipe = recipe
        ctx.x_shape = orig_shape
        ctx.out_features = out_features
        ctx.in_features = in_features
        return out.reshape(*orig_shape[:-1], out_features)

    @staticmethod
    def backward(ctx, grad_out):
        grad_x = None
        if ctx.needs_input_grad[0]:
            wq, wsc, w_inv = ctx.wstore
            g = grad_out.reshape(-1, ctx.out_features)
            # dgrad = g @ W; the stored layout is blocked along K (fprop), so
            # dequantize to bf16 for the contraction-along-N GEMM.
            w_hp = _mslk_dequant(
                wq, wsc, w_inv, (ctx.out_features, ctx.in_features), grad_out.dtype
            )
            gp, m = _pad_to_block(g, 0)
            gq, gsc, g_inv = _mslk_quantize_recipe(
                gp, QuantPolicy(stochastic=ctx.recipe.stochastic_rounding)
            )
            wdq, wdsc, wd_inv = _mslk_quantize(w_hp.t().contiguous())  # B = W.t()
            grad_x = _mslk_scaled_mm(gq, gsc, g_inv, wdq, wdsc, wd_inv, grad_out.dtype)[
                :m
            ]
            grad_x = grad_x.reshape(ctx.x_shape)
        return grad_x, None, None, None, None, None, None


class NVFP4FastFrozenBaseLinear(nn.Module):
    """Storage-mode frozen base (single FP4 weight, ~3.5x memory) with MSLK-fused
    per-step quant.

    Drop-in for :class:`NVFP4FrozenBaseLinear`, chosen when MSLK is available. Same
    single-layout FP4 storage; dgrad dequantizes the weight (no second layout).
    Gradient-side SR is applied by the recipe-aware quantizer; RHT is irrelevant
    because the base is frozen and has no wgrad.
    """

    def __init__(self, w_store, bias, out_features, in_features, recipe: NVFP4Recipe):
        super().__init__()
        wq, wsc, w_inv = w_store
        self.register_buffer("wq", wq)
        self.register_buffer("wsc", wsc)
        self.register_buffer("w_inv", w_inv)
        self.bias = bias
        self.recipe = recipe
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, x):
        out = NVFP4FastFrozenBaseFunction.apply(
            x,
            self.wq,
            self.wsc,
            self.w_inv,
            self.out_features,
            self.in_features,
            self.recipe,
        )
        return out if self.bias is None else out + self.bias

    @property
    def weight(self) -> torch.Tensor:
        # Read-only dequantized [N,K] for PEFT; writes don't persist (see
        # NVFP4FrozenBaseLinear.weight).
        return _mslk_dequant(
            self.wq, self.wsc, self.w_inv, (self.out_features, self.in_features)
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, recipe: NVFP4Recipe
    ) -> "NVFP4FastFrozenBaseLinear":
        w = linear.weight.detach()  # [N, K]
        w_store = _mslk_quantize(w)  # single fprop layout, blocked along K
        return cls(w_store, linear.bias, w.shape[0], w.shape[1], recipe)


# Base modules whose forward is an FP4 GEMM (no high-precision .weight to read).
# The fused LoRA kernels detect these to route the base GEMM through FP4 instead
# of reading base_layer.weight (which only NVFP4Linear, the hp mode, exposes).
def _nvfp4_base_classes() -> tuple:
    return (
        NVFP4Linear,
        NVFP4FrozenBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
    )


def is_nvfp4_base(module) -> bool:
    return isinstance(module, _nvfp4_base_classes())


def nvfp4_base_fprop(x: torch.Tensor, base) -> torch.Tensor:
    """``x @ W.T`` in FP4 for any native NVFP4 base module (2D ``x`` [M, K]).

    Mirrors each module's forward GEMM but as a plain (no-autograd) call so the
    fused LoRA autograd Functions can invoke it inside their own forward. Pads M
    to the FP4 alignment and slices it back.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    xp, m = _pad_to_block(x, 0)
    if isinstance(base, NVFP4FastComputeBaseLinear):
        xq, xsc = _mslk_quantize_sl(xp)
        out = _mslk_fprop_mm(xq, xsc, base.wq_f, base.wsc_f, base.w_inv_f, x.dtype)
    elif isinstance(base, NVFP4FastFrozenBaseLinear):
        xq, xsc = _mslk_quantize_sl(xp)
        out = _mslk_fprop_mm(xq, xsc, base.wq, base.wsc, base.w_inv, x.dtype)
    elif isinstance(base, NVFP4ComputeBaseLinear):
        # w_fprop is stored contiguous [N,K]; .t() gives the [K,N] b-operand
        # whose qdata.t() is contiguous (torchao's _addmm_nvfp4_dispatch invariant).
        out = _addmm_nvfp4_dispatch(
            _quantize(xp, QuantPolicy()), base.w_fprop.t(), torch.ops.aten.mm.default
        )
    elif isinstance(base, NVFP4FrozenBaseLinear):
        out = _addmm_nvfp4_dispatch(
            _quantize(xp, QuantPolicy()), base.w_q.t(), torch.ops.aten.mm.default
        )
    else:  # NVFP4Linear (hp): high-precision master weight, per-step requant
        out = _fp4_mm(xp, base.weight.t(), QuantPolicy(), QuantPolicy())
    return out[:m]


def nvfp4_base_fprop_many(
    x: torch.Tensor, bases: list | tuple
) -> list[torch.Tensor] | None:
    """Run several NVFP4 frozen-base fprops while sharing one activation pack.

    Fused LoRA QKV and gate/up projections all consume the same activation. The
    single-base helper quantizes that activation once per projection; this helper
    quantizes it once and feeds each pre-quantized base weight. Returns ``None``
    when any base is not a compatible frozen NVFP4 base so callers can fall back
    to the existing per-projection path.
    """
    if not bases or not all(is_nvfp4_base(base) for base in bases):
        return None
    if not all(
        isinstance(
            base,
            (
                NVFP4FastComputeBaseLinear,
                NVFP4FastFrozenBaseLinear,
                NVFP4ComputeBaseLinear,
                NVFP4FrozenBaseLinear,
            ),
        )
        for base in bases
    ):
        # NVFP4Linear has a live high-precision master weight and per-call weight
        # quantization, so only the frozen pre-quantized base modes are shared.
        return None

    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    xp, m = _pad_to_block(x2d, 0)
    lead = orig_shape[:-1]
    outs: list[torch.Tensor] = []

    if all(
        isinstance(base, (NVFP4FastComputeBaseLinear, NVFP4FastFrozenBaseLinear))
        for base in bases
    ):
        xq, xsc = _mslk_quantize_sl(xp)
        for base in bases:
            if isinstance(base, NVFP4FastComputeBaseLinear):
                out = _mslk_fprop_mm(
                    xq, xsc, base.wq_f, base.wsc_f, base.w_inv_f, x.dtype
                )
            else:
                out = _mslk_fprop_mm(xq, xsc, base.wq, base.wsc, base.w_inv, x.dtype)
            outs.append(out[:m].reshape(*lead, base.out_features))
        return outs

    if all(
        isinstance(base, (NVFP4ComputeBaseLinear, NVFP4FrozenBaseLinear))
        for base in bases
    ):
        from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

        a_q = _quantize(xp, QuantPolicy())
        for base in bases:
            if isinstance(base, NVFP4ComputeBaseLinear):
                # w_fprop stored contiguous [N,K]; .t() is the [K,N] b-operand.
                out = _addmm_nvfp4_dispatch(
                    a_q, base.w_fprop.t(), torch.ops.aten.mm.default
                )
                out_features = base.out_features
            else:
                out = _addmm_nvfp4_dispatch(
                    a_q, base.w_q.t(), torch.ops.aten.mm.default
                )
                out_features = base.out_features
            outs.append(out[:m].reshape(*lead, out_features))
        return outs

    return None


def nvfp4_base_dgrad(g: torch.Tensor, base) -> torch.Tensor:
    """``g @ W`` in FP4 (the base contribution to the input gradient), 2D ``g``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import _addmm_nvfp4_dispatch

    gp, m = _pad_to_block(g, 0)
    sr = QuantPolicy(stochastic=base.recipe.stochastic_rounding)
    if isinstance(base, NVFP4FastComputeBaseLinear):
        gq, gsc, g_inv = _mslk_quantize_recipe(gp, sr)
        out = _mslk_scaled_mm(
            gq, gsc, g_inv, base.wq_d, base.wsc_d, base.w_inv_d, g.dtype
        )
    elif isinstance(base, NVFP4ComputeBaseLinear):
        # w_dgrad stored contiguous [K,N]; .t() is the [N,K] b-operand.
        out = _addmm_nvfp4_dispatch(
            _quantize(gp, sr), base.w_dgrad.t(), torch.ops.aten.mm.default
        )
    elif isinstance(base, NVFP4FastFrozenBaseLinear):
        # single FP4 layout: dequantize the stored weight for the dgrad GEMM.
        w_hp = _mslk_dequant(
            base.wq,
            base.wsc,
            base.w_inv,
            (base.out_features, base.in_features),
            g.dtype,
        )
        out = _fp4_mm(gp, w_hp, sr, QuantPolicy())
    elif isinstance(base, NVFP4FrozenBaseLinear):
        out = _fp4_mm(gp, base.w_q.dequantize(g.dtype), sr, QuantPolicy())
    else:  # NVFP4Linear (hp)
        out = _fp4_mm(gp, base.weight, sr, QuantPolicy())
    return out[:m]


def _is_swappable(module: nn.Linear) -> bool:
    # Both in and out are contraction dims across fprop/dgrad, so both must meet
    # the _scaled_mm packed-contraction rule (logical %32, not just block %16) —
    # an out_features of 16 packs to 8 and trips "trailing dim divisible by 16".
    return (
        module.in_features % _GEMM_ALIGN == 0 and module.out_features % _GEMM_ALIGN == 0
    )


def _embedding_swappable(emb: nn.Embedding) -> bool:
    # Only the hidden dim is a quant block axis for the lookup (no GEMM), so the
    # NVFP4 block_size 16 is the only constraint — vocab is unrestricted.
    return emb.embedding_dim % _BLOCK_SIZE == 0


# Decoder-block norm attributes whose output feeds an NVFP4 base linear (qkv /
# gate-up). q_norm/k_norm (head-dim, feed attention) and the final norm (feeds
# lm_head, not an NVFP4 base) are intentionally excluded.
_FUSE_NORM_NAMES = frozenset(
    {
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    }
)


def convert_norms_to_nvfp4_fused(model: nn.Module) -> int:
    """Swap decoder-block RMSNorms for the fused RMSNorm->NVFP4-quant module.

    The fused norm emits the normalized activation AND its NVFP4 quant in one
    kernel, so the consuming base linear skips re-quantizing. The gamma convention
    (plain ``weight`` vs zero-centered ``1 + weight``) is detected per norm, and
    every swap is VERIFIED against the original on a probe before committing —
    a mismatch reverts the swap (never silently diverges). No-op if MSLK is missing.
    """
    try:
        from axolotl.kernels.nvfp4_rmsnorm import NVFP4FusedRMSNorm
    except Exception as exc:  # MSLK / triton unavailable
        LOG.warning("NVFP4 fused RMSNorm unavailable (%s); skipping norm fusion", exc)
        return 0

    parents = dict(model.named_modules())
    swapped = 0
    skipped = 0
    for name, module in list(model.named_modules()):
        attr = name.rsplit(".", 1)[-1]
        if attr not in _FUSE_NORM_NAMES:
            continue
        if not type(module).__name__.endswith("RMSNorm") or not hasattr(
            module, "weight"
        ):
            continue
        w = module.weight
        if not w.is_cuda:  # the fused kernel + the verify probe need CUDA
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = parents.get(parent_name)
        if parent is None:
            continue
        fused = NVFP4FusedRMSNorm.from_norm(module)
        # Verify the fused norm reproduces the original before committing — guards
        # against any unhandled gamma convention silently corrupting the forward.
        with torch.no_grad():
            probe = torch.randn(8, w.shape[-1], device=w.device, dtype=w.dtype)
            ref = module(probe).float()
            rel = (fused(probe).float() - ref).norm() / (ref.norm() + 1e-9)
        if rel > 0.05:
            LOG.warning("NVFP4: skip fused norm %s (rel-err %.3f)", name, rel.item())
            skipped += 1
            continue
        setattr(parent, attr, fused)
        swapped += 1
    LOG.info(
        "NVFP4 training: fused %d decoder RMSNorms (%d skipped on verify)",
        swapped,
        skipped,
    )
    return swapped


def convert_to_nvfp4_training(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
    skip_first_n_blocks: int = 0,
    skip_last_n_blocks: int = 0,
) -> int:
    """Swap eligible ``nn.Linear`` layers for ``NVFP4Linear`` in place.

    Sensitive layers stay high-precision per the convergence recipe: ``exclude``
    name fragments (lm_head/embeddings) plus the first/last N transformer blocks
    (block index parsed from the ``...layers.<i>...`` path). Returns the number
    of layers swapped.

    NOTE: ``skip_first_n_blocks``/``skip_last_n_blocks`` need the total block
    count; that policy is finalized by the integration layer. Here we only honor
    explicit name-fragment exclusion and dims%16 eligibility; block-range
    exclusion is applied by the caller via ``exclude`` for now.
    """
    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if any(frag in name for frag in exclude):
            continue
        if not _is_swappable(module):
            LOG.warning(
                "NVFP4: skipping %s (in=%d out=%d not both divisible by 32)",
                name,
                module.in_features,
                module.out_features,
            )
            continue
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, leaf, NVFP4Linear.from_linear(module, recipe))
        swapped += 1
    LOG.info("NVFP4 training: swapped %d linear layers", swapped)
    return swapped


def swap_frozen_linear_to_nvfp4(
    model: nn.Module,
    name: str,
    recipe: NVFP4Recipe | None = None,
    *,
    base_mode: str = "compute",
) -> bool:
    """Swap a single bare frozen ``nn.Linear`` (e.g. an un-targeted lm_head in a
    LoRA run) for the matching NVFP4 base module.

    The LoRA base converter only touches ``lora.Linear`` modules; a frozen
    lm_head that isn't a LoRA target stays a bare ``nn.Linear`` and is invisible
    to it. This swaps that bare module in place using the same three base modes:
    ``compute`` (default), ``storage``, or ``hp``. Returns True if swapped.
    """
    recipe = recipe or NVFP4Recipe()
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return False
    if not isinstance(module, nn.Linear) or not _is_swappable(module):
        return False

    fast = _mslk_available()
    if base_mode == "compute":
        cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
        build = lambda src: cls.from_linear(src, recipe)  # noqa: E731
    elif base_mode == "storage":
        if fast:
            build = lambda src: NVFP4FastFrozenBaseLinear.from_linear(  # noqa: E731
                src, recipe
            )
        else:
            build = lambda src: NVFP4FrozenBaseLinear.from_linear(  # noqa: E731
                src, recipe, fsdp=False
            )
    else:
        # hp mode keeps a high-precision trainable master weight (no quant), so
        # there is no FP4 transient to stream around; swap in place.
        module.weight.requires_grad_(False)
        hp = NVFP4Linear.from_linear(module, recipe)
        _set_submodule(model, name, hp)
        _dynamo_disable_forward(hp)
        LOG.info("NVFP4 training: swapped frozen %s (mode=%s)", name, base_mode)
        return True

    new_module = _stream_quantize_swap(model, name, module, build)
    _dynamo_disable_forward(new_module)
    LOG.info("NVFP4 training: swapped frozen %s (mode=%s)", name, base_mode)
    return True


def swap_frozen_lm_head_tileable(
    model: nn.Module,
    name: str,
    recipe: NVFP4Recipe | None = None,
) -> bool:
    """Swap a bare frozen lm_head to the row-sliceable torchao FP4 store.

    The fused FP4 cross-entropy reads the packed lm_head weight tile by tile, which
    is only bit-exact when the e4m3 block scales are row-major (NOT swizzled). The
    MSLK-fast storage class keeps swizzled scales, so force the torchao
    :class:`NVFP4FrozenBaseLinear` (non-swizzled) here regardless of MSLK
    availability. lm_head is one frozen layer and the fused path dequantizes it
    itself, so skipping the MSLK fast-quant for it costs nothing.
    """
    recipe = recipe or NVFP4Recipe()
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return False
    if not isinstance(module, nn.Linear) or not _is_swappable(module):
        return False
    new_module = _stream_quantize_swap(
        model,
        name,
        module,
        lambda src: NVFP4FrozenBaseLinear.from_linear(src, recipe, fsdp=False),
    )
    _dynamo_disable_forward(new_module)
    LOG.info("NVFP4 training: swapped frozen %s (tileable storage for fused CE)", name)
    return True


def _set_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".", 1)[-1], new_module)


def _dynamo_disable_forward(module: nn.Module) -> None:
    """Force ``module.forward`` to run eager under torch.compile.

    The FP4 lm_head sits at the tail of the graph between the final RMSNorm and
    cross-entropy. With gradient_checkpointing OFF (activations saved, not
    recomputed) + flash-attention-2 + torch.compile, Inductor fuses the
    flash-attn backward with the FP4 lm_head dgrad into a graph that produces a
    NaN input gradient on the very first step (the loss is briefly finite, then
    collapses to ln(vocab)). Any one of {gc-on, sdpa, eager, bf16 lm_head}
    avoids it, so the FP4 lm_head GEMM math itself is correct — the failure is
    that specific fused region. Breaking the graph around the one lm_head module
    keeps it out of that fusion (it is a single frozen layer; eager costs
    nothing) while the rest of the model stays compiled.
    """
    inner = module.forward

    @torch._dynamo.disable
    def _eager_forward(*args, **kwargs):
        return inner(*args, **kwargs)

    module.forward = _eager_forward


def _reclaim_gpu() -> None:
    """Sync then release freed GPU blocks back to the allocator.

    Sync first: the quant kernels still read the source weight async, and
    expandable_segments would otherwise hand a freed-but-in-flight block to the
    next allocation (use-after-free / illegal access).
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _stream_quantize_swap(model, name, source, build):
    """Quantize ``source`` to its NVFP4 module keeping the transient peak small.

    The lm_head/embed/vision swaps run AFTER the full model is on the GPU, so on a
    near-full card the source bf16 + the new FP4 layouts (+ quant scratch) coexist
    and OOM. The torchao quant scratch is already bounded by the chunked quantizer
    (see ``_to_nvfp4_chunked``); this drops the bf16 source the moment it is no
    longer needed and reclaims it before the next swap, so the resident bf16 isn't
    carried alongside the FP4 copies — mirroring the LoRA base streaming. Quant
    stays on the GPU (CPU quant is NOT bit-identical: torchao's e2m1 rounding
    diverges between CPU and CUDA).
    """
    new_module = build(source)
    _set_submodule(model, name, new_module)
    # Free the now-replaced bf16 source. Sync first: the quant kernels may still
    # read it async, and expandable_segments would otherwise hand the freed block
    # to the next allocation (use-after-free / illegal access).
    if isinstance(source, nn.Module) and source is not new_module:
        if getattr(source, "weight", None) is not None:
            source.weight = None
        if getattr(source, "bias", None) is not None:
            source.bias = None
    _reclaim_gpu()
    return new_module


def swap_frozen_embedding_to_nvfp4(
    model: nn.Module, name: str
) -> NVFP4Embedding | None:
    """Swap a FROZEN ``nn.Embedding`` for :class:`NVFP4Embedding` in place.

    Skips (returns None) a trainable embedding — an FP4-stored weight can't carry
    gradients — or a hidden dim not divisible by 16. Returns the new module on
    success so the tied path can route the lm_head through the same store.
    """
    try:
        module = model.get_submodule(name)
    except AttributeError:
        return None
    if not isinstance(module, nn.Embedding):
        return None
    if module.weight is not None and module.weight.requires_grad:
        LOG.warning(
            "nvfp4_training.quantize_embeddings: %s is trainable; skipping (an "
            "FP4-stored embedding has no high-precision master for gradients).",
            name,
        )
        return None
    if not _embedding_swappable(module):
        LOG.warning(
            "nvfp4_training.quantize_embeddings: %s hidden dim %d not divisible "
            "by %d; keeping it in high precision.",
            name,
            module.embedding_dim,
            _BLOCK_SIZE,
        )
        return None
    new_module = _stream_quantize_swap(
        model, name, module, lambda src: NVFP4Embedding.from_embedding(src)
    )
    LOG.info("NVFP4 training: swapped frozen embedding %s", name)
    return new_module


def swap_tied_embedding_and_lm_head_to_nvfp4(
    model: nn.Module,
    embed_name: str,
    lm_head_name: str,
    recipe: NVFP4Recipe | None = None,
) -> bool:
    """Quantize a tied (shared) FROZEN weight ONCE and route both roles to it.

    The shared weight becomes a single :class:`NVFP4Embedding` store; the input
    embedding reads it for the lookup and the lm_head reads the SAME store for
    its GEMM (:class:`NVFP4TiedLMHead`), so the dequantized weight is identical
    for both. No-op (False) if the shared weight is trainable or not eligible —
    the caller must keep RAISING on a trainable tied weight.
    """
    recipe = recipe or NVFP4Recipe()
    try:
        embed = model.get_submodule(embed_name)
        lm_head = model.get_submodule(lm_head_name)
    except AttributeError:
        return False
    if not isinstance(embed, nn.Embedding) or not isinstance(lm_head, nn.Linear):
        return False
    if not _embedding_swappable(embed) or not _is_swappable(lm_head):
        LOG.warning(
            "nvfp4_training: tied embedding/lm_head dims not NVFP4-eligible "
            "(hidden %%16 and lm_head %%32); keeping both in high precision."
        )
        return False
    # Capture the lm_head bias before streaming frees the shared weight, then
    # quantize the shared weight ONCE via the embedding and route the lm_head GEMM
    # at the SAME store. The shared weight is held by BOTH modules, so drop the
    # lm_head's reference too — otherwise streaming the embedding can't reclaim it.
    lm_head_bias = lm_head.bias
    lm_head.weight = None
    new_embed = _stream_quantize_swap(
        model, embed_name, embed, lambda src: NVFP4Embedding.from_embedding(src)
    )
    tied_head = NVFP4TiedLMHead(new_embed, lm_head_bias, recipe)
    _set_submodule(model, lm_head_name, tied_head)
    _dynamo_disable_forward(tied_head)
    LOG.info("NVFP4 training: tied embedding/lm_head quantized once (shared FP4 store)")
    return True


# Vision-tower submodule names / class-name fragments to locate the encoder.
_VISION_ATTR_NAMES = ("visual", "vision_tower", "vision_model")
_VISION_CLASS_FRAGS = ("Vision",)
# Linears under the vision tower that are NOT encoder GEMMs: the merger /
# patch-embed projection feed the language model, not the attention/MLP stack.
_VISION_SKIP_FRAGS = ("merger", "patch_embed", "deepstack")


def _find_vision_tower(model: nn.Module):
    """Return (name, module) of the vision encoder, or (None, None).

    Prefers the conventional attribute names (``visual``/``vision_tower``/
    ``vision_model``, possibly one level under ``model``); falls back to the
    first submodule whose class name contains "Vision".
    """
    for prefix in ("", "model."):
        for attr in _VISION_ATTR_NAMES:
            name = f"{prefix}{attr}"
            try:
                mod = model.get_submodule(name)
            except AttributeError:
                continue
            if isinstance(mod, nn.Module):
                return name, mod
    for name, mod in model.named_modules():
        if not name:
            continue
        cls = type(mod).__name__
        if any(frag in cls for frag in _VISION_CLASS_FRAGS) and any(
            isinstance(c, nn.Linear) for c in mod.modules()
        ):
            return name, mod
    return None, None


def convert_vision_tower_to_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    base_mode: str = "compute",
) -> int:
    """Swap eligible FROZEN ``nn.Linear`` layers under the vision tower to NVFP4.

    Scopes strictly to linears under the located vision encoder (attn qkv/proj,
    mlp fc1/fc2); the merger / patch-embed projection and any %32-ineligible or
    trainable linear are skipped. Warns and returns 0 if no vision tower is found
    (text-only model). Returns the count swapped.
    """
    recipe = recipe or NVFP4Recipe()
    vt_name, vt = _find_vision_tower(model)
    if vt is None:
        LOG.warning(
            "nvfp4_training.quantize_vision_tower: no vision tower found "
            "(visual/vision_tower/vision_model or a *Vision* module); skipping."
        )
        return 0

    swapped = 0
    for name, module in list(vt.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(frag in name for frag in _VISION_SKIP_FRAGS):
            continue
        if module.weight is not None and module.weight.requires_grad:
            continue
        if not _is_swappable(module):
            LOG.warning(
                "nvfp4_training.quantize_vision_tower: skipping %s.%s "
                "(in=%d out=%d not both divisible by %d)",
                vt_name,
                name,
                module.in_features,
                module.out_features,
                _GEMM_ALIGN,
            )
            continue
        fast = _mslk_available()
        if base_mode == "compute":
            cls = NVFP4FastComputeBaseLinear if fast else NVFP4ComputeBaseLinear
            build = lambda src, cls=cls: cls.from_linear(src, recipe)  # noqa: E731
        elif base_mode == "storage":
            if fast:
                build = lambda src: NVFP4FastFrozenBaseLinear.from_linear(  # noqa: E731
                    src, recipe
                )
            else:
                build = lambda src: NVFP4FrozenBaseLinear.from_linear(  # noqa: E731
                    src, recipe, fsdp=False
                )
        else:
            module.weight.requires_grad_(False)
            _set_submodule(vt, name, NVFP4Linear.from_linear(module, recipe))
            swapped += 1
            continue
        _stream_quantize_swap(vt, name, module, build)
        swapped += 1
    LOG.info(
        "nvfp4_training.quantize_vision_tower: swapped %d linears under %s (mode=%s)",
        swapped,
        vt_name,
        base_mode,
    )
    return swapped


def te_nvfp4_available() -> tuple[bool, str]:
    """Return (ok, reason) for the Transformer Engine NVFP4 backend."""
    try:
        import transformer_engine.pytorch  # noqa: F401
        from transformer_engine.common.recipe import NVFP4BlockScaling  # noqa: F401
    except Exception as exc:  # ImportError or the cuBLAS-symbol OSError
        return False, (
            f"Transformer Engine NVFP4 backend unavailable ({type(exc).__name__}: "
            f"{exc}). Install axolotl[transformer-engine] (source build; on sm_120 "
            "use NVTE_CUDA_ARCHS=120 and preload the system cuBLAS)."
        )
    return True, ""


def te_nvfp4_recipe(recipe: "NVFP4Recipe"):
    """Build a TE NVFP4BlockScaling recipe. On consumer Blackwell (sm_120) the
    RHT/SR/2D fusion kernels do not run, so disable them and warn — TE there is
    a recipe-less FP4 GEMM. On sm_100 (B200) the full recipe runs."""
    from transformer_engine.common.recipe import NVFP4BlockScaling

    cap = torch.cuda.get_device_capability()
    if cap == (12, 0):
        LOG.warning(
            "TE NVFP4 on sm_120 (consumer Blackwell): RHT/stochastic-rounding/2D "
            "kernels do not run here, disabling them — convergence recipe is OFF "
            "(unproven at scale). Use backend=native for the full recipe on sm_120."
        )
        return NVFP4BlockScaling(
            disable_rht=True,
            disable_stochastic_rounding=True,
            disable_2d_quantization=True,
        )
    return NVFP4BlockScaling(
        disable_rht=not recipe.hadamard,
        disable_stochastic_rounding=not recipe.stochastic_rounding,
    )


def convert_to_te_nvfp4_training(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap eligible ``nn.Linear`` for ``transformer_engine.pytorch.Linear`` so
    they run NVFP4 GEMMs under TE's ``NVFP4BlockScaling`` recipe (FFT only).

    The caller must wrap the training step in ``te.fp8_autocast(fp8_recipe=
    te_nvfp4_recipe(recipe))``. Weights are copied into the TE linear; dims must
    be divisible by 16.
    """
    import transformer_engine.pytorch as te

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or any(f in name for f in exclude):
            continue
        if not _is_swappable(module):
            LOG.warning("NVFP4(te): skipping %s (dims not divisible by 16)", name)
            continue
        te_lin = te.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            params_dtype=module.weight.dtype,
        )
        with torch.no_grad():
            te_lin.weight.copy_(module.weight)
            if module.bias is not None:
                te_lin.bias.copy_(module.bias)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], te_lin)
        swapped += 1
    LOG.info("NVFP4 training (te backend): swapped %d linear layers", swapped)
    return swapped


def convert_lora_base_to_te_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap the FROZEN base_layer inside each PEFT ``lora.Linear`` for a
    ``transformer_engine.pytorch.Linear`` so the base GEMM runs NVFP4 under TE's
    ``fp8_autocast`` (the trainer wraps the step via ``te_nvfp4_recipe``).

    The trainable LoRA adapters stay high-precision ``nn.Linear`` and are left
    untouched; only the frozen base runs FP4, so TE computes its dgrad (input
    gradient) but no wgrad — which also sidesteps TE's wgrad token-dim %32
    constraint. Weights are copied in; dims must be divisible by 16.
    """
    import transformer_engine.pytorch as te
    from peft.tuners.lora import Linear as LoraLinear

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoraLinear):
            continue
        if any(frag in name for frag in exclude):
            continue
        base = module.base_layer
        if not isinstance(base, nn.Linear) or not _is_swappable(base):
            continue
        te_lin = te.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            params_dtype=base.weight.dtype,
        )
        with torch.no_grad():
            te_lin.weight.copy_(base.weight)
            if base.bias is not None:
                te_lin.bias.copy_(base.bias)
        for p in te_lin.parameters():
            p.requires_grad_(False)
        module.base_layer = te_lin
        swapped += 1
    LOG.info("NVFP4 training (te backend): swapped %d LoRA base layers", swapped)
    return swapped


def convert_lora_base_to_nvfp4(
    model: nn.Module,
    recipe: NVFP4Recipe | None = None,
    *,
    quantized_storage: bool = False,
    compute_base: bool = False,
    fsdp: bool = False,
    exclude: tuple[str, ...] = ("lm_head", "embed_tokens"),
) -> int:
    """Swap the FROZEN base_layer inside each PEFT ``lora.Linear`` for an NVFP4
    linear, leaving the trainable adapters in high precision.

    Three base modes (mutually exclusive, checked in priority order):

    - ``compute_base=True`` (LoRA + FP4 compute, recommended): base_layer ->
      NVFP4ComputeBaseLinear. The frozen base is pre-quantized ONCE into two
      NVFP4 layouts; fprop+dgrad run as pure FP4 GEMMs with no per-step base
      quant prologue. ~1.75x weight memory and the fastest base compute.
    - ``quantized_storage=True`` (NVFP4-QLoRA): base_layer ->
      NVFP4FrozenBaseLinear, base stored packed in FP4 (~3.5x weight memory);
      backward dequantizes to bf16. Max memory, modest speed. When MSLK is
      available (and not FSDP) the MSLK-fused NVFP4FastFrozenBaseLinear is used
      instead — same single-layout storage, fast per-step quant.
    - neither (default): base_layer -> NVFP4Linear, base kept high-precision and
      re-quantized each step. FP4 base GEMM, no memory win.

    Returns the number of base layers swapped. Requires PEFT-wrapped adapters.
    """
    from peft.tuners.lora import Linear as LoraLinear

    recipe = recipe or NVFP4Recipe()
    swapped = 0
    skipped_offloaded = 0
    # Stream the swap. For the NVFP4 adapter path the loader keeps the base on
    # CPU (so the full bf16 model never sits on the GPU and device_map can't
    # strand weights on meta). Move each base weight to the GPU just-in-time,
    # quantize to FP4, and free the bf16 — the GPU only ever holds the FP4 base
    # plus one transient layer. Weights already on the GPU are quantized in place.
    # A materialized named_modules() list would pin every base_layer and defeat
    # the per-layer free, so keep only the lora.Linear references.
    target = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lora_modules = [
        (n, m) for n, m in model.named_modules() if isinstance(m, LoraLinear)
    ]
    streamed = False
    for name, module in lora_modules:
        if any(frag in name for frag in exclude):
            continue
        base = module.base_layer
        if not isinstance(base, nn.Linear) or not _is_swappable(base):
            continue
        # A meta weight has no data to quantize (device_map offloaded it because
        # even the CPU-streamed load couldn't place it). Can't support it.
        if base.weight is None or base.weight.is_meta:
            skipped_offloaded += 1
            continue
        if base.weight.device != target:
            base = base.to(target)  # stream this layer to the GPU
            streamed = True
        if compute_base:
            # MSLK-fast compute stores raw uint8/fp8 buffers (no NVFP4Tensor
            # subclass to hook), so it can't shard under FSDP — fall back to the
            # torchao compute class (which carries the all-gather hooks) under FSDP.
            fast = _mslk_available() and not fsdp
            if fast:
                module.base_layer = NVFP4FastComputeBaseLinear.from_linear(base, recipe)
            else:
                module.base_layer = NVFP4ComputeBaseLinear.from_linear(
                    base, recipe, fsdp=fsdp
                )
        elif quantized_storage:
            # MSLK-fused storage has no FSDP all-gather hooks yet, so keep the
            # torchao NVFP4FrozenBaseLinear (which carries them) under FSDP.
            if _mslk_available() and not fsdp:
                module.base_layer = NVFP4FastFrozenBaseLinear.from_linear(base, recipe)
            else:
                module.base_layer = NVFP4FrozenBaseLinear.from_linear(
                    base, recipe, fsdp=fsdp
                )
        else:
            base.weight.requires_grad_(False)
            module.base_layer = NVFP4Linear.from_linear(base, recipe)
        swapped += 1
        # Free the now-replaced bf16 base immediately. Otherwise the full bf16
        # model stays resident while the FP4 copies accumulate on top (the swap
        # transiently doubles memory) — which OOMs large models on load even
        # though the FP4 base itself is far smaller. hp mode keeps its weight.
        if not (compute_base or quantized_storage):
            continue
        # Drop the bf16 weight now so peak stays near the FP4 footprint, not
        # bf16's. Sync first: the quant kernels still read this weight async, and
        # expandable_segments would hand the freed block to the next layer's
        # quant — a use-after-free that corrupts the kernel (illegal access).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        base.weight = None
        base.bias = None
        del base
    # The FP4 base is in place; move the rest (embeddings, norms, lm_head, the
    # LoRA adapters) onto the GPU now that the heavy weights are quantized.
    if streamed:
        model.to(target)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if compute_base:
        mode = "compute (mslk-fast)" if (_mslk_available() and not fsdp) else "compute"
    elif quantized_storage:
        mode = "storage (mslk-fast)" if (_mslk_available() and not fsdp) else "storage"
    else:
        mode = "hp"
    LOG.info("NVFP4 training: swapped %d LoRA base layers (mode=%s)", swapped, mode)
    if skipped_offloaded:
        raise RuntimeError(
            f"nvfp4_training: {skipped_offloaded} base weight(s) are on meta with no "
            "data to quantize — the model didn't fully materialize even via the "
            "CPU-streamed load. Use a smaller base, add VRAM/GPUs, or adapter: qlora."
        )
    return swapped


# --- NVFP4-packed save/load (opt-in nvfp4_training.save_nvfp4) -----------------
#
# safetensors cannot serialize the NVFP4Tensor subclass nor the packed FP4
# buffers (invalid python storage on the float4/uint8 inner tensors), so the
# packed weights go to a torch.save sidecar next to the standard save. The bf16
# weights of the FP4 modules are then dropped from the main (safetensors) shard
# to realize the disk win.

NVFP4_PACKED_SIDECAR = "nvfp4_packed.pt"


def _all_nvfp4_modules(model: nn.Module):
    """Yield (name, module) for every NVFP4 module that owns an FP4 store.

    NVFP4TiedLMHead is intentionally excluded: it has no buffer of its own and
    reads the NVFP4Embedding's ``w_q`` (saved once via the embedding).
    """
    owning = _nvfp4_base_classes() + (NVFP4Embedding,)
    for name, module in model.named_modules():
        if isinstance(module, owning):
            yield name, module


def collect_nvfp4_packed_state(model: nn.Module) -> tuple[dict, set[str]]:
    """Build the FP4-packed sidecar dict and the set of bf16 keys it supersedes.

    Returns ``(packed, drop_keys)`` where ``packed`` maps ``"<module>.<buffer>"``
    to the FP4 tensor (NVFP4Tensor subclass or plain packed buffer) and
    ``drop_keys`` are the full-model state_dict keys whose bf16 form is now
    redundant (the FP4 modules' weights) and should be omitted from the main
    safetensors shard.

    For frozen modules the FP4 buffers are already the stored form (bit-exact).
    For the FFT NVFP4Linear (bf16 master) the weight is packed to FP4 here —
    LOSSY: no bf16 master is kept, so this is for storage/inference export, not
    exact resume.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        QuantizeTensorToNVFP4Kwargs,
        per_tensor_amax_to_scale,
    )

    packed: dict = {}
    drop: set[str] = set()
    for name, module in _all_nvfp4_modules(model):
        prefix = f"{name}." if name else ""
        if isinstance(module, NVFP4Linear):
            # FFT: pack the bf16 master to FP4 (lossy for resume).
            w = module.weight.detach()
            pts = per_tensor_amax_to_scale(_abs_amax(w))
            w_q = _to_nvfp4_chunked(
                w.contiguous(), pts, QuantizeTensorToNVFP4Kwargs(block_size=_BLOCK_SIZE)
            )
            packed[f"{prefix}w_q"] = w_q
            drop.add(f"{prefix}weight")
        else:
            # Frozen modules: the FP4 buffers ARE the stored form (bit-exact).
            for bname, buf in module.named_buffers(recurse=False):
                packed[f"{prefix}{bname}"] = buf
                drop.add(f"{prefix}{bname}")
    return packed, drop


def save_nvfp4_packed(model: nn.Module, output_dir) -> int:
    """Write the FP4-packed sidecar for ``model`` into ``output_dir``.

    Returns the number of packed tensors written (0 => no NVFP4 modules, no
    sidecar). The sidecar is consumed by :func:`load_nvfp4_packed` at load.
    """
    import os

    packed, _ = collect_nvfp4_packed_state(model)
    if not packed:
        return 0
    cpu = {k: (v.cpu() if hasattr(v, "cpu") else v) for k, v in packed.items()}
    path = os.path.join(str(output_dir), NVFP4_PACKED_SIDECAR)
    torch.save(cpu, path)
    has_fft = any(isinstance(m, NVFP4Linear) for _, m in _all_nvfp4_modules(model))
    LOG.info(
        "NVFP4 save_nvfp4: wrote %d packed tensor(s) to %s%s",
        len(packed),
        path,
        " (FFT weights are FP4-only — LOSSY for exact resume)" if has_fft else "",
    )
    return len(packed)


def load_nvfp4_packed(model: nn.Module, model_dir) -> int:
    """Restore FP4-packed weights from a sidecar into the converted ``model``.

    Run AFTER the nvfp4 module conversion (so the FP4 modules exist). Frozen
    modules load their FP4 buffers bit-exactly; the FFT NVFP4Linear has its bf16
    master replaced by the dequantized FP4 (export fidelity, not exact). Returns
    the number of tensors restored (0 => no sidecar found).
    """
    import os

    path = os.path.join(str(model_dir), NVFP4_PACKED_SIDECAR)
    if not os.path.isfile(path):
        return 0
    packed = torch.load(path, weights_only=False, map_location="cpu")  # nosec B614
    by_module: dict[str, dict] = {}
    for key, tensor in packed.items():
        mod_name, buf_name = key.rsplit(".", 1)
        by_module.setdefault(mod_name, {})[buf_name] = tensor

    restored = 0
    for mod_name, buffers in by_module.items():
        try:
            module = model.get_submodule(mod_name)
        except AttributeError:
            continue
        device = next(
            (b.device for b in module.buffers()),
            next((p.device for p in module.parameters()), torch.device("cpu")),
        )
        if isinstance(module, NVFP4Linear) and "w_q" in buffers:
            # FFT load-back: dequantize the FP4 into the bf16 master (lossy).
            w_q = buffers["w_q"].to(device)
            with torch.no_grad():
                module.weight.copy_(w_q.dequantize(module.weight.dtype))
            restored += 1
            continue
        for bname, tensor in buffers.items():
            if not hasattr(module, bname):
                continue
            tensor = tensor.to(device)
            existing = getattr(module, bname)
            if existing is not None and isinstance(existing, nn.Parameter):
                with torch.no_grad():
                    existing.copy_(tensor)
            else:
                module.register_buffer(bname, tensor)
            restored += 1
    LOG.info("NVFP4 save_nvfp4: restored %d packed tensor(s) from %s", restored, path)
    return restored
