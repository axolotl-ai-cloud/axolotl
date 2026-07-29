"""Ternary modules: latent bf16 weights, fake-quant forward, bake-on-save."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from types import ModuleType
from typing import Any, Literal, get_args

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.weak import WeakTensorKeyDictionary
from transformers import PreTrainedModel

from axolotl.utils.logging import get_logger

from . import quant
from .args import (
    LEARNABLE_SCALE_MODES,
    TWO_PLANE_SCALE_MODES,
    Codebook,
    WeightScaleMode,
)

LOG = get_logger(__name__)

# quantized activations shared by the linears that consume one tensor (q/k/v, gate/up)
_ACT_MEMO: WeakTensorKeyDictionary = WeakTensorKeyDictionary()

SCALE_MODES: frozenset[str] = frozenset(get_args(WeightScaleMode))
CODEBOOKS: frozenset[str] = frozenset(get_args(Codebook))

# schema-accepted scale modes whose quantizer has not landed yet
UNIMPLEMENTED_SCALE_MODES: frozenset[str] = frozenset()

# scale modes carrying one scale per output channel, shaped `(out_features, 1)`
ROW_SCALE_MODES: frozenset[str] = frozenset({"learnable_row", "dual", "trit_planes"})

# the grids the fused Triton kernels and the W2A8 int8 forward implement; everything
# else runs the eager oracle until a kernel for its scale layout lands
FUSED_SCALE_MODES: frozenset[str] = frozenset({"absmean", "group"})
INT8_FORWARD_SCALE_MODES: frozenset[str] = frozenset({"absmean", "learnable"})

# state-dict entries holding a learnable scale, in `__init__` order
SCALE_ATTRS: tuple[str, ...] = ("scale", "scale_lo")

# where `subln.insert_subln` parks the norm on the source Linear, and the attribute
# the ternary module adopts it into — so it serializes as `<linear>.sub_norm.weight`
SUBLN_ATTR: str = "sub_norm"


@functools.lru_cache(maxsize=1)
def _fused_ops() -> ModuleType | None:
    """Return the Triton kernel module when it can run here, else `None`."""
    try:
        from .kernels import triton as triton_ops

        if triton_ops.is_available():
            return triton_ops
    except (ImportError, NotImplementedError):
        return None
    return None


def as_dtensor(tensor: torch.Tensor) -> torch.Tensor | None:
    """Return the DTensor behind `tensor` (possibly its `.data`), or `None` if plain.

    Sharded parameters must never reach a Triton kernel: a DTensor reports the global
    `numel` while its storage only holds the local shard, so the kernel indexes past
    the allocation.
    """
    for candidate in (tensor, getattr(tensor, "data", None)):
        if candidate is not None and hasattr(candidate, "to_local"):
            return candidate
    return None


def as_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the rank-local shard of a DTensor, or `tensor` itself when it is plain."""
    sharded = as_dtensor(tensor)
    return tensor if sharded is None else sharded.to_local()


def _gathered(tensor: torch.Tensor) -> torch.Tensor:
    """Return the full tensor behind a DTensor, or `tensor` itself when it is plain."""
    sharded = as_dtensor(tensor)
    return tensor if sharded is None else sharded.full_tensor()


def _redistribute(reference: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
    """Shard `full` the way `reference` is sharded."""
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(full, reference.device_mesh, reference.placements)


@functools.lru_cache(maxsize=1)
def _int8_ops() -> ModuleType | None:
    """Return the int8 W2A8 forward module, or `None` when it cannot be imported."""
    try:
        from .kernels import int8_gemm
    except ImportError:
        return None
    return int8_gemm


class _FusedQuantSTE(torch.autograd.Function):
    """Straight-through estimator around a fused kernel; backward is the identity."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, fn, *args) -> torch.Tensor:  # type: ignore[override]
        """Return `fn(tensor, *args)`; the kernel is the forward, autograd is bypassed."""
        ctx.num_args = len(args)
        return fn(tensor, *args)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        """Pass the incoming gradient straight through to `tensor`."""
        return (grad_output, None) + (None,) * ctx.num_args


class TernaryLinear(nn.Module):
    """Linear layer whose weight is fake-quantized to `{-s, 0, +s}` on every forward.

    Deliberately not an `nn.Linear` subclass: the stored `weight` is a *latent*
    full-precision parameter that receives dense gradients, and the ternary values
    only exist inside the forward. Biases are unsupported.

    `weight_scale: dual` widens the grid to the five values `{0, ±s_lo, ±s_hi}` per
    row, carried by two learnable scale vectors; the pair is kept ordered by reading
    them as `(min, max)`, so both keep receiving gradient however they move.

    `codebook: binary` narrows it the other way, to `{-s, +s}`: the same scale
    machinery drives a sign plane with no zero state, so every grid-shaped thing here
    — the bake, the baked probe, the learnable-scale seed, the monitoring snapshot —
    takes the binary member of the pair and nothing else changes.

    Attributes:
        weight: Latent `nn.Parameter` of shape `(out_features, in_features)`.
        scale: `nn.Parameter` holding `log_s`, shape `(1,)` in `weight_scale:
            learnable` and `(out_features, 1)` in the per-row modes (`log_s_hi` under
            `dual`); `None` otherwise.
        scale_lo: `nn.Parameter` holding `log_s_lo`, shape `(out_features, 1)`, in
            `weight_scale: dual`; `None` otherwise.
        sub_norm: RMSNorm over the input, applied before quantization, when
            `ternary.subln` inserted one for this family; `None` otherwise.
        lambda_: Current quantization strength in `[0, 1]`, driven by
            `LambdaScheduleCallback`; 1.0 means fully ternary.
        weight_scale: Scale mode this module was built with.
        codebook: `"ternary"` for `{-s, 0, +s}`, `"binary"` for the sign plane.
        group_size: Group size in group-scale mode, `None` for per tensor.
        activation_bits: 8 for per-token int8 activation quantization, `None` for
            weight-only QAT.
        fused: Whether the fused Triton path is used instead of the eager oracle. Only
            the per-tensor and per-group grids have kernels, so the per-row modes run
            the eager oracle whatever this says.
        int8_forward: Whether the W2A8 int8 GEMM replaces the fake-quant forward once
            λ reaches 1; `"auto"` and `True` both fall back whenever the shapes,
            device or quantizer options do not qualify — including every scale mode
            whose grid the one-scalar epilogue cannot express.
        baked: Whether `weight` currently holds the exact ternary values, in which
            case the forward stops re-quantizing it. Set by `_post_training` and by
            the structural probe that runs whenever a baked master is loaded; it
            lapses as soon as an optimizer step moves the latent again.
        share_act_quant: Opt-in reuse of one quantized activation across the linears
            that consume the same input tensor.
    """

    weight: nn.Parameter
    scale: nn.Parameter | None
    scale_lo: nn.Parameter | None
    sub_norm: nn.Module | None
    lambda_: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: WeightScaleMode = "absmean",
        codebook: Codebook = "ternary",
        group_size: int | None = None,
        activation_bits: int | None = 8,
        fused: bool = True,
        int8_forward: Literal["auto"] | bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Build an (uninitialized) ternary linear; see `from_linear` for conversion.

        Raises:
            ValueError: If `group_size` does not divide `in_features`, the scale
                mode and `group_size` disagree, or the codebook and the scale mode
                cannot be combined.
            NotImplementedError: For a scale mode the schema accepts but the
                quantizer does not implement yet.
        """
        super().__init__()
        if weight_scale not in SCALE_MODES:
            raise ValueError(f"unknown ternary weight_scale mode: {weight_scale!r}")
        if codebook not in CODEBOOKS:
            raise ValueError(f"unknown ternary codebook: {codebook!r}")
        if codebook == "binary" and weight_scale in TWO_PLANE_SCALE_MODES:
            raise ValueError(
                f"codebook='binary' cannot be combined with weight_scale="
                f"{weight_scale!r}: every state its second plane adds is a zero or a "
                "second magnitude, and a sign codebook has neither"
            )
        if weight_scale in UNIMPLEMENTED_SCALE_MODES:
            raise NotImplementedError(
                f"ternary.weight_scale: {weight_scale} is accepted by the schema but "
                "its quantizer is not implemented yet"
            )
        if weight_scale == "group":
            if group_size is None:
                raise ValueError("weight_scale='group' requires a group_size")
            if in_features % group_size:
                raise ValueError(
                    f"group_size {group_size} does not divide in_features {in_features}"
                )
        elif group_size is not None:
            raise ValueError(
                f"group_size is only valid with weight_scale='group', got {weight_scale!r}"
            )
        if activation_bits not in (None, 8):
            raise ValueError(
                f"activation_bits must be 8 or None, got {activation_bits!r}"
            )
        if int8_forward not in ("auto", True, False):
            raise ValueError(
                f"int8_forward must be 'auto', True or False, got {int8_forward!r}"
            )

        self.in_features = in_features
        self.out_features = out_features
        self.weight_scale = weight_scale
        self.codebook = codebook
        self.group_size = group_size
        self.activation_bits = activation_bits
        self.fused = fused
        self.int8_forward = int8_forward
        self.baked = False
        self.share_act_quant = False
        self.lambda_ = 1.0
        self._int8_warned = False
        self._baked_version: int | None = None
        self._baked_grad_hook: Any = None

        self.add_module(SUBLN_ATTR, None)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if weight_scale == "learnable":
            # shape (1,), not a scalar: FSDP2 refuses to shard 0-dim parameters
            self.scale = nn.Parameter(
                torch.zeros(1, device=device, dtype=torch.float32)
            )
        elif weight_scale in ROW_SCALE_MODES:
            self.scale = nn.Parameter(
                torch.zeros(out_features, 1, device=device, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)
        if weight_scale in TWO_PLANE_SCALE_MODES:
            self.scale_lo = nn.Parameter(
                torch.zeros(out_features, 1, device=device, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale_lo", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_scale: WeightScaleMode = "absmean",
        codebook: Codebook = "ternary",
        group_size: int | None = None,
        activation_bits: int | None = 8,
        fused: bool = True,
        int8_forward: Literal["auto"] | bool = False,
    ) -> "TernaryLinear":
        """Build a `TernaryLinear` that adopts `linear`'s weight as its latent weight.

        A sub-norm parked on the source module by `subln.insert_subln` is adopted with
        it, so it serializes and reloads as a child of the ternary module.

        Args:
            linear: Source module; its weight is moved (not copied) where possible so
                the swap does not double peak memory.
            weight_scale: Scale mode.
            codebook: `"ternary"` or `"binary"`.
            group_size: Group size for group-scale mode.
            activation_bits: 8 or `None`.
            fused: Use the fused Triton kernels.
            int8_forward: Use the W2A8 int8 GEMM once λ reaches 1.

        Returns:
            The replacement module, on the source module's device and dtype.

        Raises:
            ValueError: If `linear` has a bias (unsupported in v1).
        """
        if getattr(linear, "bias", None) is not None:
            raise ValueError(
                "ternary conversion does not support biased Linear layers; keep biased "
                "projections in full precision via ternary.keep_fp_modules"
            )

        module = cls(
            linear.in_features,
            linear.out_features,
            weight_scale=weight_scale,
            codebook=codebook,
            group_size=group_size,
            activation_bits=activation_bits,
            fused=fused,
            int8_forward=int8_forward,
            device="meta",
            dtype=linear.weight.dtype,
        )
        weight = linear.weight
        module.weight = (
            weight if isinstance(weight, nn.Parameter) else nn.Parameter(weight)
        )
        sub_norm = getattr(linear, SUBLN_ATTR, None)
        if sub_norm is not None:
            setattr(module, SUBLN_ATTR, sub_norm)
        module._detect_baked()
        module.refresh_scale_from_weight()
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize the weight (and activations when `activation_bits == 8`), then GEMM."""
        if self.sub_norm is not None:
            x = self.sub_norm(x)
        lambda_ = self.lambda_
        if lambda_ <= 0.0:
            return F.linear(x, self.weight)
        if self.int8_forward and lambda_ >= 1.0 and x.is_cuda:
            out = self._int8_linear(x)
            if out is not None:
                return out
        if self.activation_bits == 8:
            x = self._quant_act(x, lambda_)
        weight = self.weight if self.is_baked() else self._quant_weight(lambda_)
        return F.linear(x, weight)

    def set_lambda(self, value: float) -> None:
        """Set the quantization strength for subsequent forwards."""
        self.lambda_ = float(min(max(value, 0.0), 1.0))

    def is_baked(self) -> bool:
        """Whether `weight` still holds the exact quantized values it was baked to.

        Training the latent lapses the flag, so the quantizer takes over again. Two
        independent signals invalidate it, because neither covers the other: the
        weight's autograd version counter catches any in-place write, and a gradient
        landing on the weight catches the fused optimizers (`adamw_torch_fused` and
        friends) that update through `torch._fused_adamw_` without touching the
        version — those would otherwise keep a fit-initialized module "baked" for a
        whole run, so it would never bake at save time.
        """
        return self.baked and self.weight._version == self._baked_version

    def baked_weight(self) -> torch.Tensor:
        """Return the exact `codes * s16` weight in the latent dtype (λ-independent)."""
        with torch.no_grad():
            weight = self.weight.detach()
            if self.is_baked():
                return weight.clone()
            sharded = as_dtensor(weight)
            if sharded is None:
                return self._bake(weight, gathered=False)
            # a per-tensor scale is global, so bake off the gathered tensor and
            # redistribute; every rank runs the same all-gather
            baked = self._bake(sharded.full_tensor(), gathered=True)
            return _redistribute(sharded, baked)

    def code_snapshot(self) -> torch.Tensor:
        """Return the current codes packed 4-per-byte, for flip-rate monitoring.

        Under FSDP2 the weight is a DTensor, so the snapshot covers this rank's shard
        only and the statistics derived from it are per-rank. A five-value module
        snapshots two ternary planes per weight (see `quant.dual_state_planes`).
        """
        with torch.no_grad():
            weight = as_local(self.weight.detach())
            if self.weight_scale == "trit_planes":
                first, second = self._trit_plane_scales(weight)
                codes = quant.trit_plane_codes(weight, first, second)
            elif self.weight_scale == "dual":
                low, high = self._dual_scales(weight)
                codes = quant.dual_state_planes(quant.dual_codes(weight, low, high))
            else:
                scale = self._snapshot_scale()
                if scale is None:
                    scale = self._statistic_scale(weight)
                codes = self._codes(weight, scale)
        ops = self._ops(codes)
        if ops is not None:
            return ops.pack_codes(codes)
        return quant.pack_codes(codes)

    def code_count(self) -> int:
        """Number of codes `code_snapshot` covers — the local shard under FSDP2."""
        count = as_local(self.weight.detach()).numel()
        two_plane = self.weight_scale in TWO_PLANE_SCALE_MODES
        return count * quant.DUAL_PLANES if two_plane else count

    def refresh_scale_from_weight(self) -> None:
        """Re-seed the learnable scale(s) from the current latent weight.

        The swap seeds them, but a PTQ initializer rewrites the latents afterwards, so
        a scale left over from the pre-fit tensor would code the fitted one wrong.
        Rebinds the parameters, so it only runs before any parallelism wrapping.
        """
        if self.weight_scale not in LEARNABLE_SCALE_MODES:
            return
        weight = self.weight
        if (
            weight.device.type == "meta"
            or not weight.numel()
            or as_dtensor(weight) is not None
        ):
            return
        if self.weight_scale == "trit_planes":
            first, second = self._initial_trit_plane_scales()
            self.scale_lo = nn.Parameter(second.log().to(weight.device))
            self.scale = nn.Parameter(first.log().to(weight.device))
            return
        if self.weight_scale == "dual":
            low, high = self._initial_dual_scales()
            self.scale_lo = nn.Parameter(low.log().to(weight.device))
            self.scale = nn.Parameter(high.log().to(weight.device))
            return
        shape = (self.out_features, 1) if self.weight_scale == "learnable_row" else (1,)
        scale = self._initial_scale().reshape(shape)
        self.scale = nn.Parameter(scale.log().to(weight.device))

    def _post_training(self, model: PreTrainedModel, name: str) -> None:
        """Bake the latent weight to `codes * s16` in place so every save path emits the master.

        Called by axolotl's post-training module hook before the model is saved. A
        learnable scale is folded into the baked values and then dropped.

        Raises:
            RuntimeError: If the write did not land — a silently unbaked parameter
                would be saved as an ordinary FP checkpoint labelled as a master.
        """
        self._record_final_lambda(model, name)
        # re-derive from the values rather than trusting the flag: this runs once per
        # module, and a stale flag here writes the trained latents into the master
        if not self._detect_baked():
            baked = self.baked_weight()
            with torch.no_grad():
                # write through the parameter: rebinding `.data` is a silent no-op on
                # a DTensor, which would leave FP latents behind a `baked` flag
                self.weight.copy_(baked)
            if not bool(torch.equal(as_local(self.weight.detach()), as_local(baked))):
                raise RuntimeError(
                    f"ternary: baking {name} did not land in its weight; the checkpoint "
                    "would hold latent values under a ternary manifest"
                )
            self._mark_baked()
        self._record_baked_scales(model, name)
        # unconditional: a module already on its grid still carries the parameters,
        # and a master that ships them reloads into a different function
        for attr in SCALE_ATTRS:
            if getattr(self, attr) is not None:
                setattr(self, attr, None)

    def extra_repr(self) -> str:
        """Return the shape, scale mode and activation bits for `repr(model)`."""
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"weight_scale={self.weight_scale}",
        ]
        if self.codebook != "ternary":
            parts.append(f"codebook={self.codebook}")
        if self.group_size is not None:
            parts.append(f"group_size={self.group_size}")
        parts.append(f"activation_bits={self.activation_bits}")
        parts.append(f"lambda={self.lambda_:g}")
        if self.baked:
            parts.append("baked=True")
        return ", ".join(parts)

    def _statistic_scale(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the scale the statistic modes re-derive from `weight` every forward."""
        group_size = self._scale_group_size()
        if self.codebook == "binary":
            return quant.binary_scale(weight, group_size)
        return quant.absmean_scale(weight, group_size)

    def _codes(self, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Return this module's codebook's codes for `weight` under `scale`."""
        if self.codebook == "binary":
            return quant.binary_codes(weight, scale)
        return quant.ternary_codes(weight, scale)

    def _baked_codes_and_scale(
        self, weight: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Recover `(codes, s16)` from `weight` on this module's grid, or `None`."""
        group_size = self._scale_group_size()
        if self.codebook == "binary":
            return quant.baked_binary_codes_and_scale(weight, group_size)
        return quant.baked_codes_and_scale(weight, group_size)

    def _ops(self, tensor: torch.Tensor) -> ModuleType | None:
        # the fused kernels are CUDA-only and reject sharded tensors; anything else
        # falls back to the eager oracle
        if not self.fused or not tensor.is_cuda or as_dtensor(tensor) is not None:
            return None
        return _fused_ops()

    def _weight_ops(self, tensor: torch.Tensor) -> ModuleType | None:
        """`_ops`, restricted to the grids the single-plane fused kernel covers."""
        if self.weight_scale not in FUSED_SCALE_MODES:
            return None
        return self._ops(tensor)

    def _multi_state_ops(
        self, tensor: torch.Tensor, lambda_: float = 1.0
    ) -> ModuleType | None:
        """The fused kernels for the grids the eager assignment is slowest on.

        `dual` and `trit_planes` materialize a comparison tensor per candidate level
        eagerly; `binary` is cheap either way but shares the pass. All three fall back
        to the oracle off CUDA, under FSDP2 sharding, or without Triton.

        One more fallback keeps fused and eager the *same function* rather than merely
        a close one: in fp32 with λ < 1 the kernel's blend contracts to an FMA and
        lands a single ulp from the oracle's two separately-rounded operations. bf16
        and f16 round that away, and λ = 1 skips the blend entirely, so the fused path
        covers every case a real heal spends its time in — an fp32 λ-warmup is the one
        it hands back.
        """
        if self._ops(tensor) is None:
            return None
        if tensor.dtype is torch.float32 and lambda_ < 1.0:
            return None
        try:
            from .kernels.triton import multi_state
        except (ImportError, NotImplementedError):  # pragma: no cover
            return None
        return multi_state

    def _int8_linear(self, x: torch.Tensor) -> torch.Tensor | None:
        # the W2A8 epilogue rescales by one scalar per tensor, over ternary codes
        supported = (
            self.weight_scale in INT8_FORWARD_SCALE_MODES and self.codebook == "ternary"
        )
        ops = _int8_ops() if supported else None
        out = None if ops is None else ops.int8_linear_forward(self, x)
        if out is None and self.int8_forward is True and not self._int8_warned:
            self._int8_warned = True
            LOG.warning(
                f"ternary: int8_forward is unavailable for a {self.out_features}x"
                f"{self.in_features} {x.dtype} linear; using the fake-quant forward"
            )
        return out

    def _scale_group_size(self) -> int | None:
        """Group size the quantizer sees: a per-row scale is one group of `in_features`."""
        if self.weight_scale == "learnable_row":
            return self.in_features
        return self.group_size

    def _scale(self, gathered: bool = False) -> torch.Tensor | None:
        """Return the fp32 scale, floored; `gathered` unshards it first.

        `None` in the statistic modes, and under `dual` — that grid carries a pair,
        see `_dual_scales`.
        """
        if self.scale is None or self.weight_scale in TWO_PLANE_SCALE_MODES:
            return None
        # a (1,) scale sharded over N ranks leaves most of them an empty local
        scale = _gathered(self.scale) if gathered else self.scale
        shape = (self.out_features, 1) if self.weight_scale == "learnable_row" else ()
        return scale.float().exp().reshape(shape).clamp_min(quant.SCALE_EPS)

    def _snapshot_scale(self) -> torch.Tensor | None:
        """Return the scale paired with the rank-local weight shard."""
        if self.weight_scale != "learnable_row":
            return self._scale(gathered=True)
        scale = self._scale()
        # per-row scales shard the way the rows do, so the local halves already match
        return None if scale is None else as_local(scale)

    def _dual_scales(
        self, weight: torch.Tensor, gathered: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordered fp32 per-row `(s_lo, s_hi)` pair aligned with `weight`.

        Falls back to the statistic once `_post_training` has folded the parameters
        into the values and dropped them.
        """
        if self.scale_lo is None or self.scale is None:
            return quant.dual_absmean_scales(weight)
        low: torch.Tensor = self.scale_lo
        high: torch.Tensor = self.scale
        if gathered:
            low, high = _gathered(low), _gathered(high)
        elif as_dtensor(weight) is None:
            # per-row scales shard the way the rows do, so the local halves match
            low, high = as_local(low), as_local(high)
        low, high = low.float().exp(), high.float().exp()
        return (
            torch.minimum(low, high).clamp_min(quant.SCALE_EPS),
            torch.maximum(low, high).clamp_min(quant.SCALE_EPS),
        )

    def quantization_scale(self) -> torch.Tensor:
        """Return the fp32 scale one code step spans, broadcastable over the weight.

        The two-plane grids report the *finer* of their pair: it is the step that
        separates neighbouring states, so anything measured against it (init jitter,
        for one) stays proportional to the grid rather than to the coarse plane.
        """
        weight = as_local(self.weight.detach())
        if self.weight_scale == "trit_planes":
            return self._trit_plane_scales(weight)[1]
        if self.weight_scale == "dual":
            return self._dual_scales(weight)[0]
        scale = self._scale()
        if scale is None:
            return self._statistic_scale(weight)
        return scale

    def adopt_scales(self, *scales: torch.Tensor) -> bool:
        """Take a persisted two-plane grid as this module's own.

        Returns:
            Whether the weight is on the adopted grid — `False` leaves the module
            untouched, so a mismatched sidecar cannot quietly redefine a master.
        """
        if self.weight_scale not in TWO_PLANE_SCALE_MODES or len(scales) != 2:
            return False
        weight = as_local(self.weight.detach())
        first, second = (scale.to(weight.device).to(torch.float32) for scale in scales)
        if self.weight_scale == "trit_planes":
            recovered = quant.baked_trit_plane_codes_and_scales(weight, (first, second))
        else:
            recovered = quant.baked_dual_codes_and_scales(weight)
        if recovered is None:
            return False
        self.scale_lo = nn.Parameter(recovered[2].log().to(weight.device))
        self.scale = nn.Parameter(recovered[1].log().to(weight.device))
        self._mark_baked()
        return True

    def _trit_plane_scales(
        self, weight: torch.Tensor, gathered: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ordered fp32 per-row `(s1, s2)` pair aligned with `weight`.

        Falls back to the statistic once `_post_training` has folded the parameters
        into the values and dropped them.
        """
        if self.scale_lo is None or self.scale is None:
            return quant.trit_plane_absmean_scales(weight)
        second: torch.Tensor = self.scale_lo
        first: torch.Tensor = self.scale
        if gathered:
            first, second = _gathered(first), _gathered(second)
        elif as_dtensor(weight) is None:
            # per-row scales shard the way the rows do, so the local halves match
            first, second = as_local(first), as_local(second)
        first, second = first.float().exp(), second.float().exp()
        return (
            torch.maximum(first, second).clamp_min(quant.SCALE_EPS),
            torch.minimum(first, second).clamp_min(quant.SCALE_EPS),
        )

    def _bake(self, weight: torch.Tensor, gathered: bool) -> torch.Tensor:
        """Return `weight` on the module's quantization grid, exactly."""
        if self.weight_scale == "trit_planes":
            first, second = self._trit_plane_scales(weight, gathered=gathered)
            return quant.fake_quant_weight_trit_planes(weight, 1.0, first, second)
        if self.weight_scale == "dual":
            low, high = self._dual_scales(weight, gathered=gathered)
            return quant.fake_quant_weight_dual(weight, 1.0, low, high)
        fake_quant = (
            quant.fake_quant_weight_binary
            if self.codebook == "binary"
            else quant.fake_quant_weight
        )
        return fake_quant(
            weight, 1.0, self._scale_group_size(), self._scale(gathered=gathered)
        )

    def _initial_scale(self) -> torch.Tensor:
        """Return the fp32 scale a learnable `log_s` starts from."""
        weight = as_local(self.weight.detach())
        recovered = self._baked_codes_and_scale(weight)
        if recovered is not None:
            return recovered[1]
        return self._statistic_scale(weight)

    def _initial_dual_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fp32 `(s_lo, s_hi)` pair the learnable dual scales start from."""
        weight = as_local(self.weight.detach())
        recovered = quant.baked_dual_codes_and_scales(weight)
        if recovered is not None:
            return recovered[1], recovered[2]
        return quant.dual_absmean_scales(weight)

    def _initial_trit_plane_scales(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the fp32 `(s1, s2)` pair the learnable free-sum scales start from."""
        weight = as_local(self.weight.detach())
        recovered = quant.baked_trit_plane_codes_and_scales(weight)
        if recovered is not None:
            return recovered[1], recovered[2]
        return quant.trit_plane_absmean_scales(weight)

    def _detect_baked(self) -> bool:
        """Flag the module when its weight already holds exactly the quantized values.

        A baked master is the interchange artifact, so it is what gets loaded back;
        re-deriving an absmean scale from ternary values would shrink every magnitude
        by the non-zero code fraction, and a per-row grid read per tensor would lose
        the rows entirely.
        """
        weight = as_local(self.weight.detach())
        if weight.device.type == "meta" or not weight.numel():
            return False
        if self.weight_scale == "trit_planes":
            self.baked = quant.baked_trit_plane_codes_and_scales(weight) is not None
        elif self.weight_scale == "dual":
            self.baked = quant.baked_dual_codes_and_scales(weight) is not None
        else:
            self.baked = self._baked_codes_and_scale(weight) is not None
        if self.baked:
            self._mark_baked()
        else:
            self._clear_baked()
        return self.baked

    def _mark_baked(self) -> None:
        """Flag the latent as holding exact quantized values, and arm the invalidators."""
        self.baked = True
        self._baked_version = self.weight._version
        if self._baked_grad_hook is None and self.weight.requires_grad:
            self._baked_grad_hook = self.weight.register_post_accumulate_grad_hook(
                lambda _param: self._clear_baked()
            )

    def _clear_baked(self) -> None:
        """A trained latent is no longer on the grid, whatever the optimizer did."""
        self.baked = False
        self._baked_version = None
        if self._baked_grad_hook is not None:
            self._baked_grad_hook.remove()
            self._baked_grad_hook = None

    def _record_baked_scales(self, model: PreTrainedModel | None, name: str) -> None:
        """Persist the grid this module baked on, where the values cannot carry it.

        Every `dual` value is a pure multiple of one scale, so a reload recovers the
        pair from the magnitudes. A free sum is not: a row that uses only combination
        states stores `s1 + s2` and `s1 - s2` and never `s1` itself, which two
        unknowns cannot be read back out of. Those masters carry their scales in the
        manifest sidecar instead of leaving the loader to guess.
        """
        from .swap import get_manifest

        manifest = None if model is None else get_manifest(model)
        if manifest is None or self.weight_scale not in TWO_PLANE_SCALE_MODES:
            return
        weight = as_local(self.weight.detach())
        pair = (
            self._trit_plane_scales(weight)
            if self.weight_scale == "trit_planes"
            else self._dual_scales(weight)
        )
        manifest.record_scales(name, *pair)

    def _record_final_lambda(self, model: PreTrainedModel | None, name: str) -> None:
        """Note the λ the bake happened at, and warn when the schedule never finished."""
        from .swap import get_manifest

        manifest = None if model is None else get_manifest(model)
        first = manifest is None or manifest.final_lambda is None
        if manifest is not None:
            manifest.final_lambda = min(
                self.lambda_, 1.0 if first else manifest.final_lambda
            )
        if self.lambda_ < 1.0 and first:
            LOG.warning(
                f"ternary: baking {name} at lambda={self.lambda_:g}. The saved master "
                "is fully ternary, but the model was trained and evaluated as "
                f"w + {self.lambda_:g}*(wq - w) — a different function. Extend training "
                "or shorten ternary.lambda_warmup_steps so the schedule reaches 1.0."
            )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        """Re-run the baked probe after `load_state_dict` replaces the weight.

        A master baked at save time ships no scales — they were folded into its values
        — so a learnable mode re-seeds them from the reloaded weight.
        """
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._detect_baked()
        if not any(f"{prefix}{attr}" in state_dict for attr in SCALE_ATTRS):
            self.refresh_scale_from_weight()

    def _quant_weight(self, lambda_: float) -> torch.Tensor:
        multi_state = self._multi_state_ops(self.weight, lambda_)
        if self.weight_scale == "trit_planes":
            first, second = self._trit_plane_scales(self.weight)
            return quant.fake_quant_weight_trit_planes_ste(
                self.weight,
                lambda_,
                first,
                second,
                impl=None
                if multi_state is None
                else multi_state.fake_quant_weight_trit_planes,
            )
        if self.weight_scale == "dual":
            low, high = self._dual_scales(self.weight)
            return quant.fake_quant_weight_dual_ste(
                self.weight,
                lambda_,
                low,
                high,
                impl=None
                if multi_state is None
                else multi_state.fake_quant_weight_dual,
            )
        scale = self._scale()
        # fused kernels carry no gradient for a learnable scale
        ops = self._weight_ops(self.weight) if scale is None else None
        if ops is not None:
            kernel = (
                multi_state.fake_quant_weight_binary
                if self.codebook == "binary" and multi_state is not None
                else ops.fake_quant_weight
            )
            if self.codebook != "binary" or multi_state is not None:
                return _FusedQuantSTE.apply(
                    self.weight, kernel, lambda_, self.group_size, None
                )
        fake_quant_ste = (
            quant.fake_quant_weight_binary_ste
            if self.codebook == "binary"
            else quant.fake_quant_weight_ste
        )
        return fake_quant_ste(self.weight, lambda_, self._scale_group_size(), scale)

    def _quant_act(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        if not self.share_act_quant:
            return self._quant_act_uncached(x, lambda_)
        stamp = (lambda_, x._version, torch.is_grad_enabled())
        cached = _ACT_MEMO.get(x)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        quantized = self._quant_act_uncached(x, lambda_)
        _ACT_MEMO[x] = (stamp, quantized)
        return quantized

    def _quant_act_uncached(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ops = self._ops(x)
        if ops is not None:
            return _FusedQuantSTE.apply(x, ops.act_quant, lambda_)
        return quant.act_quant_ste(x, lambda_)


class TernaryExperts(nn.Module):
    """Ternary replacement for fused 3D MoE expert stacks (planned, M5)."""

    def __init__(self, *args, **kwargs) -> None:
        """Always raises; fused expert tensors are detected and rejected by the swap."""
        super().__init__()
        raise NotImplementedError(
            "ternary conversion of fused MoE expert stacks is not supported yet"
        )


def iter_ternary_modules(model: nn.Module) -> Iterator[tuple[str, TernaryLinear]]:
    """Yield `(name, module)` for every `TernaryLinear` in `model`."""
    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            yield name, module


def assert_codebook_applied(model: nn.Module, codebook: Codebook) -> None:
    """Refuse a converted model whose modules do not implement the configured codebook.

    The codebook is a property of the module, and the swap is what puts it there.
    Nothing downstream can tell the two apart afterwards — a binary run whose modules
    were built ternary trains, bakes, exports and reloads as a perfectly consistent
    ternary model — so the disagreement is caught here rather than shipped.

    Raises:
        RuntimeError: If any `TernaryLinear` carries a different codebook.
    """
    built = {module.codebook for _, module in iter_ternary_modules(model)}
    wrong = sorted(built - {codebook})
    if wrong:
        raise RuntimeError(
            f"ternary.codebook: {codebook} was configured, but the swap built "
            f"{', '.join(wrong)} modules. The codebook reaches a module only through "
            "`TernaryLinear.from_linear(..., codebook=...)`, so `swap.convert_model` "
            "has to pass `ternary_cfg.codebook` (and record it on the manifest for "
            "the export side)"
        )


def set_act_quant_sharing(model: nn.Module, enabled: bool = True) -> int:
    """Toggle the shared activation-quant memo on every `TernaryLinear` in `model`.

    Returns:
        The number of modules updated.
    """
    count = 0
    for _, module in iter_ternary_modules(model):
        module.share_act_quant = enabled
        count += 1
    return count
