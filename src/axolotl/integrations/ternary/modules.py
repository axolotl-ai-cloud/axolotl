"""Ternary modules: latent bf16 weights, fake-quant forward, bake-on-save."""

from __future__ import annotations

import functools
from collections.abc import Iterator
from types import ModuleType
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.weak import WeakTensorKeyDictionary
from transformers import PreTrainedModel

from axolotl.utils.logging import get_logger

from . import quant
from .args import WeightScaleMode

LOG = get_logger(__name__)

# quantized activations shared by the linears that consume one tensor (q/k/v, gate/up)
_ACT_MEMO: WeakTensorKeyDictionary = WeakTensorKeyDictionary()


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

    Attributes:
        weight: Latent `nn.Parameter` of shape `(out_features, in_features)`.
        scale: `nn.Parameter` holding `log_s` in `weight_scale: learnable` mode,
            `None` otherwise.
        lambda_: Current quantization strength in `[0, 1]`, driven by
            `LambdaScheduleCallback`; 1.0 means fully ternary.
        weight_scale: Scale mode this module was built with.
        group_size: Group size in group-scale mode, `None` for per tensor.
        activation_bits: 8 for per-token int8 activation quantization, `None` for
            weight-only QAT.
        fused: Whether the fused Triton path is used instead of the eager oracle.
        int8_forward: Whether the W2A8 int8 GEMM replaces the fake-quant forward once
            λ reaches 1; `"auto"` and `True` both fall back whenever the shapes,
            device or quantizer options do not qualify.
        baked: Whether `weight` currently holds the exact ternary values, in which
            case the forward stops re-quantizing it. Set by `_post_training` and by
            the structural probe that runs whenever a baked master is loaded; it
            lapses as soon as an optimizer step moves the latent again.
        share_act_quant: Opt-in reuse of one quantized activation across the linears
            that consume the same input tensor.
    """

    weight: nn.Parameter
    scale: nn.Parameter | None
    lambda_: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_scale: WeightScaleMode = "absmean",
        group_size: int | None = None,
        activation_bits: int | None = 8,
        fused: bool = True,
        int8_forward: Literal["auto"] | bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Build an (uninitialized) ternary linear; see `from_linear` for conversion.

        Raises:
            ValueError: If `group_size` does not divide `in_features`, or the scale
                mode and `group_size` disagree.
        """
        super().__init__()
        if weight_scale not in ("absmean", "group", "learnable"):
            raise ValueError(f"unknown ternary weight_scale mode: {weight_scale!r}")
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
        self.group_size = group_size
        self.activation_bits = activation_bits
        self.fused = fused
        self.int8_forward = int8_forward
        self.baked = False
        self.share_act_quant = False
        self.lambda_ = 1.0
        self._int8_warned = False
        self._baked_version: int | None = None

        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        if weight_scale == "learnable":
            # shape (1,), not a scalar: FSDP2 refuses to shard 0-dim parameters
            self.scale = nn.Parameter(
                torch.zeros(1, device=device, dtype=torch.float32)
            )
        else:
            self.register_parameter("scale", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        weight_scale: WeightScaleMode = "absmean",
        group_size: int | None = None,
        activation_bits: int | None = 8,
        fused: bool = True,
        int8_forward: Literal["auto"] | bool = False,
    ) -> "TernaryLinear":
        """Build a `TernaryLinear` that adopts `linear`'s weight as its latent weight.

        Args:
            linear: Source module; its weight is moved (not copied) where possible so
                the swap does not double peak memory.
            weight_scale: Scale mode.
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
        module._detect_baked()
        if module.scale is not None:
            scale = module._initial_scale()
            module.scale = nn.Parameter(scale.reshape(1).log().to(weight.device))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fake-quantize the weight (and activations when `activation_bits == 8`), then GEMM."""
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
        """Whether `weight` still holds the exact ternary values it was baked to.

        An optimizer step (or any other in-place write) bumps the parameter's version
        counter, which lapses the flag so the quantizer takes over again.
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
                return quant.fake_quant_weight(
                    weight, 1.0, self.group_size, self._scale()
                )
            # the per-tensor scale is global, so bake off the gathered tensor and
            # redistribute; every rank runs the same all-gather
            baked = quant.fake_quant_weight(
                sharded.full_tensor(), 1.0, self.group_size, self._scale(gathered=True)
            )
            return _redistribute(sharded, baked)

    def code_snapshot(self) -> torch.Tensor:
        """Return the current codes packed 4-per-byte, for flip-rate monitoring.

        Under FSDP2 the weight is a DTensor, so the snapshot covers this rank's shard
        only and the statistics derived from it are per-rank.
        """
        with torch.no_grad():
            weight = as_local(self.weight.detach())
            scale = self._scale(gathered=True)
            if scale is None:
                scale = quant.absmean_scale(weight, self.group_size)
            codes = quant.ternary_codes(weight, scale)
        ops = self._ops(codes)
        if ops is not None:
            return ops.pack_codes(codes)
        return quant.pack_codes(codes)

    def code_count(self) -> int:
        """Number of codes `code_snapshot` covers — the local shard under FSDP2."""
        return as_local(self.weight.detach()).numel()

    def _post_training(self, model: PreTrainedModel, name: str) -> None:
        """Bake the latent weight to `codes * s16` in place so every save path emits the master.

        Called by axolotl's post-training module hook before the model is saved. A
        learnable scale is folded into the baked values and then dropped.

        Raises:
            RuntimeError: If the write did not land — a silently unbaked parameter
                would be saved as an ordinary FP checkpoint labelled as a master.
        """
        if self.is_baked():
            return
        self._record_final_lambda(model, name)
        baked = self.baked_weight()
        with torch.no_grad():
            # write through the parameter: rebinding `.data` is a silent no-op on a
            # DTensor, which would leave FP latents behind a `baked` flag
            self.weight.copy_(baked)
        if not bool(torch.equal(as_local(self.weight.detach()), as_local(baked))):
            raise RuntimeError(
                f"ternary: baking {name} did not land in its weight; the checkpoint "
                "would hold latent values under a ternary manifest"
            )
        self._mark_baked()
        if self.scale is not None:
            self.scale = None

    def extra_repr(self) -> str:
        """Return the shape, scale mode and activation bits for `repr(model)`."""
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"weight_scale={self.weight_scale}",
        ]
        if self.group_size is not None:
            parts.append(f"group_size={self.group_size}")
        parts.append(f"activation_bits={self.activation_bits}")
        parts.append(f"lambda={self.lambda_:g}")
        if self.baked:
            parts.append("baked=True")
        return ", ".join(parts)

    def _ops(self, tensor: torch.Tensor) -> ModuleType | None:
        # the fused kernels are CUDA-only and reject sharded tensors; anything else
        # falls back to the eager oracle
        if not self.fused or not tensor.is_cuda or as_dtensor(tensor) is not None:
            return None
        return _fused_ops()

    def _int8_linear(self, x: torch.Tensor) -> torch.Tensor | None:
        ops = _int8_ops()
        out = None if ops is None else ops.int8_linear_forward(self, x)
        if out is None and self.int8_forward is True and not self._int8_warned:
            self._int8_warned = True
            LOG.warning(
                f"ternary: int8_forward is unavailable for a {self.out_features}x"
                f"{self.in_features} {x.dtype} linear; using the fake-quant forward"
            )
        return out

    def _scale(self, gathered: bool = False) -> torch.Tensor | None:
        """Return the fp32 per-tensor scale, floored; `gathered` unshards it first."""
        if self.scale is None:
            return None
        scale = self.scale
        if gathered:
            # a (1,) scale sharded over N ranks leaves most of them an empty local
            sharded = as_dtensor(scale)
            scale = scale if sharded is None else sharded.full_tensor()
        return scale.float().exp().reshape(()).clamp_min(quant.SCALE_EPS)

    def _initial_scale(self) -> torch.Tensor:
        """Return the fp32 scale a learnable `log_s` starts from."""
        weight = as_local(self.weight.detach())
        recovered = quant.baked_codes_and_scale(weight, self.group_size)
        if recovered is not None:
            return recovered[1]
        return quant.absmean_scale(weight, self.group_size)

    def _detect_baked(self) -> bool:
        """Flag the module when its weight already holds exactly `{-s16, 0, +s16}`.

        A baked master is the interchange artifact, so it is what gets loaded back;
        re-deriving an absmean scale from ternary values would shrink every magnitude
        by the non-zero code fraction.
        """
        weight = as_local(self.weight.detach())
        if weight.device.type == "meta" or not weight.numel():
            return False
        self.baked = quant.baked_codes_and_scale(weight, self.group_size) is not None
        self._baked_version = self.weight._version if self.baked else None
        return self.baked

    def _mark_baked(self) -> None:
        self.baked = True
        self._baked_version = self.weight._version

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
        """Re-run the baked probe after `load_state_dict` replaces the weight."""
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._detect_baked()

    def _quant_weight(self, lambda_: float) -> torch.Tensor:
        scale = self._scale()
        # fused kernels carry no gradient for a learnable scale
        ops = self._ops(self.weight) if scale is None else None
        if ops is not None:
            return _FusedQuantSTE.apply(
                self.weight, ops.fake_quant_weight, lambda_, self.group_size, None
            )
        return quant.fake_quant_weight_ste(self.weight, lambda_, self.group_size, scale)

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
