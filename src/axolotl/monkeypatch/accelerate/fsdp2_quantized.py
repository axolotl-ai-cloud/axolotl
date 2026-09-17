"""Capability helpers for sharding pre-quantized / mixed-dtype models under FSDP2.

These are only engaged when a model actually carries non-float (quantized) Parameters — e.g. a
pre-quantized NVFP4 checkpoint loaded for LoRA. They are model-agnostic: model families register
the class names of modules that must stay fp32 (e.g. DeepSeek-V4 mHC) via
:func:`register_fp32_shard_classes`. Pure-bf16 models never touch this path.
"""

from __future__ import annotations

import contextlib

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Class names of modules that must be sharded in their own fp32 group (registered by adapters).
_FP32_SHARD_CLASS_NAMES: set[str] = set()

# Tensor-subclass names that are quantized but report a logical FLOAT dtype (so torch.is_floating_point
# is True) — e.g. torchao NVFP4Tensor / Float8Tensor / MXTensor. These still need the quantized
# FSDP2 dtype/sharding policy even though they're "floating point".
_QUANT_TENSOR_CLASS_NAMES: set[str] = {"NVFP4Tensor", "Float8Tensor", "MXTensor"}


def register_fp32_shard_classes(names) -> None:
    """Register module class names that the quantized FSDP2 path keeps in a separate fp32 shard."""
    _FP32_SHARD_CLASS_NAMES.update(names)


def register_quantized_tensor_classes(names) -> None:
    """Register quantized tensor-subclass names (float-logical, e.g. NVFP4Tensor) for detection."""
    _QUANT_TENSOR_CLASS_NAMES.update(names)


def model_has_nonfloat_params(model) -> bool:
    """True iff the model carries PLAIN non-float Parameters (e.g. uint8 packed) — the case that
    needs the nn.Parameter.__new__ requires_grad guard during sharding."""
    return any(not torch.is_floating_point(p) for p in model.parameters())


def _is_quantized_param(p) -> bool:
    if not torch.is_floating_point(p):
        return True  # plain packed (uint8) quantized data
    # torchao tensor subclasses report a logical float dtype; detect by class name (no hard dep).
    if type(p).__name__ in _QUANT_TENSOR_CLASS_NAMES:
        return True
    data = getattr(p, "data", None)
    return data is not None and type(data).__name__ in _QUANT_TENSOR_CLASS_NAMES


def model_has_quantized_params(model) -> bool:
    """Capability check for the quantized dtype/sharding policy: plain non-float params OR float-
    logical quantized tensor subclasses (NVFP4Tensor/Float8Tensor/MXTensor)."""
    return any(_is_quantized_param(p) for p in model.parameters())


def _is_float_logical_quantized_param(p) -> bool:
    """True only for torchao float-logical quantized subclasses (NVFP4Tensor/Float8Tensor/MXTensor)
    — the ones the dtype/cast/sharding policy below is designed for. Deliberately excludes plain
    non-float params (e.g. bnb Params4bit uint8): standard QLoRA+FSDP2 keeps its fp32 LoRA and must
    not be downcast by cast_residual_fp32."""
    if type(p).__name__ in _QUANT_TENSOR_CLASS_NAMES:
        return True
    data = getattr(p, "data", None)
    return data is not None and type(data).__name__ in _QUANT_TENSOR_CLASS_NAMES


def model_has_float_logical_quantized_params(model) -> bool:
    """Capability check for the dtype/cast/sharding policy: ONLY float-logical quantized tensor
    subclasses (the pre-quantized NVFP4/Float8 checkpoint case). Plain bnb Params4bit QLoRA is
    excluded so its fp32 LoRA adapters are not silently cast to the compute dtype."""
    return any(_is_float_logical_quantized_param(p) for p in model.parameters())


@contextlib.contextmanager
def nonfloat_param_guard(model):
    """Freeze existing non-float params and make ``nn.Parameter`` construction default to
    ``requires_grad=False`` for non-float data for the duration of sharding.

    FSDP2's sharded ``nn.Parameter()`` defaults ``requires_grad=True``, which errors on non-float
    (uint8) data *before* it copies the original flag. The ``nn.Parameter.__new__`` patch is a
    PROCESS-GLOBAL monkeypatch, so it is restored in a ``finally`` — an exception during
    ``fully_shard`` can no longer leave the process globally patched.
    """
    import torch.nn as nn

    frozen = 0
    for p in model.parameters():
        if not torch.is_floating_point(p) and p.requires_grad:
            p.requires_grad_(False)
            frozen += 1
    if frozen:
        LOG.info(
            "fsdp2-quant: froze %d non-float (quantized) params before sharding", frozen
        )

    orig_new = nn.Parameter.__new__

    def _nonfloat_safe_new(cls, data=None, requires_grad=True):
        if data is not None and not torch.is_floating_point(data):
            requires_grad = False
        return orig_new(cls, data, requires_grad)

    nn.Parameter.__new__ = _nonfloat_safe_new
    try:
        yield
    finally:
        nn.Parameter.__new__ = orig_new


def shard_fp32_modules(model, fsdp2_kwargs, compute_dtype=torch.bfloat16) -> int:
    """Shard registered keep-fp32 modules (e.g. DSV4 mHC) in their own fp32 group: compute in fp32
    (cast inputs up) but emit ``compute_dtype`` outputs so they don't feed fp32 activations into
    downstream low-precision layers. No-op if no classes are registered."""
    if not _FP32_SHARD_CLASS_NAMES:
        return 0
    from torch.distributed.fsdp import (
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )

    fp32_mp = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=compute_dtype,
        cast_forward_inputs=True,
    )
    kwargs = {**fsdp2_kwargs, "mp_policy": fp32_mp}
    n = 0
    for m in model.modules():
        if type(m).__name__ in _FP32_SHARD_CLASS_NAMES and not isinstance(
            m, FSDPModule
        ):
            fully_shard(m, **kwargs)
            n += 1
    if n:
        LOG.info("fsdp2-quant: fp32-sharded %d keep-fp32 modules separately", n)
    return n


def cast_residual_fp32(model, compute_dtype=torch.bfloat16) -> int:
    """Cast remaining plain fp32 float params (e.g. PEFT-upcast LoRA) to ``compute_dtype`` for a
    uniform per-shard original dtype. Skips DTensors (already-sharded fp32 groups)."""
    from torch.distributed.tensor import DTensor

    n = 0
    for p in model.parameters():
        if p.dtype == torch.float32 and not isinstance(p.data, DTensor):
            p.data = p.data.to(compute_dtype)
            n += 1
    if n:
        LOG.info("fsdp2-quant: cast %d residual fp32 params to %s", n, compute_dtype)
    return n
