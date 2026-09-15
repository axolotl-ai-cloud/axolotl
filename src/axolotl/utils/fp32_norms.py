"""Helpers for keeping selected norm modules in fp32 under FSDP2."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_FP32_NORM_SUFFIXES: tuple[str, ...] = ("RMSNorm", "LayerNorm")


def _matches_norm_class(module: "torch.nn.Module", patterns: Sequence[str]) -> bool:
    """Match a module against class-name patterns.

    Two matching modes, chosen per-pattern by presence of a dot:
      - Fully qualified (contains "."): matches f"{module.__module__}.{cls}" exactly.
      - Suffix (no dot): matches type(module).__name__.endswith(pattern).
    Empty / whitespace-only patterns are skipped (``cls_name.endswith("")``
    is True for every class, which would silently match everything).
    """
    cls = type(module)
    cls_name = cls.__name__
    qualified = f"{cls.__module__}.{cls_name}"
    for pattern in patterns:
        if not pattern or not pattern.strip():
            continue
        if "." in pattern:
            if qualified == pattern:
                return True
        elif cls_name.endswith(pattern):
            return True
    return False


def get_fp32_norm_patterns(source) -> list[str] | None:
    """Resolve configured fp32 norm patterns from a config or tagged model."""
    tagged_patterns = getattr(source, "_axolotl_fp32_norm_patterns", None)
    if tagged_patterns is not None:
        return list(tagged_patterns)

    if not getattr(source, "fp32_norms", False):
        return None

    configured_patterns = getattr(source, "fp32_norm_classes", None)
    if configured_patterns:
        return list(configured_patterns)

    return list(DEFAULT_FP32_NORM_SUFFIXES)


def tag_model_fp32_norms(model: "torch.nn.Module", cfg) -> list[str] | None:
    """Attach the resolved fp32 norm patterns to the model for FSDP2 prepare."""
    patterns = get_fp32_norm_patterns(cfg)
    if patterns is None:
        if hasattr(model, "_axolotl_fp32_norm_patterns"):
            delattr(model, "_axolotl_fp32_norm_patterns")
        return None

    model._axolotl_fp32_norm_patterns = list(patterns)
    return patterns


def shard_norms_fp32(
    model: "torch.nn.Module",
    source=None,
    *,
    patterns: Sequence[str] | None = None,
    fully_shard_kwargs: dict[str, Any] | None = None,
) -> int:
    """Wrap matching norm modules with FSDP2 + fp32 MixedPrecisionPolicy."""
    if source is not None and not getattr(source, "fp32_norms", False):
        return 0

    if source is not None and getattr(source, "fsdp_version", None) != 2:
        raise ValueError(
            "fp32_norms requires fsdp_version: 2. FSDP1 enforces flat-param "
            "dtype uniformity within each wrap group, which is incompatible "
            "with keeping norms in fp32 while the rest of the layer is bf16."
        )

    patterns = (
        list(patterns)
        if patterns is not None
        else get_fp32_norm_patterns(source or model)
    )
    if not patterns:
        return 0

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    outer_policy = (fully_shard_kwargs or {}).get("mp_policy")
    output_dtype = getattr(outer_policy, "param_dtype", None)
    fp32_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=output_dtype,
    )

    matches = [
        (name, module)
        for name, module in model.named_modules()
        if _matches_norm_class(module, patterns)
    ]

    if not matches:
        LOG.warning(
            "fp32_norms enabled but no modules matched patterns %s. Check "
            "fp32_norm_classes against the model's actual norm class names.",
            patterns,
        )
        return 0

    shard_kwargs = dict(fully_shard_kwargs or {})
    shard_kwargs["mp_policy"] = fp32_policy

    for _name, module in matches:
        for param in module.parameters(recurse=False):
            param.data = param.data.to(torch.float32)
        for buffer in module.buffers(recurse=False):
            if buffer.dtype.is_floating_point:
                buffer.data = buffer.data.to(torch.float32)
        fully_shard(module, **shard_kwargs)

    LOG.info(
        "Sharded %d norm modules with fp32 MixedPrecisionPolicy "
        "(patterns=%s, output_dtype=%s)",
        len(matches),
        patterns,
        output_dtype,
    )
    return len(matches)
