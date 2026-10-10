"""Native diffusion packed-row token budgets.

Kept outside ``axolotl.core`` so light callers (decision canvas building for
serving) can resolve the budget without importing the trainer stack;
``axolotl.core.trainers.diffusion_lm.sampling`` re-exports these names.
"""

from __future__ import annotations

__ci_config_keys__ = ("diffusion", "diffusion_lm")

from dataclasses import dataclass
from typing import Any

FLEX_BUCKET_SIZE = 128


@dataclass(frozen=True)
class NativePackingBudget:
    """Logical payload capacity derived from one native packed-row budget."""

    total: int
    payload_capacity: int
    layout: str
    bucket_size: int | None
    reserved_capacity: int
    options: dict[str, Any]


def resolve_native_packing_budget(
    cfg: Any,
    *,
    packed: bool | None = None,
    batch_size: int | None = None,
    is_eval: bool = False,
) -> NativePackingBudget | None:
    """Resolve the logical capacity which cannot exceed native physical allocation."""

    options = native_packing_options(cfg)
    if options is None:
        return None
    if packed is None:
        packed = bool(getattr(cfg, "sample_packing", False))
    if not packed:
        return None
    if (
        batch_size is None
        and is_eval
        and not (
            getattr(cfg, "sample_packing", False)
            and getattr(cfg, "eval_sample_packing", None) is not False
        )
    ):
        batch_size = getattr(cfg, "eval_batch_size", None)
    if batch_size is None:
        batch_size = getattr(cfg, "micro_batch_size", None)
    total = int(cfg.sequence_len) * int(batch_size or 1)
    layout = options["layout"]
    if getattr(cfg, "attn_implementation", None) != "flex_attention":
        return NativePackingBudget(total, total, layout, None, 0, options)
    rounded_total = total // FLEX_BUCKET_SIZE * FLEX_BUCKET_SIZE
    reserved_capacity = FLEX_BUCKET_SIZE if layout == "encoder_canvas" else 0
    payload_capacity = rounded_total - reserved_capacity
    if payload_capacity <= 0:
        minimum = (
            2 * FLEX_BUCKET_SIZE if layout == "encoder_canvas" else FLEX_BUCKET_SIZE
        )
        raise ValueError(
            "native Flex packed-row budget is too small for "
            f"{layout}: got {total}, require at least {minimum} tokens"
        )
    return NativePackingBudget(
        total, payload_capacity, layout, FLEX_BUCKET_SIZE, reserved_capacity, options
    )


def native_packing_options(cfg: Any) -> dict[str, Any] | None:
    """Resolve native collation costs for preprocessing and runtime samplers."""
    from axolotl.model_support import (
        get_model_support,
        is_native_diffusion,
        resolve_model_support,
    )
    from axolotl.utils.dict import DictDefault

    if not is_native_diffusion(cfg):
        return None
    config = cfg.diffusion_lm
    if isinstance(config, dict):
        config = DictDefault(config)
    support = resolve_model_support(get_model_support(cfg.model_config_type))
    if support is None or support.diffusion is None:
        return None
    spec = support.diffusion
    return {
        "layout": spec.layout.value,
        "canvas_width": getattr(config, "canvas_width", None) or spec.max_canvas,
        "eos_tail": getattr(config, "eos_tail", None),
        "logical_sequence_length": int(cfg.sequence_len),
    }
