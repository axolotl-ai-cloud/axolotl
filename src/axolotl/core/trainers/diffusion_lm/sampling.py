"""Scheduling lengths for native packed diffusion batches."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion", "diffusion_lm")

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from axolotl.utils.samplers import get_dataset_lengths

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


def native_packing_lengths(
    dataset: Any,
    *,
    eos_tail: str | None,
    logical_sequence_length: int | None,
    layout: str = "full_sequence",
    canvas_width: int | None = None,
) -> Sequence[int]:
    """Return the post-collation length charged by the multipack sampler."""
    lengths = (
        get_dataset_lengths(dataset)
        if hasattr(dataset, "column_names")
        else [len(record["input_ids"]) for record in dataset]
    )
    if eos_tail == "visible_supervised" and logical_sequence_length is not None:
        if any(length > logical_sequence_length for length in lengths):
            raise ValueError(
                "diffusion logical example exceeds configured logical length"
            )
        lengths = [logical_sequence_length] * len(lengths)
    if layout == "full_sequence":
        return lengths
    if layout != "encoder_canvas" or canvas_width is None:
        raise ValueError("encoder_canvas scheduling requires canvas_width")
    costs: list[int] = []
    for index, encoder_length in enumerate(lengths):
        record = dataset[index]
        supervised = [
            position for position, label in enumerate(record["labels"]) if label != -100
        ]
        if encoder_length > len(record["input_ids"]):
            supervised.extend(range(len(record["input_ids"]), encoder_length))
        if not supervised:
            raise ValueError("encoder_canvas scheduling requires supervised tokens")
        span = supervised[-1] - supervised[0] + 1
        costs.append(encoder_length + min(canvas_width, span))
    return costs


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


def filter_native_diffusion_dataset(cfg: Any, dataset: Any, *, split: str) -> Any:
    """Apply per-example overflow policy before scheduling optimizer steps."""
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.logging import get_logger

    options = native_packing_options(cfg)
    if options is None or dataset is None:
        return dataset
    config = DictDefault(cfg.diffusion_lm)
    logical_limit = options["logical_sequence_length"]
    packed = cfg.sample_packing and (
        split == "train" or cfg.eval_sample_packing is not False
    )
    packing_budget = resolve_native_packing_budget(
        cfg, packed=packed, is_eval=split == "eval"
    )
    physical_limit = None if packing_budget is None else packing_budget.payload_capacity
    kept = []
    rejected = []
    for index, record in enumerate(dataset):
        oversized = (
            logical_limit is not None and len(record["input_ids"]) > logical_limit
        )
        if not oversized and physical_limit is not None:
            oversized = native_packing_lengths([record], **options)[0] > physical_limit
        if oversized:
            rejected.append(index)
        else:
            kept.append(index)
    if not rejected:
        return dataset
    if config.overflow_policy != "drop":
        raise ValueError(
            f"native diffusion {split} example {rejected[0]} exceeds its logical "
            f"or packed token budget ({len(rejected)} oversized examples)"
        )
    get_logger(__name__).warning(
        "Dropped %s/%s native diffusion %s examples exceeding token budgets",
        len(rejected),
        len(dataset),
        split,
    )
    if not kept:
        raise ValueError(f"all native diffusion {split} examples exceed token budgets")
    return dataset.select(kept)
