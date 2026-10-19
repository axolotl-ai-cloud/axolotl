"""Scheduling lengths for native packed diffusion batches."""

from __future__ import annotations

__ci_config_keys__ = ("diffusion", "diffusion_lm")

from collections.abc import Sequence
from typing import Any

from axolotl.model_support.native_packing import (  # noqa: F401  (re-exported)
    FLEX_BUCKET_SIZE,
    NativePackingBudget,
    native_packing_options,
    resolve_native_packing_budget,
)
from axolotl.utils.samplers import get_dataset_lengths


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
