"""Runtime resolution for learned decision-slot embedding rows."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

from axolotl.model_support import DiffusionSpec

from ._util import (
    _value,
    active_control_ids,
    optional_token_id,
    parse_token_ids,
    require_diffusion_spec,
)
from .slots import SlotInit, SlotPlan


@dataclass(frozen=True)
class TrainableSlotRuntime:
    """Validated learned/prompt slot rows and their actual input-embedding path."""

    plan: SlotPlan
    embedding_module_path: str


def _set_value(value: Any, name: str, item: Any) -> None:
    if isinstance(value, MutableMapping):
        value[name] = item
    elif isinstance(value, Mapping):
        raise TypeError("decision slot configuration mapping must be mutable")
    else:
        setattr(value, name, item)


def _embedding_path(model: Any, embedding: Any) -> str:
    if not hasattr(model, "named_modules"):
        raise TypeError("model must expose named_modules for trainable slot rows")
    matches = [name for name, module in model.named_modules() if module is embedding]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(
            "could not determine one non-root input embedding module path for "
            "peft_trainable_token_indices"
        )
    return matches[0]


def resolve_trainable_slot_runtime(
    cfg: Any,
    model: Any,
    decision: Any,
    *,
    spec: DiffusionSpec | None = None,
) -> TrainableSlotRuntime | None:
    """Resolve learned/prompt rows from config and the model's actual embedding."""
    latent = _value(decision, "latent")
    if latent is None:
        raise ValueError("diffusion_decision requires latent settings")
    mode = _value(latent, "mode", "none")
    if mode not in {"learned", "prompt"}:
        return None
    if not hasattr(model, "get_input_embeddings"):
        raise TypeError(
            "model must expose get_input_embeddings for trainable slot rows"
        )
    embedding = model.get_input_embeddings()
    vocab_size = _value(embedding, "num_embeddings")
    if (
        isinstance(vocab_size, bool)
        or not isinstance(vocab_size, int)
        or vocab_size < 1
    ):
        raise ValueError("model input embedding must expose a positive num_embeddings")
    config = _value(model, "config")
    if config is None:
        raise TypeError("model must expose config for trainable slot rows")
    mask_token_id = optional_token_id(_value(config, "mask_token_id"), "mask_token_id")
    pad_token_id = optional_token_id(_value(config, "pad_token_id"), "pad_token_id")
    control_ids = active_control_ids(config)
    invalid_control_ids = sorted(
        token_id for token_id in control_ids if token_id >= vocab_size
    )
    if invalid_control_ids:
        raise ValueError(
            "model config control IDs must be within the input embedding vocabulary: "
            f"{invalid_control_ids}"
        )
    token_ids = parse_token_ids(
        _value(latent, "token_ids", ()) or (), "latent.token_ids"
    )
    if set(token_ids) & control_ids:
        raise ValueError(
            "learned/prompt slot token_ids cannot use active pad, bos, eos, or mask IDs"
        )
    plan = SlotInit(
        mode=mode,
        token_ids=token_ids,
        num_slots=_value(latent, "num_slots", 0),
        vocab_size=vocab_size,
        pad_id=pad_token_id,
        spec=require_diffusion_spec(cfg) if spec is None else spec,
        mask_token_id=mask_token_id,
    ).build()
    return TrainableSlotRuntime(plan, _embedding_path(model, embedding))


def merge_trainable_slot_indices(cfg: Any, runtime: TrainableSlotRuntime) -> None:
    """Merge learned/prompt rows into PEFT's embedding-row configuration."""
    configured = _value(cfg, "peft_trainable_token_indices")
    additions = list(runtime.plan.trainable_token_ids)
    if not additions:
        return
    if configured is None:
        _set_value(cfg, "peft_trainable_token_indices", additions)
        return
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        _set_value(
            cfg,
            "peft_trainable_token_indices",
            _merged_indices(configured, additions),
        )
        return
    if isinstance(configured, Mapping):
        existing = configured.get(runtime.embedding_module_path)
        if existing is None:
            raise ValueError(
                "peft_trainable_token_indices mapping must include the actual input "
                f"embedding path {runtime.embedding_module_path!r} before learned/prompt "
                "decision slots can be merged"
            )
        if not isinstance(existing, Sequence) or isinstance(existing, (str, bytes)):
            raise ValueError(
                "peft_trainable_token_indices mapping values must be integer sequences"
            )
        updated = dict(configured)
        updated[runtime.embedding_module_path] = _merged_indices(existing, additions)
        _set_value(cfg, "peft_trainable_token_indices", updated)
        return
    raise ValueError(
        "peft_trainable_token_indices must be an integer sequence or an embedding-path mapping"
    )


def _merged_indices(existing: Sequence[Any], additions: Sequence[int]) -> list[int]:
    result: list[int] = []
    for value in (*existing, *additions):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "peft_trainable_token_indices must contain nonnegative integers"
            )
        if value not in result:
            result.append(value)
    return result
