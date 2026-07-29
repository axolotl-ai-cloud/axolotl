"""
Deterministic per-example mixing of chat-template kwargs.

Lets one dataset be rendered under several template settings (for example Qwen3-style
``enable_thinking`` on and off) at chosen proportions, so a hybrid-mode student sees both
renderings. Everything downstream consumes the rendered token ids, so any teacher --
online HTTP or in-process -- scores exactly what the student was rendered with.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

# 8 bytes of digest, read as a fraction of 2**64
_DIGEST_BYTES = 8
_DRAW_SCALE = float(1 << (8 * _DIGEST_BYTES))


@dataclass(frozen=True)
class ChatTemplateKwargsMixEntry:
    """One rendering mode and its (unnormalized) share of the dataset."""

    kwargs: Mapping[str, Any]
    weight: float


def _coerce_entry(entry: Any, position: int) -> ChatTemplateKwargsMixEntry:
    if isinstance(entry, ChatTemplateKwargsMixEntry):
        kwargs: Any = entry.kwargs
        weight: Any = entry.weight
    elif isinstance(entry, Mapping):
        # DictDefault-style mappings answer any attribute with None, so mapping access
        # has to be tried before duck-typing a pydantic model
        kwargs = entry.get("kwargs")
        weight = entry.get("weight")
    elif callable(getattr(entry, "model_dump", None)):
        dumped = entry.model_dump(by_alias=True)
        kwargs = dumped.get("kwargs")
        weight = dumped.get("weight")
    else:
        raise ValueError(
            f"chat_template_kwargs_mix[{position}] must be a mapping with 'kwargs' and "
            f"'weight', got {type(entry).__name__}"
        )

    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(
            f"chat_template_kwargs_mix[{position}].kwargs must be a mapping, got "
            f"{type(kwargs).__name__}"
        )

    if weight is None:
        weight = 1.0
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise ValueError(
            f"chat_template_kwargs_mix[{position}].weight must be a number, got "
            f"{type(weight).__name__}"
        )
    if weight < 0:
        raise ValueError(
            f"chat_template_kwargs_mix[{position}].weight must be non-negative, got {weight}"
        )

    return ChatTemplateKwargsMixEntry(kwargs=dict(kwargs), weight=float(weight))


class ChatTemplateKwargsMix:
    """
    Assigns each example one set of chat-template kwargs.

    The choice is a pure function of ``(seed, index)``, so it is identical across runs,
    resumes, and any ``num_proc`` sharding of ``dataset.map`` -- there is no RNG state to
    carry and no dependence on how the dataset was split up.
    """

    def __init__(self, entries: Iterable[Any], seed: int = 0):
        coerced = [_coerce_entry(entry, i) for i, entry in enumerate(entries or [])]
        if not coerced:
            raise ValueError("chat_template_kwargs_mix must have at least one entry")

        total = sum(entry.weight for entry in coerced)
        if total <= 0:
            raise ValueError("chat_template_kwargs_mix weights must sum to more than 0")

        self.entries: Sequence[ChatTemplateKwargsMixEntry] = [
            ChatTemplateKwargsMixEntry(kwargs=entry.kwargs, weight=entry.weight / total)
            for entry in coerced
        ]
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def weights(self) -> list[float]:
        return [entry.weight for entry in self.entries]

    def draw(self, index: int) -> float:
        digest = hashlib.blake2b(
            f"{self.seed}:{int(index)}".encode(), digest_size=_DIGEST_BYTES
        ).digest()
        return int.from_bytes(digest, "big") / _DRAW_SCALE

    def entry_for_index(self, index: int) -> ChatTemplateKwargsMixEntry:
        draw = self.draw(index)
        cumulative = 0.0
        for entry in self.entries:
            cumulative += entry.weight
            if draw < cumulative:
                return entry
        return self.entries[-1]

    def kwargs_for_index(self, index: int) -> dict[str, Any]:
        return dict(self.entry_for_index(index).kwargs)


def build_chat_template_kwargs_mix(
    entries: Any, seed: int = 0
) -> ChatTemplateKwargsMix | None:
    """Build a mix from raw config, or None when no mix is configured."""
    if not entries:
        return None
    if isinstance(entries, ChatTemplateKwargsMix):
        return entries
    return ChatTemplateKwargsMix(entries, seed=seed)
