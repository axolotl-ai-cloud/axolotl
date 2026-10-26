# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Which recurrent mixers can follow a tree, and how they are made to.

A ``TreeFamily`` names the recurrent mixer classes of some model types. Its
``install`` puts hooks in place for one modeling module (see
``tree_recurrence.integrations.transformers``); a family without one follows trees
through forwards of its own, marked with ``mark_tree_forward``. A model's own code
(an in-tree model, say) registers its family next to the forward it marks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

_FOLLOWS_ATTR = "__tree_recurrence_follows__"


@dataclass(frozen=True)
class TreeFamily:
    name: str
    model_types: tuple[str, ...]
    # every recurrent mixer class those model types use
    mixers: tuple[str, ...]
    # the config ``layer_types`` those mixers implement
    layer_types: tuple[str, ...] = ()
    # "chains": the mixers keep whole samples apart but cannot follow a trie, so
    # each sample has to be its own chain
    support: Literal["tree", "chains"] = "tree"
    # (modeling module, install context) -> owner name; see integrations.transformers
    install: Callable[..., str] | None = None


_FAMILIES: dict[str, TreeFamily] = {}


def register_family(family: TreeFamily) -> None:
    """Add ``family``, replacing a family of the same name."""
    _FAMILIES[family.name] = family


def unregister_family(name: str) -> None:
    _FAMILIES.pop(name, None)


def families() -> tuple[TreeFamily, ...]:
    return tuple(_FAMILIES.values())


def tree_family(model_types: Iterable[str]) -> TreeFamily | None:
    """The family registered for any of ``model_types``."""
    model_types = set(model_types)
    for family in _FAMILIES.values():
        if model_types.intersection(family.model_types):
            return family
    return None


def mixer_family(mixer_class_name: str) -> TreeFamily | None:
    for family in _FAMILIES.values():
        if mixer_class_name in family.mixers:
            return family
    return None


def mark_tree_forward(forward: Any) -> Any:
    """Mark a mixer forward that follows ``TreePlan``s itself."""
    setattr(forward, _FOLLOWS_ATTR, True)
    return forward


def follows_trees(module: Any) -> bool:
    """Whether ``module``'s class forward is marked with ``mark_tree_forward``."""
    return bool(getattr(type(module).forward, _FOLLOWS_ATTR, False))
