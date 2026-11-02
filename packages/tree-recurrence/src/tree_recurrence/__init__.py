# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Prefix-shared ("tree") packing for recurrent and linear-attention layers.

Samples that share a prefix are packed into one row as a trie, so the prefix is
forwarded once. ``install(model)`` makes a transformers model's recurrent mixers
(Gated DeltaNet, Mamba2, LFM2's short convs) follow that trie, reading it from the
``tree_*`` kwargs a tree-packing trainer such as TRL's ``AsyncGRPOTrainer`` passes;
attention follows it through the trainer's tree attention. The driver
(``tree_conv``, ``tree_scan``) runs any family's kernels on a ``TreePlan``.
"""

from importlib.metadata import PackageNotFoundError, version

from tree_recurrence.driver import (
    DEFAULT_POLICY,
    ConvFn,
    Policy,
    ScanFn,
    padded_scan,
    restore_tail,
    tree_conv,
    tree_scan,
)
from tree_recurrence.integrations.transformers import (
    Support,
    install,
    uninstall,
)
from tree_recurrence.plan import (
    TREE_KWARGS,
    Level,
    TreePlan,
    segment_ids,
    tree_kwargs,
    tree_plan,
)
from tree_recurrence.registry import (
    TreeFamily,
    families,
    follows_trees,
    mark_tree_forward,
    register_family,
    tree_family,
    unregister_family,
)

try:
    __version__ = version("tree-recurrence")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_POLICY",
    "TREE_KWARGS",
    "ConvFn",
    "Level",
    "Policy",
    "ScanFn",
    "Support",
    "TreeFamily",
    "TreePlan",
    "families",
    "follows_trees",
    "install",
    "mark_tree_forward",
    "padded_scan",
    "register_family",
    "restore_tail",
    "segment_ids",
    "tree_conv",
    "tree_family",
    "tree_kwargs",
    "tree_plan",
    "tree_scan",
    "uninstall",
    "unregister_family",
]
