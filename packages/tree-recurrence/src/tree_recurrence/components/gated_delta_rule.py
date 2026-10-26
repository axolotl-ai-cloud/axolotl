# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The Gated DeltaNet component (Qwen3.5, Qwen3.5-MoE, Qwen3-Next, OLMo-Hybrid).

FLA's ``chunk_gated_delta_rule`` takes per-sequence initial states with varlen
boundaries and returns the final states, so the driver can fork the delta-rule state
at the trie's branches. transformers' PyTorch reference takes neither, so it runs
every root-to-leaf path as one right-padded batch.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tree_recurrence.driver import ScanFn, padded_scan


def gated_delta_rule_scan(
    chunk_gated_delta_rule: Callable[..., Any],
    *,
    use_qk_l2norm_in_kernel: bool = True,
    varlen: bool = True,
) -> ScanFn:
    """A scan over ``chunk_gated_delta_rule`` (inputs q, k, v, g, beta).

    ``varlen``: the kernel takes ``cu_seqlens`` and per-sequence initial states
    (forkable); otherwise it gets a padded batch and the scan cannot fork.
    """
    if varlen:

        def scan(inputs, cu_seqlens, initial_state):
            query, key, value, g, beta = inputs
            return chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                cu_seqlens=cu_seqlens,
            )

        return scan

    def batched(inputs, initial_state):
        query, key, value, g, beta = inputs
        return chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )[0]

    return padded_scan(batched)
