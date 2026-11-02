# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The Kimi Delta Attention component (Kimi-Linear, Bailing hybrid).

FLA's ``chunk_kda`` takes per-sequence initial states with varlen boundaries and
returns the final states, so the driver can fork the KDA state at the trie's
branches; the q/k/v short convs run with each segment's path history. The models
using it follow trees through their own forwards and register their families.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tree_recurrence.driver import ConvFn, ScanFn


def kda_scan(
    chunk_kda: Callable[..., Any], *, use_qk_l2norm_in_kernel: bool = True
) -> ScanFn:
    """A forkable scan over FLA's ``chunk_kda`` (inputs q, k, v, g, beta)."""

    def scan(inputs, cu_seqlens, initial_state):
        q, k, v, g, beta = inputs
        return chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )

    return scan


def short_convolution(conv: Callable[..., Any]) -> ConvFn:
    """FLA's ``ShortConvolution`` module (``conv(x=[B, T, D], cu_seqlens=...)``,
    returning the output and a cache)."""

    def run(x, cu_seqlens):
        return conv(x=x, cu_seqlens=cu_seqlens)[0]

    return run
