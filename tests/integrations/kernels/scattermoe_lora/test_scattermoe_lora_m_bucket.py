# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests for the ``M_BUCKET`` autotune-key bucketing on the scattermoe-lora
fused forward kernel.

The kernel runs on the real ``M`` (loop bounds + masks); only the
``@triton.autotune`` cache key is bucketed via :func:`_bucket_m`. These tests
pin both halves of that contract:

  * ``_bucket_m`` rounds up to a multiple of the granularity (pure-Python
    unit test, no GPU).
  * Two distinct real ``M`` values that share a bucket produce **one**
    cache entry (the whole point — no resweep on small seqlen variation).
  * Two real ``M`` values in different buckets produce **two** cache
    entries (we didn't accidentally collapse to a single key).

Run on CUDA only; the bucketing assertion needs an actual Triton launch.
"""

from __future__ import annotations

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import lora_ops
from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.lora_ops import (
    _M_BUCKET_GRANULARITY,
    _bucket_m,
    scatter2scatter_lora,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)


def test_bucket_m_rounds_up_to_granularity():
    g = _M_BUCKET_GRANULARITY
    assert _bucket_m(1) == g
    assert _bucket_m(g) == g
    assert _bucket_m(g + 1) == 2 * g
    assert _bucket_m(2 * g) == 2 * g
    # Realistic seqlen variation: at granularity=1024 and top_k=8 the three
    # seqlens 16300/16400/16500 straddle one bucket boundary, so they collapse
    # to 2 cache entries rather than 3 (16400 and 16500 share a bucket).
    assert _bucket_m(16400 * 8) == _bucket_m(16500 * 8)
    assert _bucket_m(16300 * 8) != _bucket_m(16400 * 8)
    distinct = {_bucket_m(s * 8) for s in (16300, 16400, 16500)}
    assert len(distinct) == 2


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for kernel launch"
)


_DEVICE = "cuda"
_DTYPE = torch.bfloat16
_E = 4
_K = 64
_N = 64
_TOP_K = 2
_R = 16


def _launch_once(m: int) -> None:
    """One fused fwd launch at the given real M; minimal shapes for speed."""
    torch.manual_seed(m)
    x = torch.randn(m, _K, device=_DEVICE, dtype=_DTYPE)
    W = torch.randn(_E, _K, _N, device=_DEVICE, dtype=_DTYPE) * 0.02
    lora_A = torch.randn(_R * _E, _K, device=_DEVICE, dtype=_DTYPE) * 0.01
    lora_B = torch.randn(_N, _R * _E, device=_DEVICE, dtype=_DTYPE) * 0.01
    logits = torch.randn(m, _E, device=_DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _TOP_K, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, _E)
    scatter2scatter_lora(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        lora_A=lora_A,
        lora_B=lora_B,
        scaling=1.0 / _R,
    )


def test_autotune_cache_collapses_within_bucket_and_grows_across_buckets():
    cache = lora_ops._scatter2scatter_lora.cache
    cache.clear()

    g = _M_BUCKET_GRANULARITY
    # Two M values that both ceil to bucket B1.
    m_a = g - 1
    m_b = g // 2 + 1
    assert _bucket_m(m_a) == g
    assert _bucket_m(m_b) == g

    _launch_once(m_a)
    assert len(cache) == 1, (
        f"first launch should create exactly one cache entry, got {len(cache)}"
    )

    _launch_once(m_b)
    assert len(cache) == 1, (
        f"second launch in the same bucket must not add a cache entry "
        f"(M={m_a} and M={m_b} both bucket to {g}); got {len(cache)} entries"
    )

    # An M strictly past the bucket boundary lands in bucket 2*g.
    m_c = g + 1
    assert _bucket_m(m_c) == 2 * g
    _launch_once(m_c)
    assert len(cache) == 2, (
        f"launch in a different bucket (M={m_c} -> {2 * g}) must add a "
        f"second cache entry; got {len(cache)}"
    )
