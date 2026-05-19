# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Reproducer for the cuBLAS / illegal-memory-access failure surfaced at
seq=512K with 16 shards in the tiled-MLP long-context bench.

The originally-reported symptom (``CUBLAS_STATUS_EXECUTION_FAILED`` from
``cublasGemmStridedBatchedEx`` at
``parallel_experts.py:72``'s ``gates.unsqueeze(1) @ output_expanded``)
is a downstream effect — the actual fault is in the upstream
``scatter2scatter`` Triton kernel's pointer-offset arithmetic. When the
output of the up-projection has
``L_scattered * y_dim >= 2 ** 31`` elements (i.e. the kernel's
``M_block * stride_ym`` int32 multiplication overflows), the kernel
silently writes to wrong addresses, which can in turn trip the next
kernel (the gates @ output_expanded bmm or whatever else follows).

The repro shape mirrors the failing bench config:

* shard tokens ``T = 32768`` (= 524288 // 16),
* ``top_k = 8`` → ``L_scattered = T * top_k = 262144``,
* ``num_experts = 128``, ``hidden = 2048``, ``intermediate = 8192``.

At that shape, the up-projection's scatter2scatter output is
``[262144, 2 * 8192] = [262144, 16384]`` = 2**32 elements. The
overflow boundary for the M_block * stride_ym int32 product is
``M_block < 2 ** 31 / 16384 = 131072`` — exactly half the output rows.
"""

from __future__ import annotations

import pytest
import torch

# Failing-config constants (mirror tests/integrations/monkeypatch/
# bench_tiled_mlp_moe.py at seq=524288, shards=16).
_T = 32768
_TOP_K = 8
_NUM_EXPERTS = 128
_HIDDEN = 2048
_INTERMEDIATE = 8192
_DTYPE = torch.bfloat16


def _requires_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA required for the repro"
    )


_SCATTER2SCATTER_INT32_LIMIT = 2**31


@_requires_cuda()
def test_scatter2scatter_below_threshold_no_overhead():
    """At shapes well below the int32 overflow boundary the auto-dispatch
    in ``ParallelLinear`` picks ``INT64_INDICES=False`` and the kernel
    output is bit-identical to a direct call with the same flag.

    This is the regression guard for "don't accidentally penalise the
    common-case path that does not need int64 indices".
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
        scatter2scatter,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        flatten_sort_count,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # Small shape: T=512 tokens → L_scattered=4096 → output is
    # 4096 * 16384 = 67 M elements, well below the 2**31 threshold.
    T_small = 512
    x = torch.randn(T_small, _HIDDEN, device=device, dtype=_DTYPE)
    W = (
        torch.randn(
            _NUM_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device, dtype=_DTYPE
        )
        * 0.01
    )

    logits = torch.randn(T_small, _NUM_EXPERTS, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _TOP_K, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, _NUM_EXPERTS)

    assert sei.size(0) * W.size(-1) < _SCATTER2SCATTER_INT32_LIMIT

    out_i32 = scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=False,
    )
    out_i64 = scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_i32, out_i64), (
        "INT64_INDICES must not change MMA/accumulation order at small shapes"
    )


@_requires_cuda()
def test_scatter2scatter_no_corruption_at_overflow_shape():
    """The kernel-level int64 fix must keep every output row populated
    when the shape straddles the 2**31-element boundary.

    Background: with INT64_INDICES=False the Triton ``scatter2scatter``
    kernel computes pointer offsets as
    ``Y_ptr + M_block * stride_ym + N_block * stride_yn`` in int32. At
    the bench shape (L_scattered=262144, y_dim=16384 → 2**32 elements
    of output) the trailing rows past ``M_block >= 2**31 / y_dim``
    overflow and their masked stores silently drop, leaving those rows
    as all-zeros. With INT64_INDICES=True the M_block range is cast to
    int64 before it enters the multiplication and the overflow is
    eliminated at the kernel level.

    This test calls the kernel directly with INT64_INDICES=True and
    asserts every sampled row past the boundary has at least one
    non-zero element. (The ``ParallelLinear`` wrapper's auto-dispatch
    is covered separately by ``test_parallel_linear_long_seq_routing_combination``.)
    """
    from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
        scatter2scatter,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        flatten_sort_count,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    x = torch.randn(_T, _HIDDEN, device=device, dtype=_DTYPE)
    W = (
        torch.randn(
            _NUM_EXPERTS, _HIDDEN, 2 * _INTERMEDIATE, device=device, dtype=_DTYPE
        )
        * 0.01
    )

    logits = torch.randn(_T, _NUM_EXPERTS, device=device)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), _TOP_K, dim=-1)
    sei, ssi, _ = flatten_sort_count(top_idx, _NUM_EXPERTS)

    L_scattered = sei.size(0)
    y_dim = W.size(-1)
    assert L_scattered * y_dim >= _SCATTER2SCATTER_INT32_LIMIT, (
        f"repro precondition: L_scattered * y_dim ({L_scattered * y_dim}) "
        f"must straddle the int32 overflow boundary "
        f"({_SCATTER2SCATTER_INT32_LIMIT})"
    )

    output = scatter2scatter(
        X=x,
        W=W,
        sorted_expert_idxs=sei,
        sorted_scattered_idxs=ssi,
        k=_TOP_K,
        x_grouped=False,
        y_grouped=True,
        int64_indices=True,
    )
    torch.cuda.synchronize()

    overflow_threshold_row = _SCATTER2SCATTER_INT32_LIMIT // y_dim
    sample_rows = [
        0,
        overflow_threshold_row // 2,
        overflow_threshold_row - 1,
        overflow_threshold_row,
        overflow_threshold_row + 1,
        (overflow_threshold_row + L_scattered) // 2,
        L_scattered - 1,
    ]
    for row in sample_rows:
        nz = (output[row] != 0).any().item()
        assert nz, (
            f"row {row} of scatter2scatter output is all-zero "
            f"(overflow_threshold_row={overflow_threshold_row}, "
            f"L_scattered={L_scattered}, y_dim={y_dim})"
        )


@_requires_cuda()
def test_parallel_linear_long_seq_routing_combination():
    """End-to-end repro through ``parallel_linear`` matching the bench path.

    Replicates the ``ScatterMoEGatedMLP.forward`` shape sequence (up
    projection at line 374 → activation → down projection at line 385
    with ``gates=routing_weights``) at the seq=524288/shards=16 inner
    config. Before the fix this raises
    ``CUBLAS_STATUS_EXECUTION_FAILED`` (or a subsequent illegal-memory-
    access) at the down-projection's ``gates @ output_expanded`` bmm.
    """
    import torch.nn.functional as F

    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        flatten_sort_count,
        parallel_linear,
    )

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    layer_input = torch.randn(_T, _HIDDEN, device=device, dtype=_DTYPE)
    # Match the bench's ScatterMoEGatedMLP weight layout: input_linear
    # is [E, 2*INTERMEDIATE, HIDDEN] then .transpose(2, 1) →
    # [E, HIDDEN, 2*INTERMEDIATE]. output_linear is [E, HIDDEN,
    # INTERMEDIATE] then .transpose(2, 1) → [E, INTERMEDIATE, HIDDEN].
    in_w = (
        torch.randn(
            _NUM_EXPERTS, 2 * _INTERMEDIATE, _HIDDEN, device=device, dtype=_DTYPE
        )
        * 0.02
    )
    out_w = (
        torch.randn(_NUM_EXPERTS, _HIDDEN, _INTERMEDIATE, device=device, dtype=_DTYPE)
        * 0.02
    )

    router_logits = torch.randn(_T, _NUM_EXPERTS, device=device)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, _TOP_K, dim=-1)
    routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(_DTYPE)

    sei, ssi, eo = flatten_sort_count(selected_experts, _NUM_EXPERTS)

    # Up projection — this is the overflow-prone call (output
    # numel = T * top_k * 2*INTERMEDIATE = 2**32 at the bench shape).
    with torch.no_grad():
        gup = parallel_linear(
            layer_input,
            in_w.transpose(2, 1),
            _TOP_K,
            sei,
            ssi,
            eo,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gup.chunk(2, dim=-1)
        h = F.silu(gates) * h

        # Down projection — its gates @ output_expanded bmm is where
        # the reported CUBLAS_STATUS_EXECUTION_FAILED surfaces. The
        # crash, however, is a downstream symptom of the up-projection
        # corruption above.
        layer_output = parallel_linear(
            h,
            out_w.transpose(2, 1),
            1,
            sei,
            ssi,
            eo,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )
        # Force the (otherwise lazy) CUDA error to surface synchronously.
        torch.cuda.synchronize()

    assert layer_output.shape == (_T, _HIDDEN)
    # The output must have real values, not zero rows or NaN/Inf.
    # ``(.abs().sum(dim=-1) == 0)`` would catch the silent-zero
    # corruption pattern even when the kernel did not crash hard.
    assert torch.isfinite(layer_output).all().item(), (
        "layer_output has non-finite values — likely overflow corruption"
    )
    row_sums = layer_output.float().abs().sum(dim=-1)
    assert (row_sums > 0).all().item(), (
        "layer_output has all-zero rows — silent overflow corruption "
        "in the up-projection scatter2scatter"
    )
