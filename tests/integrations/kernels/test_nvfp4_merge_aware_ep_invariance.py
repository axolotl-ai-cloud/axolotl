# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Expert-parallelism invariance of the merge-aware NVFP4 snap.

Expert parallelism shards the ``[E, N, K]`` expert stack on dim 0 (each rank owns
a contiguous ``[E_local]`` slice; see ``_scatter_expert_from_rank0``). The
merge-aware fresh-grid quantizer snaps every expert independently on its own
per-expert ``pts`` grid, with the E2M1 blocks living on the last dim, so a dim-0
slice must snap to the exact same bytes it would inside the full tensor. Since the
shards tile ``[0, E)``, per-shard equality also means the gathered merge is
byte-for-byte the checkpoint a single GPU would write: what a rank trains against
is what the merge writes, independent of ``ep_size``.

Runs entirely on CPU (the off-CUDA path is torchao's reference quantizer, which
``fake_quant_nvfp4_dispatch`` falls back to).
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
    fake_quant_nvfp4_dispatch,
    quantize_nvfp4_merge,
)

E, N, K = 8, 48, 96  # E divisible by the ep sizes below; K % 16 == 0


def _pts(kind):
    if kind == "none":
        return None
    if kind == "scalar":
        return torch.tensor(0.7)
    return (torch.rand(E) * 0.5 + 0.5).float()  # per-expert weight_scale_2


@pytest.mark.parametrize("pts_kind", ["none", "scalar", "perexpert"])
@pytest.mark.parametrize("ep_size", [2, 4])
def test_snap_commutes_with_expert_sharding(pts_kind, ep_size):
    """A per-rank ``[E_local]`` snap == the global snap's slice, bit-identical."""
    torch.manual_seed(0)
    w = (torch.randn(E, N, K) * 0.05).bfloat16()
    pts = _pts(pts_kind)

    packed_full, scale_full = quantize_nvfp4_merge(w, pts, scale_mode="fresh")
    deq_full = fake_quant_nvfp4_dispatch(w, pts)

    step = E // ep_size
    for r in range(ep_size):
        lo, hi = r * step, (r + 1) * step
        spts = pts if (pts is None or pts.numel() == 1) else pts[lo:hi].contiguous()
        packed_s, scale_s = quantize_nvfp4_merge(
            w[lo:hi].contiguous(), spts, scale_mode="fresh"
        )
        deq_s = fake_quant_nvfp4_dispatch(w[lo:hi].contiguous(), spts)

        assert torch.equal(packed_s, packed_full[lo:hi]), "qdata bytes diverged"
        assert torch.equal(scale_s, scale_full[lo:hi]), "block-scale bytes diverged"
        assert torch.equal(deq_s, deq_full[lo:hi]), "dequant values diverged"
