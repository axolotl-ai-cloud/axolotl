# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU round-trip tests for the FSDP2 sharding substrate on torchao ``Float8Tensor``.

The substrate (``axolotl.monkeypatch.accelerate.float8_fsdp``) teaches a blockwise-FP8
``Float8Tensor`` the aten ops FSDP2 needs to dim-0 shard a frozen weight: ``split`` /
``narrow`` / ``new_zeros`` / ``clone`` / ``detach`` / ``as_strided`` / ``copy_`` / ``view``
plus the ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather`` reconstruct hooks. These are pure
tensor logic (qdata splits by ``s``, the blockwise scale by ``s // block_size[0]``) and are
fully exercisable on CPU - no GPU, no Blackwell, no fused kernel.

Each test simulates the FSDP2 shard -> all-gather -> reconstruct flow and asserts the
reconstructed tensor dequantizes equal to the unsharded original, across several world sizes
including one where the (block-row) shard axis does NOT divide evenly. Covers the B11 data-
independent FSDP2 collective contract for the Float8 substrate.
"""

from __future__ import annotations

import pytest
import torch

torchao_quant = pytest.importorskip("torchao.quantization", reason="torchao required")
Float8Tensor = torchao_quant.Float8Tensor
KernelPreference = pytest.importorskip(
    "torchao.quantization.quantize_.common", reason="torchao required"
).KernelPreference

from axolotl.monkeypatch.accelerate import float8_fsdp  # noqa: E402

float8_fsdp.patch_float8_fsdp()


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_blockwise_float8(N: int, K: int, b0: int, b1: int, seed: int = 0):
    """Build a frozen blockwise-FP8 ``Float8Tensor`` [N,K] with scale [N//b0, K//b1].

    This is the layout float8_fsdp targets (the 128x128-blocked weight torchao's own split
    rejects with ``AssertionError("not yet implemented")``). PerRow/PerTensor checkpoints are
    a special case (b0=1 or b0=N), covered separately via from_hp.
    """
    torch.manual_seed(seed)
    W = torch.randn(N, K, dtype=torch.bfloat16)
    scale = torch.empty(N // b0, K // b1)
    qdata = torch.empty(N, K, dtype=torch.float8_e4m3fn)
    for i in range(N // b0):
        for j in range(K // b1):
            blk = W[i * b0 : (i + 1) * b0, j * b1 : (j + 1) * b1].float()
            s = blk.abs().amax().clamp_min(1e-6) / 448.0
            scale[i, j] = s
            qdata[i * b0 : (i + 1) * b0, j * b1 : (j + 1) * b1] = (blk / s).to(
                torch.float8_e4m3fn
            )
    return Float8Tensor(
        qdata, scale, [b0, b1], None, None, KernelPreference.AUTO, dtype=torch.bfloat16
    )


def _shard_all_gather_reconstruct(ft, world: int):
    """Simulate one FSDP2 dim-0 shard -> all-gather -> reconstruct cycle on CPU.

    Pads dim 0 to a per-rank-equal multiple of block_size[0] (FSDP2 pads uneven shards via
    new_zeros + copy_ into a narrow), chunks into ``world`` shards, runs each shard's
    fsdp_pre_all_gather, concatenates the gathered inner tensors, and reconstructs via
    fsdp_post_all_gather. Returns the reconstructed (unpadded) Float8Tensor.
    """
    N = ft.shape[0]
    blk0 = ft.block_size[0]
    shard = _cdiv(_cdiv(N, blk0), world) * blk0  # multiple of block_size[0]
    padded = shard * world
    if padded != N:
        full = ft.new_zeros([padded, *ft.shape[1:]])
        full.narrow(0, 0, N).copy_(ft)
    else:
        full = ft.clone()
    shards = torch.split(full, shard, 0)
    assert len(shards) == world, (len(shards), world)
    pre = [s.fsdp_pre_all_gather(mesh=None) for s in shards]
    meta = pre[0][1]
    gathered = tuple(
        torch.cat([p[0][k] for p in pre], 0) for k in range(len(pre[0][0]))
    )
    recon, _ = shards[0].fsdp_post_all_gather(gathered, meta, torch.bfloat16)
    return recon.narrow(0, 0, N)


@pytest.mark.parametrize(
    "N,K,b0,b1",
    [(16, 32, 8, 16), (32, 64, 16, 16)],
    ids=["blocked_8x16", "blocked_16x16"],
)
def test_float8_blockwise_split_roundtrips(N, K, b0, b1):
    """split/narrow/slice on a blockwise Float8Tensor dequantize equal to the original slice."""
    ft = _make_blockwise_float8(N, K, b0, b1)
    orig = ft.dequantize()

    parts = torch.split(ft, b0, 0)
    rec = torch.cat([p.dequantize() for p in parts], 0)
    assert torch.equal(rec, orig), "split round-trip diverged"

    nar = torch.narrow(ft, 0, b0, b0)
    assert torch.equal(nar.dequantize(), orig[b0 : 2 * b0]), (
        "narrow round-trip diverged"
    )

    assert torch.equal(ft.clone().dequantize(), orig), "clone diverged"
    assert torch.equal(ft.detach().dequantize(), orig), "detach diverged"


@pytest.mark.parametrize(
    "N,K,b0,b1,worlds",
    [
        (16, 32, 8, 16, (1, 2)),  # 2 block-rows, even splits
        (24, 32, 8, 16, (1, 2, 3)),  # 3 block-rows; world=2 -> uneven (padded)
    ],
    ids=["even", "uneven_padded"],
)
def test_float8_fsdp_all_gather_roundtrip(N, K, b0, b1, worlds):
    """FSDP2 shard -> all-gather -> reconstruct round-trips for blockwise Float8 across worlds.

    Includes a world size where the block-row axis does NOT divide evenly (the shard is padded
    on dim 0 via new_zeros/copy_ and narrowed back). Pure CPU tensor logic; no GPU needed.
    """
    ft = _make_blockwise_float8(N, K, b0, b1)
    orig = ft.dequantize()
    for world in worlds:
        recon = _shard_all_gather_reconstruct(ft, world)
        assert torch.equal(recon.dequantize(), orig), (
            f"FSDP all-gather reconstruct diverged at world={world}"
        )


def test_float8_perrow_split_roundtrips():
    """A PerRow Float8Tensor (block_size [1, K], the common from_hp checkpoint) shards on dim 0."""
    torch.manual_seed(1)
    W = torch.randn(16, 32, dtype=torch.bfloat16)
    ft = Float8Tensor.from_hp(W)
    assert ft.block_size[0] == 1, ft.block_size
    orig = ft.dequantize()

    parts = torch.split(ft, 4, 0)
    rec = torch.cat([p.dequantize() for p in parts], 0)
    assert torch.equal(rec, orig), "PerRow split round-trip diverged"

    recon = _shard_all_gather_reconstruct(ft, 3)  # 16 rows / 3 -> uneven, padded
    assert torch.equal(recon.dequantize(), orig), "PerRow uneven all-gather diverged"


def test_float8_new_zeros_scale_is_coarser_by_block_size():
    """new_zeros must scale the inner scale by block_size on every dim (FSDP pad invariant)."""
    ft = _make_blockwise_float8(16, 32, 8, 16)
    nz = ft.new_zeros([24, 32])
    assert nz.qdata.shape == (24, 32)
    assert nz.scale.shape == (24 // 8, 32 // 16)


def test_float8_split_dim_other_than_zero_rejected():
    """Sharding axis is dim 0 only; a non-zero dim split must raise (documents the restriction)."""
    ft = _make_blockwise_float8(16, 32, 8, 16)
    with pytest.raises(NotImplementedError, match="only on dim 0"):
        torch.split(ft, 16, 1)
