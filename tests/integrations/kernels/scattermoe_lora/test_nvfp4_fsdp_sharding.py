# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU round-trip tests for the FSDP2 sharding substrate on torchao ``NVFP4Tensor``.

The substrate (``...scattermoe_lora.nvfp4_fsdp``) teaches a frozen, expert-sharded
``NVFP4Tensor`` the aten ops FSDP2 needs to dim-0 (expert axis) shard the weight:
``split`` / ``narrow`` / ``slice`` / ``new_zeros`` / ``clone`` / ``detach`` /
``as_strided`` / ``view`` / ``copy_`` plus the ``fsdp_pre_all_gather`` /
``fsdp_post_all_gather`` reconstruct hooks. These are pure tensor logic over the inner
``(qdata, scale, per_tensor_scale)`` triple and are fully exercisable on CPU - no GPU, no
Blackwell, no fused kernel.

Each round-trip test simulates the FSDP2 shard -> all-gather -> reconstruct flow and asserts
the reconstructed tensor dequantizes equal to the unsharded original across several world
sizes including one where the expert axis does NOT divide evenly. Covers the B11 data-
independent FSDP2 collective contract for the NVFP4 substrate.

Round-trip verification uses a unit (scalar) per_tensor_scale - the real DSV4 frozen-expert
case and the only one torchao's ``NVFP4Tensor.dequantize`` supports against per-block scales.
A per-expert per_tensor_scale buffer is checked at the tensor-shape level (its split/narrow
slicing) since torchao cannot dequantize that shape for an equality check.

Skip gate: requires torchao's NVFP4Tensor. The substrate module is loaded by file path so the
scattermoe_lora package __init__ (which imports triton) is NOT executed - the ops are pure CPU
tensor logic, so this runs on the no-CUDA / triton-less CI lane (matching the Float8 sibling).
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

NVFP4Tensor = pytest.importorskip(
    "torchao.prototype.mx_formats.nvfp4_tensor", reason="torchao required"
).NVFP4Tensor


def _load_nvfp4_fsdp():
    """Load nvfp4_fsdp by file path so the scattermoe_lora package __init__ (which imports triton) is
    NOT executed. The module needs only torch + torchao, so it imports triton-free and runs on CPU."""
    import axolotl

    path = os.path.join(
        os.path.dirname(axolotl.__file__),
        "integrations",
        "kernels",
        "libs",
        "scattermoe_lora",
        "nvfp4_fsdp.py",
    )
    spec = importlib.util.spec_from_file_location(
        "_axolotl_nvfp4_fsdp_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nvfp4_fsdp = _load_nvfp4_fsdp()
nvfp4_fsdp.patch_nvfp4_fsdp()

DEV = "cpu"


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_nvfp4(E: int, N: int, K: int, seed: int = 0, per_expert_pts: bool = False):
    """Build a frozen ``NVFP4Tensor`` [E,N,K] (qdata [E,N,K//2], scale [E,N,K//16]).

    per_expert_pts=False uses a unit scalar per_tensor_scale (the real DSV4 case, dequant-able);
    True uses a per-expert [E] buffer to exercise the per-expert pts split path at the shape level.
    """
    torch.manual_seed(seed)
    packed = torch.randint(0, 255, (E, N, K // 2), dtype=torch.uint8)
    scale = (torch.rand(E, N, K // 16) * 0.5 + 0.5).to(torch.float8_e4m3fn)
    pts = (torch.rand(E) * 0.5 + 0.5) if per_expert_pts else torch.ones(())
    return NVFP4Tensor(packed.contiguous(), scale.contiguous(), 16, torch.bfloat16, pts)


def _shard_all_gather_reconstruct(nv, world: int):
    """Simulate one FSDP2 dim-0 (expert axis) shard -> all-gather -> reconstruct cycle on CPU.

    Pads the expert axis to a per-rank-equal count (FSDP2 pads uneven shards via new_zeros +
    copy_ into a narrow), chunks into ``world`` shards, runs each shard's fsdp_pre_all_gather,
    concatenates the gathered inner tensors (a scalar per_tensor_scale is replicated, not
    concatenated), and reconstructs via fsdp_post_all_gather. Returns the unpadded NVFP4Tensor.
    """
    E = nv.shape[0]
    shard = _cdiv(E, world)
    padded = shard * world
    if padded != E:
        full = nv.new_zeros([padded, *nv.shape[1:]])
        full.narrow(0, 0, E).copy_(nv)
    else:
        full = nv.clone()
    shards = torch.split(full, shard, 0)
    assert len(shards) == world, (len(shards), world)
    pre = [s.fsdp_pre_all_gather(mesh=None) for s in shards]
    meta = pre[0][1]
    gathered = []
    for k in range(len(pre[0][0])):
        tensors = [p[0][k] for p in pre]
        if tensors[0].dim() == 0:
            gathered.append(
                tensors[0]
            )  # scalar per_tensor_scale: replicated, not sharded
        else:
            gathered.append(torch.cat(tensors, 0))
    recon, _ = shards[0].fsdp_post_all_gather(tuple(gathered), meta, torch.bfloat16)
    return recon.narrow(0, 0, E)


@pytest.mark.parametrize("E,N,K", [(4, 8, 16), (6, 8, 32)], ids=["E4_K16", "E6_K32"])
def test_nvfp4_split_narrow_slice_roundtrip(E, N, K):
    """split/narrow/slice on the expert axis dequantize equal to the original slice."""
    nv = _make_nvfp4(E, N, K)
    orig = nv.dequantize(torch.bfloat16)

    parts = torch.split(nv, 2, 0)
    rec = torch.cat([p.dequantize(torch.bfloat16) for p in parts], 0)
    assert torch.equal(rec, orig), "split round-trip diverged"

    nar = torch.narrow(nv, 0, 1, 2)
    assert torch.equal(nar.dequantize(torch.bfloat16), orig[1:3]), "narrow diverged"

    sl = nv[1:3]  # aten.slice on dim 0
    assert torch.equal(sl.dequantize(torch.bfloat16), orig[1:3]), "slice diverged"


def test_nvfp4_clone_detach_as_strided_roundtrip():
    """clone / detach / as_strided preserve the dequantized value (view ops return new tensors)."""
    nv = _make_nvfp4(6, 8, 32)
    orig = nv.dequantize(torch.bfloat16)
    assert torch.equal(nv.clone().dequantize(torch.bfloat16), orig), "clone diverged"
    assert torch.equal(nv.detach().dequantize(torch.bfloat16), orig), "detach diverged"

    shape = list(nv.shape)
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    ast = torch.as_strided(nv, shape, stride)
    assert torch.equal(ast.dequantize(torch.bfloat16), orig), "as_strided diverged"


@pytest.mark.parametrize(
    "E,worlds",
    [(6, (1, 2, 3)), (6, (4,)), (8, (3,))],
    ids=["even_1_2_3", "uneven_world4_E6", "uneven_world3_E8"],
)
def test_nvfp4_fsdp_all_gather_roundtrip(E, worlds):
    """FSDP2 shard -> all-gather -> reconstruct round-trips across worlds, incl. uneven expert splits.

    world=4 over E=6 and world=3 over E=8 do NOT divide evenly: the shard is padded on the expert
    axis (new_zeros + copy_ into a narrow) and narrowed back after reconstruct. Pure CPU logic.
    """
    nv = _make_nvfp4(E, 8, 32)
    orig = nv.dequantize(torch.bfloat16)
    for world in worlds:
        recon = _shard_all_gather_reconstruct(nv, world)
        assert torch.equal(recon.dequantize(torch.bfloat16), orig), (
            f"FSDP all-gather reconstruct diverged at world={world} (E={E})"
        )


def test_nvfp4_new_zeros_preserves_trailing_shapes():
    """new_zeros varies only dim 0 (experts); qdata/scale trailing dims are preserved verbatim."""
    nv = _make_nvfp4(6, 8, 32)
    nz = nv.new_zeros([10, 8, 32])
    assert nz.qdata.shape == (10, *nv.qdata.shape[1:])
    assert nz.scale.shape == (10, *nv.scale.shape[1:])


def test_nvfp4_per_expert_pts_split_narrow_shapes():
    """A per-expert per_tensor_scale [E] is sliced alongside the expert axis on split/narrow.

    torchao cannot dequantize a per-expert per_tensor_scale against per-block scales, so this
    checks the slicing at the shape level (the FSDP substrate must carry the [E] buffer through).
    """
    nv = _make_nvfp4(6, 8, 32, per_expert_pts=True)
    assert nv.per_tensor_scale is not None and nv.per_tensor_scale.shape == (6,)

    parts = torch.split(nv, 2, 0)
    assert all(p.per_tensor_scale.shape == (2,) for p in parts), (
        "per-expert per_tensor_scale not split on the expert axis"
    )
    assert torch.equal(
        torch.cat([p.per_tensor_scale for p in parts], 0), nv.per_tensor_scale
    )

    nar = torch.narrow(nv, 0, 1, 3)
    assert nar.per_tensor_scale.shape == (3,)
    assert torch.equal(nar.per_tensor_scale, nv.per_tensor_scale[1:4])


def test_nvfp4_split_dim_other_than_zero_rejected():
    """Sharding is restricted to dim 0 (expert axis); a non-zero dim split must raise."""
    nv = _make_nvfp4(4, 8, 16)
    with pytest.raises(NotImplementedError, match="only on dim 0"):
        torch.split(nv, 4, 1)
