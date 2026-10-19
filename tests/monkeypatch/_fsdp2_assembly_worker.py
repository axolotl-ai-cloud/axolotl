"""Distributed CPU worker: slice-view assembly of gathered checkpoint tensors is bit-identical to
the index (meshgrid) assembly it replaced, and still rejects gaps and overlaps."""

import os
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from axolotl.monkeypatch.accelerate import fsdp2_checkpoint
from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _check_errors,
    _coordinates,
    _dtype,
    _index,
    _layout,
    _local,
    _transfer,
)


def index_gather(value, owner=None):
    """The meshgrid assembly ``_gather_tensor`` used before the slice-view rewrite."""
    local = _local(value)
    record = dict(_layout(value, owner), dtype=str(local.dtype), quantized=False)
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    chunks, seen = [], set()
    for rank, info in enumerate(records):
        identity = repr(
            (info["owner"], info["offset"], info["local_shape"], info.get("axes"))
        )
        if identity in seen:
            continue
        seen.add(identity)
        chunk = dict(info)
        chunk["tensor"] = _transfer(
            local if dist.get_rank() == rank else None,
            info["local_shape"],
            _dtype(info["dtype"]),
            rank,
        )
        if dist.get_rank() == 0:
            chunks.append(chunk)
    full, error = None, None
    if dist.get_rank() == 0:
        full = torch.empty(records[0]["shape"], dtype=local.dtype)
        covered = torch.zeros(records[0]["shape"], dtype=torch.bool)
        for chunk in chunks:
            indices = _index(_coordinates(chunk))
            full[indices] = chunk["tensor"]
            covered[indices] = True
        if not covered.all():
            error = "Checkpoint does not cover every expert and shard"
    _check_errors(error)
    return full


# "cuda" runs the NCCL transfer path (pinned staging) on 4 GPUs
DEVICE = os.environ.get("AXOLOTL_ASSEMBLY_DEVICE", "cpu")


def on(tensor):
    return tensor.to(DEVICE, copy=True)


def bits(tensor):
    return tensor.view({4: torch.int32, 2: torch.int16}[tensor.element_size()])


def expect(label, value, owner, reference, indexed):
    calls = []
    original = fsdp2_checkpoint._gather_indexed

    def spy(*args):
        calls.append(1)
        return original(*args)

    with patch.object(fsdp2_checkpoint, "_gather_indexed", spy):
        new = fsdp2_checkpoint._gather_tensor(value, owner)
    old = index_gather(value, owner)
    assert bool(calls) == indexed, (label, calls)
    if dist.get_rank() == 0:
        assert new.dtype == old.dtype == reference.dtype, label
        assert torch.equal(bits(new), bits(old)), label
        assert torch.equal(bits(new), bits(reference)), label
        print(f"PASS assembly-{label}", flush=True)
    else:
        assert new is None and old is None


def expect_error(label, value, owner, message):
    try:
        fsdp2_checkpoint._gather_tensor(value, owner)
    except ValueError as exc:
        assert message in str(exc), (label, str(exc))
    else:
        raise AssertionError(f"{label}: assembly accepted an invalid layout")
    if dist.get_rank() == 0:
        print(f"PASS assembly-{label}", flush=True)


def full(*shape, dtype=torch.float32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=generator).to(dtype)


def plain_shards(flat, mesh):
    for label, tensor, placements, target in (
        ("shard0-uneven", full(10, 7, seed=1), (Shard(0),), flat),
        ("shard1-uneven", full(6, 10, seed=2), (Shard(1),), flat),
        ("shard0-empty-ranks", full(2, 5, seed=3), (Shard(0),), flat),
        ("shard0-bf16", full(13, 3, dtype=torch.bfloat16, seed=4), (Shard(0),), flat),
        ("2d-shard0-shard1", full(9, 11, seed=5), (Shard(0), Shard(1)), mesh),
        ("2d-shard0-shard0", full(11, 4, seed=6), (Shard(0), Shard(0)), mesh),
    ):
        expect(
            label,
            distribute_tensor(on(tensor), target, placements),
            None,
            tensor,
            False,
        )


def expert_layouts(mesh):
    rank = dist.get_rank()
    _, ep_rank = mesh.get_coordinate()
    # kind A, expert-major rows [E * r, in]; dp-sharded EP-local rows, uneven too
    for label, r, local, total, width in (
        ("kind-a", 4, 4, 8, 12),
        ("kind-a-uneven", 3, 3, 6, 5),
    ):
        reference = full(total * r, width, seed=10 + r)
        offset = ep_rank * local
        ep_local = reference[offset * r : (offset + local) * r]
        owner = dict(offset=offset, local=local, total=total, kind="A")
        value = distribute_tensor(on(ep_local), mesh["dp"], (Shard(0),))
        expect(label, value, owner, reference, False)
    # kind B, Beam's interleaved [out, r * E] with experts innermost, scaled down
    out, r, total = 40, 4, 8
    reference = full(out, r * total, seed=20)
    local = total // 4
    offset = rank * local
    ep_local = reference.view(out, r, total)[:, :, offset : offset + local]
    owner = dict(offset=offset, local=local, total=total, kind="B")
    expect("kind-b-ep4", on(ep_local.reshape(out, r * local)), owner, reference, False)
    out = 41
    reference = full(out, r * total, seed=21)
    local = total // 2
    offset = ep_rank * local
    ep_local = reference.view(out, r, total)[:, :, offset : offset + local]
    owner = dict(offset=offset, local=local, total=total, kind="B")
    value = distribute_tensor(
        on(ep_local.reshape(out, r * local)), mesh["dp"], (Shard(0),)
    )
    expect("kind-b-dp-uneven", value, owner, reference, False)
    # expert base weights [E, ...], one expert per dp shard
    reference = full(4, 6, 5, seed=30)
    owner = dict(offset=ep_rank * 2, local=2, total=4, kind="expert")
    value = distribute_tensor(
        on(reference[ep_rank * 2 : ep_rank * 2 + 2]), mesh["dp"], (Shard(0),)
    )
    expect("kind-expert", value, owner, reference, False)


def strided(mesh):
    reference = full(16, 6, seed=40)
    # rows interleave across ranks: only index assembly can place them
    value = distribute_tensor(
        on(reference), mesh, (Shard(0), _StridedShard(0, split_factor=2))
    )
    assert fsdp2_checkpoint._box(_layout(value)) is None
    expect("strided-index-fallback", value, None, reference, True)
    # strided placement that still leaves every rank one contiguous row block
    value = distribute_tensor(
        on(reference), mesh, (_StridedShard(0, split_factor=2), Shard(0))
    )
    expect("strided-contiguous-rows", value, None, reference, False)


def invalid():
    rank = dist.get_rank()
    value = on(full(1, 3, seed=50 + rank))
    owner = dict(offset=rank, local=1, total=8, kind="expert")
    expect_error("gap-rejected", value, owner, "does not cover every expert and shard")
    value = on(full(2, 3, seed=60 + min(rank, 2)))
    owner = dict(offset=min(rank, 2), local=2, total=4, kind="expert")
    expect_error("overlap-rejected", value, owner, "Checkpoint shards overlap")


def main():
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("cuda:nccl,cpu:gloo")
    else:
        dist.init_process_group("gloo")
    flat = init_device_mesh(DEVICE, (4,), mesh_dim_names=("tp",))
    mesh = init_device_mesh(DEVICE, (2, 2), mesh_dim_names=("dp", "ep"))
    plain_shards(flat, mesh)
    expert_layouts(mesh)
    strided(mesh)
    invalid()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
