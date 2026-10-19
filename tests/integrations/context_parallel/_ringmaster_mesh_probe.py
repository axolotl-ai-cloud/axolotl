"""Eight-rank CPU probe for DP x Ring x Ulysses mesh membership."""

import torch
import torch.distributed as dist
from ringmaster.batch import broadcast_batch
from torch.distributed.device_mesh import init_device_mesh

from axolotl.integrations.context_parallel.mesh import RingmasterMesh

dist.init_process_group("gloo")
rank = dist.get_rank()
mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("dp_shard", "cp"))
view = RingmasterMesh(mesh, ring_size=2, ulysses_size=2)
assert dist.get_process_group_ranks(view["cp"].get_group()) == list(
    range(rank // 4 * 4, rank // 4 * 4 + 4)
)
assert dist.get_process_group_ranks(view["cp_ulysses"].get_group()) == [
    rank // 2 * 2,
    rank // 2 * 2 + 1,
]
assert dist.get_process_group_ranks(view["cp_ring"].get_group()) == [
    rank // 4 * 4 + rank % 2,
    rank // 4 * 4 + rank % 2 + 2,
]

source = rank // 4 * 4
batch = (
    {"input_ids": torch.full((1, 3 + source), source, dtype=torch.long)}
    if rank == source
    else {"other": torch.ones(1, 1)}
)
broadcast_batch(batch, view["cp"].get_group())
assert set(batch) == {"input_ids"}
assert batch["input_ids"].shape == (1, 3 + source)
assert batch["input_ids"].dtype == torch.long
assert (batch["input_ids"] == source).all()
print("PASS mesh rank", rank, flush=True)
dist.destroy_process_group()
