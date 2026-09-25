"""CPU numerical parity for Accelerate meshes with TP, FSDP2, and Ringmaster."""

import copy

import ringmaster as rm
import torch
import torch.distributed as dist
from accelerate import ParallelismConfig
from ringmaster.config import RotateMethod
from ringmaster.ring.kernels import math_block
from ringmaster.shard import varlen_meta
from ringmaster.strategies.ulysses import make_ulysses_attention
from ringmaster.strategies.usp import make_usp_attention
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from axolotl.integrations.context_parallel.mesh import RingmasterMesh


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16, bias=False)
        self.k = nn.Linear(16, 16, bias=False)
        self.v = nn.Linear(16, 16, bias=False)
        self.out = nn.Linear(16, 16, bias=False)
        self.attention = None

    def forward(self, x, lengths):
        q, k, v = [
            layer(x).unflatten(-1, (-1, 2)) for layer in (self.q, self.k, self.v)
        ]
        if self.attention is not None:
            y, _ = self.attention(
                self,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                None,
                scaling=None,
                is_causal=True,
            )
        else:
            parts = []
            start = 0
            for length in lengths:
                part, _ = math_block(
                    q[:, start : start + length],
                    k[:, start : start + length],
                    v[:, start : start + length],
                    causal=True,
                    scaling=None,
                )
                parts.append(part)
                start += length
            y = torch.cat(parts, dim=1)
        return self.out(y.flatten(-2))


def check(hsdp):
    pc = ParallelismConfig(
        dp_replicate_size=2 if hsdp else 1,
        dp_shard_size=2,
        cp_size=2 if hsdp else 4,
        tp_size=2,
    )
    mesh = pc.build_device_mesh("cpu")
    original_groups = {
        name: mesh[name].get_group() for name in ("cp", "tp", "dp", "dp_shard_cp")
    }
    view = RingmasterMesh(mesh, ring_size=1 if hsdp else 2, ulysses_size=2)
    runtime = rm.setup(
        rm.RingmasterConfig(
            size=pc.cp_size,
            backend=rm.Backend.USP,
            ring_size=1 if hsdp else 2,
            ulysses_size=2,
        ),
        num_kv_heads=4,
        device_mesh=view,
        device_type="cpu",
        inner_attn="sdpa" if hsdp else "math",
    )
    for name, group in original_groups.items():
        assert mesh[name].get_group() is group
    assert view["cp"].get_group() is original_groups["cp"]
    cp_rank = dist.get_rank(original_groups["cp"])
    dp_rank = dist.get_rank(original_groups["dp"])
    dp_size = pc.dp_replicate_size * pc.dp_shard_size
    for packed in (False, True):
        torch.manual_seed(42)
        reference = Attention().double()
        model = copy.deepcopy(reference)
        parallelize_module(
            model,
            mesh["tp"],
            {
                "q": ColwiseParallel(),
                "k": ColwiseParallel(),
                "v": ColwiseParallel(),
                "out": RowwiseParallel(input_layouts=Shard(-1)),
            },
        )
        model.attention = (
            make_ulysses_attention("sdpa")
            if hsdp
            else make_usp_attention("math", "math", RotateMethod.ALLGATHER)
        )
        fully_shard(model, mesh=mesh[pc.fsdp_dim_names])
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.03)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        expected_output = None
        local_input = None
        local_lengths = None
        for sample in range(dp_size):
            torch.manual_seed(100 + sample)
            x = torch.randn(1, 16, 16, dtype=torch.float64)
            lengths = [3 + sample, 5, 8 - sample] if packed else [16]
            expected = reference(x, lengths)
            (expected.square().sum() / dp_size).backward()
            if sample == dp_rank:
                expected_output = expected.detach()
                local_input = x
                local_lengths = lengths
        positions = torch.cat(
            [torch.arange(length) for length in local_lengths]
        ).unsqueeze(0)
        runtime.varlen = varlen_meta(positions, 16) if packed else None
        chunk = 16 // pc.cp_size
        sl = slice(cp_rank * chunk, (cp_rank + 1) * chunk)
        actual = model(local_input[:, sl].contiguous(), local_lengths)
        torch.testing.assert_close(actual, expected_output[:, sl], atol=1e-6, rtol=1e-5)
        # FSDP averages across CP as well as DP; each CP rank contributes distinct tokens.
        (actual.square().sum() * pc.cp_size).backward()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad.full_tensor(),
                dict(reference.named_parameters())[name].grad,
                atol=1e-6,
                rtol=1e-5,
            )
        optimizer.step()
        expected_optimizer.step()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.full_tensor(),
                dict(reference.named_parameters())[name],
                atol=1e-6,
                rtol=1e-5,
            )
        print(f"PASS ND rank={dist.get_rank()} hsdp={hsdp} packed={packed}", flush=True)
    rm.teardown()


if __name__ == "__main__":
    dist.init_process_group("gloo")
    try:
        check(False)
        check(True)
    finally:
        dist.destroy_process_group()
