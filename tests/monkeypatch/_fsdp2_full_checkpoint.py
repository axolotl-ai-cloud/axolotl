"""Distributed CPU regression worker for full EP and 8-bit checkpoints."""

import copy
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torchao.optim import AdamW8bit
from torchao.optim.adam import single_param_adam
from torchao.optim.subclass_8bit import OptimState8bit

from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    full_model_state,
    full_optimizer_state,
    restore_model_state,
    restore_optimizer_state,
)
from axolotl.monkeypatch.torchao_optim import patch_torchao_optim_state_8bit


class Experts(nn.Module):
    def __init__(self, ep_rank):
        super().__init__()
        self.num_local_experts = 2
        self.num_experts_global = 4
        self.local_expert_offset = ep_rank * 2
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 16, 256))


class ParamWrapper(nn.Module):
    def __init__(self, ep_rank):
        super().__init__()
        self.base_layer = Experts(ep_rank)
        self.parameter_name = "gate_up_proj"
        self._ep_lora_sharded = True
        self.lora_A = nn.ModuleDict({"default": nn.Linear(256, 32, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(32, 256, bias=False)})


class Model(nn.Module):
    def __init__(self, mesh, expert_mesh, expert_placements, dense_placements):
        super().__init__()
        self.expert = ParamWrapper(mesh.get_coordinate()[-1])
        self.dense = nn.Linear(256, 32, bias=False)
        for module in self.expert.modules():
            for name, p in list(module.named_parameters(recurse=False)):
                value = distribute_tensor(
                    torch.zeros_like(p), expert_mesh, expert_placements
                )
                setattr(module, name, nn.Parameter(value))
        self.dense.weight = nn.Parameter(
            distribute_tensor(
                torch.zeros_like(self.dense.weight), mesh, dense_placements
            )
        )
        self.register_buffer("counter", torch.tensor(3))


def snapshot(value):
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, OptimState8bit):
        return {a: getattr(value, a).clone() for a in value.tensor_attrs}
    return value.detach().clone()


def compare(actual, expected):
    if isinstance(actual, DTensor):
        actual = actual.to_local()
    if isinstance(expected, dict):
        assert isinstance(actual, OptimState8bit)
        for attr in actual.tensor_attrs:
            torch.testing.assert_close(
                getattr(actual, attr), expected[attr], rtol=0, atol=0
            )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def populate(model, optimizer, ep_rank, shard_rank):
    for i, (name, p) in enumerate(model.named_parameters()):
        local = p.to_local()
        with torch.no_grad():
            local.copy_(
                torch.arange(local.numel()).reshape(local.shape) / 10000
                + ep_rank
                + shard_rank * 10
                + i
            )
        values = {
            "step": torch.tensor(2.0 + (ep_rank if name.startswith("expert.") else 0))
        }
        for key, signed in [("exp_avg", True), ("exp_avg_sq", False)]:
            buffer = optimizer._new_buffer(p, signed)
            inner = buffer.to_local()
            if isinstance(inner, OptimState8bit):
                inner.codes.copy_(
                    (
                        (
                            torch.arange(inner.numel()).reshape(inner.shape)
                            + ep_rank * 19
                            + shard_rank * 7
                        )
                        % 256
                    ).to(torch.uint8)
                )
                inner.scale.copy_(
                    torch.arange(inner.scale.numel()) / 1000
                    + 1
                    + ep_rank
                    + shard_rank * 10
                )
            else:
                inner.fill_(0.1 + ep_rank + shard_rank)
            values[key] = buffer
        optimizer.state[p] = values


def roundtrip(root, label, model, optimizer, target=None):
    rank = dist.get_rank()
    expected = {
        name: dict(
            parameter=snapshot(p),
            state={k: snapshot(v) for k, v in optimizer.state[p].items()},
        )
        for name, p in model.named_parameters()
    }
    # Keep an independent pre-save reference for each EP group's complete tensors.
    full_reference = {
        name: dict(
            parameter=snapshot(p.full_tensor()),
            state={
                k: snapshot(v.full_tensor() if isinstance(v, DTensor) else v)
                for k, v in optimizer.state[p].items()
            },
        )
        for name, p in model.named_parameters()
    }
    ep_rank = (
        model.expert.base_layer.local_expert_offset // 2
        if hasattr(model, "expert")
        else 0
    )
    references = [None] * dist.get_world_size()
    dist.all_gather_object(references, (ep_rank, full_reference))
    model_state = full_model_state(model)
    optimizer_state = full_optimizer_state(model, optimizer)
    if rank == 0:
        torch.save(model_state, root / f"{label}-model.bin")
        torch.save(optimizer_state, root / f"{label}-optimizer.bin")
        if hasattr(model, "expert"):
            assert model_state["expert.base_layer.gate_up_proj"].shape == (4, 16, 256)
            assert model_state["expert.lora_A.default.weight"].shape == (64, 256)
            assert model_state["expert.lora_B.default.weight"].shape == (256, 64)
    dist.barrier()
    if target is None:
        for p in model.parameters():
            with torch.no_grad():
                p.to_local().zero_()
        optimizer.state.clear()
    else:
        model, optimizer = target()
    model_state = (
        torch.load(root / f"{label}-model.bin", weights_only=True) if rank == 0 else {}
    )
    optimizer_state = (
        torch.load(root / f"{label}-optimizer.bin", weights_only=True)
        if rank == 0
        else {}
    )
    restore_model_state(model, model_state)
    restore_optimizer_state(model, optimizer, optimizer_state)
    if target is None:
        for name, p in model.named_parameters():
            compare(p, expected[name]["parameter"])
            for key, value in optimizer.state[p].items():
                compare(value, expected[name]["state"][key])
        # One more Adam update must be identical, including the newly quantized moments.
        for name, p in model.named_parameters():
            state = optimizer.state[p]
            local = p.to_local()
            reference_p = expected[name]["parameter"].clone()
            reference_state = copy.deepcopy(expected[name]["state"])
            for key, signed in [("exp_avg", True), ("exp_avg_sq", False)]:
                if isinstance(reference_state[key], dict):
                    parts = reference_state[key]
                    reference_state[key] = OptimState8bit(
                        *(parts[a] for a in ("codes", "scale", "qmap")),
                        signed,
                        dtype=torch.float32,
                    )
            grad = torch.full_like(local, 0.125)
            for parameter, values in [(local, state), (reference_p, reference_state)]:
                values["step"] += 1
                with torch.no_grad():
                    single_param_adam(
                        parameter,
                        grad,
                        values["step"],
                        values["exp_avg"].to_local()
                        if isinstance(values["exp_avg"], DTensor)
                        else values["exp_avg"],
                        values["exp_avg_sq"].to_local()
                        if isinstance(values["exp_avg_sq"], DTensor)
                        else values["exp_avg_sq"],
                        None,
                        torch.tensor(0.001),
                        0.9,
                        0.999,
                        0.01,
                        1e-8,
                        True,
                        False,
                    )
            compare(p, reference_p)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                compare(state[key], snapshot(reference_state[key]))
    else:
        ep_rank = model.expert.base_layer.local_expert_offset // 2
        reference = next(values for ep, values in references if ep == ep_rank)
        for name, p in model.named_parameters():
            compare(p.full_tensor(), reference[name]["parameter"])
            for key, value in optimizer.state[p].items():
                compare(
                    value.full_tensor() if isinstance(value, DTensor) else value,
                    reference[name]["state"][key],
                )
    assert model.counter.item() == 3
    if rank == 0:
        print(f"PASS {label}", flush=True)
    dist.barrier()


def main():
    patch_torchao_optim_state_8bit()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    root = Path(sys.argv[1])
    root.mkdir(exist_ok=True)
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "ep"))
    model = Model(mesh, mesh["dp"], (Shard(0),), (Shard(0), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])
    roundtrip(root, "ep-dp", model, optimizer)
    # Recreate the source after the update performed by the first check.
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])
    permuted = DeviceMesh(
        "cpu", torch.tensor([[0, 2], [1, 3]]), mesh_dim_names=("dp", "ep")
    )

    def target():
        new_model = Model(permuted, permuted["dp"], (Shard(0),), (Shard(0), Shard(0)))
        return new_model, AdamW8bit(new_model.parameters(), lr=0.001)

    roundtrip(root, "ep-permuted-ranks", model, optimizer, target)
    cp = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("cp", "ep"))
    model = Model(cp, cp["cp"], (Replicate(),), (Replicate(), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, cp.get_coordinate()[1], 0)
    roundtrip(root, "ep-cp", model, optimizer)
    hsdp = DeviceMesh(
        "cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("replicate", "ep")
    )
    # All ranks must create subgroup meshes in the same collective order.
    expert_meshes = [
        DeviceMesh(
            "cpu",
            hsdp.mesh[:, ep].reshape(2, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        for ep in range(2)
    ]
    expert_mesh = expert_meshes[hsdp.get_coordinate()[1]]
    model = Model(mesh, mesh["dp"], (Shard(0),), (Shard(0), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])

    def hsdp_target():
        target_model = Model(
            hsdp, expert_mesh, (Replicate(), Shard(0)), (Replicate(), Shard(0))
        )
        return target_model, AdamW8bit(target_model.parameters(), lr=0.001)

    roundtrip(root, "ep-dp-to-hsdp", model, optimizer, hsdp_target)
    model, optimizer = hsdp_target()
    populate(model, optimizer, hsdp.get_coordinate()[1], 0)
    roundtrip(root, "ep-hsdp", model, optimizer)
    flat = DeviceMesh("cpu", torch.arange(4), mesh_dim_names=("tp",))
    model = nn.Module()
    model.weight = nn.Parameter(
        distribute_tensor(torch.zeros(128, 256), flat, (Shard(1),))
    )
    model.register_buffer("counter", torch.tensor(3))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, rank, 0)
    # The wrapper cannot full_tensor() a column shard; compare its physical local payload.
    expected_p = snapshot(model.weight)
    expected_m = {k: snapshot(v) for k, v in optimizer.state[model.weight].items()}
    state = full_optimizer_state(model, optimizer)
    restore_optimizer_state(model, optimizer, state)
    compare(model.weight, expected_p)
    for k, v in optimizer.state[model.weight].items():
        compare(v, expected_m[k])
    if rank == 0:
        print("PASS tp-cross-global-blocks-same-layout", flush=True)
    small_source = nn.Module()
    small_source.weight = nn.Parameter(
        distribute_tensor(
            torch.zeros(64, 64, dtype=torch.bfloat16), flat, (Replicate(),)
        )
    )
    small_optimizer = AdamW8bit(small_source.parameters())
    populate(small_source, small_optimizer, 0, 0)
    expected_floats = {
        key: value.to_local().dequantize(output_dtype=torch.float32)
        for key, value in small_optimizer.state[small_source.weight].items()
        if key != "step"
    }
    small_state = full_optimizer_state(small_source, small_optimizer)
    small_target = nn.Module()
    small_target.weight = nn.Parameter(
        distribute_tensor(torch.zeros(64, 64, dtype=torch.bfloat16), flat, (Shard(0),))
    )
    small_target_optimizer = AdamW8bit(small_target.parameters())
    restore_optimizer_state(small_target, small_target_optimizer, small_state)
    for key, value in small_target_optimizer.state[small_target.weight].items():
        if key != "step":
            assert value.dtype == torch.float32
            assert not isinstance(value.to_local(), OptimState8bit)
            compare(value.to_local(), expected_floats[key].chunk(4, dim=0)[rank])
    if rank == 0:
        print("PASS quantized-to-small-fp32-shards", flush=True)

    from torch.distributed.tensor.placement_types import _StridedShard

    strided_model = nn.Module()
    strided_model.weight = nn.Parameter(
        distribute_tensor(
            torch.zeros(128, 256), mesh, (_StridedShard(0, split_factor=2), Shard(0))
        )
    )
    strided_optimizer = AdamW8bit(strided_model.parameters(), lr=0.001)
    populate(strided_model, strided_optimizer, rank, 0)
    reference_full = strided_model.weight.full_tensor().detach()
    saved_model = full_model_state(strided_model)
    if rank == 0:
        torch.testing.assert_close(
            saved_model["weight"], reference_full, rtol=0, atol=0
        )
    expected_strided = {
        k: snapshot(v) for k, v in strided_optimizer.state[strided_model.weight].items()
    }
    saved_strided = full_optimizer_state(strided_model, strided_optimizer)
    strided_optimizer.state.clear()
    restore_optimizer_state(strided_model, strided_optimizer, saved_strided)
    for key, value in strided_optimizer.state[strided_model.weight].items():
        compare(value, expected_strided[key])
    if rank == 0:
        print("PASS strided-dp-tp", flush=True)

    row_model = nn.Module()
    row_model.weight = nn.Parameter(
        distribute_tensor(torch.zeros(128, 256), flat, (Shard(0),))
    )
    try:
        restore_optimizer_state(row_model, AdamW8bit(row_model.parameters()), state)
    except ValueError as exc:
        assert "regroups saved quantization blocks" in str(exc)
    else:
        raise AssertionError("Incompatible regrouping was accepted")
    if rank == 0:
        print("PASS incompatible-restore-rejected", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
