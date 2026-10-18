"""Distributed CPU worker: the gathered EP-LoRA adapter save keeps tensors only on the writer."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from axolotl.integrations.expert_parallel import shard
from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _check_errors,
    full_model_state,
)


def make_model(mesh):
    torch.manual_seed(0)
    base = nn.Module()
    base.experts = nn.Module()
    base.experts.num_experts = 4
    base.experts.gate_up_proj = nn.Parameter(torch.ones(4, 16, 256))
    base.experts.down_proj = nn.Parameter(torch.ones(4, 256, 16))
    base.dense = nn.Linear(256, 32, bias=False)
    experts = base.experts
    ep_rank = mesh.get_coordinate()[1]
    local = experts.num_experts // mesh["ep"].size()
    offset = ep_rank * local
    for name in ("gate_up_proj", "down_proj"):
        parameter = getattr(experts, name)
        setattr(
            experts,
            name,
            nn.Parameter(parameter[offset : offset + local].detach().clone()),
        )
    experts.num_experts_global = experts.num_experts
    experts.num_local_experts = local
    experts.local_expert_offset = offset
    experts.num_experts = local
    model = get_peft_model(
        base,
        LoraConfig(
            target_modules=["dense"],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            r=16,
        ),
    )
    assert shard.shard_expert_lora(model, mesh["ep"].size()) == 4
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.requires_grad:
                setattr(
                    module,
                    name,
                    nn.Parameter(distribute_tensor(parameter, mesh["dp"], (Shard(0),))),
                )
    dp_rank = mesh.get_coordinate()[0]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                value = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                value.copy_(
                    torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape)
                    * 1e-3
                    + ep_rank
                    + 10 * dp_rank
                )
    return model


def main():
    dist.init_process_group("gloo")
    root = Path(sys.argv[1])
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "ep"))
    model = make_model(mesh)
    expected = get_peft_model_state_dict(
        model, state_dict=full_model_state(model, adapter_only=True)
    )

    kept = []
    keep = shard._keep_for_writer

    def spy(tensor):
        result = keep(tensor)
        kept.append(result)
        return result

    out = root / "ep-adapter"
    with patch.object(shard, "_keep_for_writer", spy):
        assert shard.save_ep_lora_adapter(model, str(out), mesh["ep"].get_group())
    error = None
    if dist.get_rank() == 0:
        if not kept or any(t.device.type != "cpu" for t in kept):
            error = "writer rank kept a gathered adapter tensor off CPU"
    elif not kept or any(not t.is_meta for t in kept):
        error = f"rank {dist.get_rank()} kept gathered adapter storage"
    _check_errors(error)

    if dist.get_rank() == 0:
        saved = load_file(out / "adapter_model.safetensors")
        assert set(saved) == set(expected), (sorted(saved), sorted(expected))
        for key, value in expected.items():
            torch.testing.assert_close(saved[key], value, rtol=0, atol=0)
        assert any(value.shape == (64, 256) for value in saved.values())
        print("PASS ep-lora-adapter-save-writer-only", flush=True)

    kept.clear()
    with patch.object(shard, "_keep_for_writer", spy):
        assert shard.save_fsdp2_lora_adapter(model, str(root / "fsdp2-adapter"))
    error = None
    if dist.get_rank() == 0:
        if not kept or any(t.device.type != "cpu" for t in kept):
            error = "writer rank kept a gathered FSDP2 adapter tensor off CPU"
    elif not kept or any(not t.is_meta for t in kept):
        error = f"rank {dist.get_rank()} kept gathered FSDP2 adapter storage"
    _check_errors(error)
    if dist.get_rank() == 0:
        saved = load_file(root / "fsdp2-adapter" / "adapter_model.safetensors")
        assert set(saved) == set(expected), (sorted(saved), sorted(expected))
        print("PASS fsdp2-lora-adapter-save-writer-only", flush=True)

    # a wrapper whose adapter key cannot be resolved must fail on every rank, not write a partial
    # adapter (and not leave the other ranks waiting in the barrier)
    strip = shard._strip_checkpoint_wrapper

    def unresolvable(name):
        # wrapper keys are ``<module>.lora_A.weight``; parameter names keep ``.default``
        return strip(name).replace("lora_A.weight", "lora_Z.weight")

    with patch.object(shard, "_strip_checkpoint_wrapper", unresolvable):
        try:
            shard.save_ep_lora_adapter(model, str(root / "bad"), mesh["ep"].get_group())
        except ValueError as exc:
            assert "refusing to write a partial adapter" in str(exc)
        else:
            raise AssertionError("unresolved expert LoRA key was not rejected")
    assert not (root / "bad" / "adapter_model.safetensors").exists()
    if dist.get_rank() == 0:
        print("PASS ep-lora-adapter-save-rejects-unfilled", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
