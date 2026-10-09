"""Real CPU FSDP2 and Gloo all-to-all regression for compact expert adapters."""

import copy
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from peft import get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from safetensors.torch import load_file
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

from axolotl.integrations.expert_parallel import experts_fn, torch_dispatch
from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
from axolotl.integrations.expert_parallel.shard import (
    save_ep_lora_adapter,
    shard_expert_lora,
)
from axolotl.integrations.moe_sieve.peft import (
    MoeSieveLoraConfig,
    register_selected_experts,
)
from axolotl.integrations.moe_sieve.selection import packed_experts
from axolotl.loaders.adapter import reinit_lora_from_seed
from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    full_model_state,
    full_optimizer_state,
    restore_model_state,
    restore_optimizer_state,
)


class TinyMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Qwen3MoeConfig(
            hidden_size=16, moe_intermediate_size=8, num_experts=8
        )
        self.experts = Qwen3MoeExperts(self.config)
        self.dense = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.experts.gate_up_proj.normal_(std=0.1)
            self.experts.down_proj.normal_(std=0.1)

    def forward(self, x, ids, weights):
        return self.dense(self.experts(x, ids, weights))


def make_model(selected, mesh=None):
    torch.manual_seed(42)
    base = TinyMoE()
    selection = {
        name: dict(num_experts=8, parameter_shapes=shapes, selected_experts=selected)
        for name, (_, shapes) in packed_experts(base).items()
    }
    config = MoeSieveLoraConfig(
        r=2,
        lora_alpha=4,
        target_modules=["dense"],
        target_parameters=["experts.gate_up_proj", "experts.down_proj"],
        moe_sieve_selection=selection,
    )
    if mesh is not None:
        offset = mesh["ep"].get_local_rank() * 4
        experts = base.experts
        for name in ("gate_up_proj", "down_proj"):
            setattr(
                experts,
                name,
                nn.Parameter(
                    getattr(experts, name)[offset : offset + 4].detach().clone()
                ),
            )
        experts.num_experts_global = 8
        experts.num_local_experts = experts.num_experts = 4
        experts.local_expert_offset = offset
        base.config._experts_implementation = "expert_parallel"
    register_selected_experts(base, config)
    model = get_peft_model(base, config)
    reinit_lora_from_seed(model, 42)
    if mesh is not None:
        shard_expert_lora(model, 2)
        kwargs = dict(ignored_params={p for p in model.parameters() if p.numel() == 0})
        expert_mesh = mesh[tuple(n for n in mesh.mesh_dim_names if n != "ep")]
        ExpertParallelPlugin.fully_shard_experts(model, expert_mesh, kwargs)
        outer = (
            mesh["shard", "ep"]._flatten()
            if "replicate" not in mesh.mesh_dim_names
            else mesh
        )
        if "replicate" in mesh.mesh_dim_names:
            # HSDP's sharding axis includes EP for the shared dense parameters.
            mesh["shard", "ep"]._flatten(mesh_dim_name="dense_shard")
            outer = mesh["replicate", "dense_shard"]
        fully_shard(model, mesh=outer, **kwargs)
    return model


def batch(step, rank=None):
    generator = torch.Generator().manual_seed(100 + step)
    x = torch.randn(4, 8, 16, generator=generator)
    ids = torch.arange(64).reshape(4, 8, 2) % 8
    weights = torch.softmax(torch.randn(4, 8, 2, generator=generator), dim=-1)
    if rank is None:
        return x.flatten(0, 1), ids.flatten(0, 1), weights.flatten(0, 1)
    return x[rank], ids[rank], weights[rank]


def update(model, optimizer, step, rank=None):
    optimizer.zero_grad(set_to_none=True)
    x, ids, weights = batch(step, rank)
    output = model(x.requires_grad_(), ids, weights)
    output.square().mean().backward()
    optimizer.step()


def check(mesh, selected, root, label):
    torch_dispatch.set_ep_group(mesh["ep"].get_group())
    reference = make_model(selected)
    reference_optimizer = torch.optim.AdamW(
        reference.parameters(), lr=0.01, foreach=False
    )
    model = make_model(selected, mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    for step in range(2):
        update(reference, reference_optimizer, step)
        update(model, optimizer, step, dist.get_rank())
    state = full_model_state(model)
    optimizer_state = full_optimizer_state(model, optimizer)
    adapter_dir = root / label
    assert save_ep_lora_adapter(model, str(adapter_dir), mesh["ep"].get_group())
    if dist.get_rank() == 0:
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(
                state[name], value, atol=1e-6, rtol=1e-5, msg=name
            )
        adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
        reference_adapter = get_peft_model_state_dict(reference)
        assert adapter.keys() == reference_adapter.keys()
        for name, value in reference_adapter.items():
            torch.testing.assert_close(
                adapter[name], value, atol=1e-6, rtol=1e-5, msg=name
            )
        torch.save((state, optimizer_state), root / f"{label}.pt")
    update(model, optimizer, 2, dist.get_rank())
    expected = copy.deepcopy(full_model_state(model))
    resumed = make_model(selected, mesh)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01, foreach=False)
    state, optimizer_state = (
        torch.load(root / f"{label}.pt", weights_only=True)
        if dist.get_rank() == 0
        else ({}, {})
    )
    restore_model_state(resumed, state)
    restore_optimizer_state(resumed, resumed_optimizer, optimizer_state)
    update(resumed, resumed_optimizer, 2, dist.get_rank())
    actual = full_model_state(resumed)
    if dist.get_rank() == 0:
        for name in expected:
            torch.testing.assert_close(
                actual[name], expected[name], atol=0, rtol=0, msg=name
            )
        print(f"PASS {label}", flush=True)
    dist.barrier()


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    experts_fn.register_all()
    experts_fn.set_backend("torch")
    experts_fn.set_local_implementation("eager")
    root = Path(sys.argv[1])
    root.mkdir(exist_ok=True)
    fsdp = init_device_mesh("cpu", (2, 2), mesh_dim_names=("shard", "ep"))
    hsdp = init_device_mesh(
        "cpu", (2, 1, 2), mesh_dim_names=("replicate", "shard", "ep")
    )
    for name, mesh in (("fsdp2_ep", fsdp), ("hsdp_ep", hsdp)):
        for selection in ([0, 1], [7, 0]):
            check(mesh, selection, root, f"{name}-{selection}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
