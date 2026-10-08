"""Gloo worker: an FSDP2 FULL_STATE_DICT checkpoint round-trips every EP rank's experts.

Builds a tiny Mixtral, EP-shards its experts and FSDP2-wraps it the way the plugin does
(experts on their non-ep mesh, dense layers on the whole world), takes two AdamW steps,
saves the model and optimizer through ``transformers.trainer``'s ``save_fsdp_*`` names
(the functions the Trainer calls), reloads them into a freshly built model and optimizer
through the ``load_fsdp_*`` names, and compares every rank's parameters and optimizer
state with the values before the save.

``--mode fixed`` runs inside ``ep_fsdp_checkpoint_functions`` (the trainer's EP path);
``--mode accelerate`` runs accelerate's functions as they are, to show the lost experts;
``--mode legacy`` saves with accelerate and loads with the fixed functions, which must
refuse the checkpoint on every rank.
"""

import argparse
import contextlib
import json
import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

NUM_EXPERTS = 8


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _build(ep, dp_shard, seed, checkpoint_layers):
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
    )
    from transformers import MixtralConfig, MixtralForCausalLM

    from axolotl.integrations.expert_parallel import shard
    from axolotl.integrations.expert_parallel.plugin import (
        ExpertParallelPlugin,
        expert_fsdp_mesh,
        per_rank_expert_mesh,
    )

    torch.manual_seed(seed)
    config = MixtralConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    model = MixtralForCausalLM(config)

    world = dist.get_world_size()
    if dp_shard > 1:
        # accelerate's axis order: ep innermost
        mesh = init_device_mesh(
            "cpu", (dp_shard, ep), mesh_dim_names=("dp_shard", "ep")
        )
        ep_group = mesh["ep"].get_group()
        dense_mesh = mesh[("dp_shard", "ep")]._flatten("dp_shard_ep")
    else:
        mesh = None
        ep_group = dist.group.WORLD
        dense_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))

    # every rank built the same weights; slice this rank's block in place of the CUDA scatter
    ep_rank = dist.get_rank(ep_group)

    def slice_on_cpu(module, name, count, _ranks):
        shard._replace_with_slice(module, name, ep_rank * count, (ep_rank + 1) * count)

    original = shard._scatter_expert_from_rank0
    shard._scatter_expert_from_rank0 = slice_on_cpu
    try:
        assert shard.shard_expert_weights(model, ep_group) == 2
    finally:
        shard._scatter_expert_from_rank0 = original

    if checkpoint_layers:
        for i, layer in enumerate(model.model.layers):
            model.model.layers[i] = checkpoint_wrapper(layer)
    expert_mesh = (
        expert_fsdp_mesh(mesh)
        if mesh is not None
        else per_rank_expert_mesh(None, "cpu")
    )
    ExpertParallelPlugin.fully_shard_experts(model, expert_mesh, {})
    for layer in model.model.layers:
        fully_shard(layer, mesh=dense_mesh)
    fully_shard(model, mesh=dense_mesh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    return model, optimizer, ep_group


def _train(model, optimizer, steps):
    generator = torch.Generator().manual_seed(1000 + dist.get_rank())
    for _ in range(steps):
        for param in model.parameters():
            local = torch.randn(_local(param).shape, generator=generator)
            param.grad = (
                DTensor.from_local(
                    local,
                    param.device_mesh,
                    param.placements,
                    shape=param.shape,
                    stride=param.stride(),
                )
                if isinstance(param, DTensor)
                else local
            )
        optimizer.step()
        optimizer.zero_grad()


def _snapshot(model, optimizer):
    params = {n: _local(p).detach().clone() for n, p in model.named_parameters()}
    states = {
        n: {k: _local(v).detach().clone() for k, v in optimizer.state[p].items()}
        for n, p in model.named_parameters()
    }
    return params, states


def _full_experts(model, optimizer, ep_group):
    """The true full expert tensors (all ranks take part; meaningful everywhere)."""
    from axolotl.integrations.expert_parallel.checkpoint import (
        all_gather_ep_experts,
        ep_sharded_expert_params,
    )

    weights, moments = {}, {}
    for expert in ep_sharded_expert_params(model):
        weights[expert.fqn] = all_gather_ep_experts(
            expert.param.full_tensor(), ep_group
        )
        for key in ("exp_avg", "exp_avg_sq"):
            moments[(expert.fqn, key)] = all_gather_ep_experts(
                optimizer.state[expert.param][key].full_tensor(), ep_group
            )
    return weights, moments


def _io(mode, ep_group):
    import transformers.trainer as hf_trainer

    from axolotl.integrations.expert_parallel.checkpoint import (
        ep_fsdp_checkpoint_functions,
    )

    @contextlib.contextmanager
    def fixed():
        with ep_fsdp_checkpoint_functions(ep_group):
            yield hf_trainer

    @contextlib.contextmanager
    def unpatched():
        yield hf_trainer

    return fixed if mode == "fixed" else unpatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--dp-shard", type=int, default=1)
    parser.add_argument("--mode", choices=("fixed", "accelerate", "legacy"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from accelerate import PartialState
    from accelerate.utils import FullyShardedDataParallelPlugin

    dist.init_process_group("gloo")
    PartialState()  # accelerate's checkpoint functions log through it
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == args.ep * args.dp_shard
    checkpoint_layers = args.dp_shard > 1
    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2, state_dict_type="FULL_STATE_DICT"
    )
    accelerator = SimpleNamespace(
        num_processes=world,
        process_index=rank,
        is_main_process=rank == 0,
        is_fsdp2=True,
        wait_for_everyone=dist.barrier,
    )
    checkpoint = os.path.join(args.out, "checkpoint-2")

    model, optimizer, ep_group = _build(args.ep, args.dp_shard, 0, checkpoint_layers)
    ep_rank = dist.get_rank(ep_group)
    _train(model, optimizer, 2)
    before_params, before_states = _snapshot(model, optimizer)
    full_weights, full_moments = _full_experts(model, optimizer, ep_group)

    save_io = _io("fixed" if args.mode == "fixed" else "accelerate", ep_group)
    with save_io() as io:
        io.save_fsdp_model(
            fsdp_plugin, accelerator, model, checkpoint, adapter_only=True
        )
        io.save_fsdp_optimizer(fsdp_plugin, accelerator, optimizer, model, checkpoint)
    dist.barrier()

    report = {"rank": rank, "ep_rank": ep_rank, "file": [], "mismatch": []}
    if rank == 0:
        model_file = torch.load(
            os.path.join(checkpoint, "pytorch_model_fsdp.bin"), weights_only=True
        )
        optim_file = torch.load(
            os.path.join(checkpoint, "optimizer.bin"), weights_only=True
        )
        for fqn, full in full_weights.items():
            saved = model_file[fqn]
            report["file"].append([fqn, "weight", list(saved.shape)])
            if saved.shape != full.shape or not torch.equal(saved, full):
                report["mismatch"].append(["file", fqn, "weight"])
        for (fqn, key), full in full_moments.items():
            saved = optim_file["state"][fqn][key]
            report["file"].append([fqn, key, list(saved.shape)])
            if saved.shape != full.shape or not torch.equal(saved, full):
                report["mismatch"].append(["file", fqn, key])

    fresh, fresh_optimizer, _ = _build(args.ep, args.dp_shard, 1, checkpoint_layers)
    load_io = _io(
        "fixed" if args.mode in ("fixed", "legacy") else "accelerate", ep_group
    )
    with load_io() as io:
        try:
            io.load_fsdp_model(
                fsdp_plugin, accelerator, fresh, checkpoint, adapter_only=True
            )
            io.load_fsdp_optimizer(
                fsdp_plugin, accelerator, fresh_optimizer, fresh, checkpoint
            )
        except RuntimeError as exc:
            report["load_error"] = str(exc)

    if "load_error" not in report:
        after_params, after_states = _snapshot(fresh, fresh_optimizer)
        for name, value in before_params.items():
            if not torch.equal(after_params[name], value):
                report["mismatch"].append(["param", name, "weight"])
            for key, state in before_states[name].items():
                if not torch.equal(after_states[name][key], state):
                    report["mismatch"].append(["param", name, key])

    reports = [None] * world
    dist.all_gather_object(reports, report)
    if rank == 0:
        print("EP_CHECKPOINT_REPORT " + json.dumps(reports), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
