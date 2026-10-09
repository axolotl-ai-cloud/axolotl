"""Exercise real FSDP2 expert checkpoints with ordinary AdamW on CPU."""

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
LORA_RANK = 2


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _build(ep, dp_shard, seed, checkpoint_layers, lora):
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

    torch.manual_seed(0 if lora else seed)  # a resumed LoRA run keeps its base
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

    decoder = model.model
    if lora:
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(100 + seed)  # LoRA A init; B starts at zero
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=4,
                target_modules=["q_proj"],
                target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
            ),
        )
        assert shard.shard_expert_lora(model, ep) > 0
    if checkpoint_layers:
        for i, layer in enumerate(decoder.layers):
            decoder.layers[i] = checkpoint_wrapper(layer)
    expert_mesh = (
        expert_fsdp_mesh(mesh)
        if mesh is not None
        else per_rank_expert_mesh(None, "cpu")
    )
    ExpertParallelPlugin.fully_shard_experts(model, expert_mesh, {})
    for layer in decoder.layers:
        fully_shard(layer, mesh=dense_mesh)
    fully_shard(model, mesh=dense_mesh)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.0
    )
    return model, optimizer, ep_group


def _train(model, optimizer, steps):
    generator = torch.Generator().manual_seed(1000 + dist.get_rank())
    for _ in range(steps):
        for param in (p for p in model.parameters() if p.requires_grad):
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
        n: {
            k: _local(v).detach().clone() for k, v in optimizer.state.get(p, {}).items()
        }
        for n, p in model.named_parameters()
    }
    return params, states


def _full_tensors(model, optimizer, ep_group):
    """The true full tensors of every EP-sharded parameter (experts and expert LoRA) and
    of their Adam moments, keyed ``(fqn, "weight" | state)``: gathered over the non-ep
    mesh, then along ep the way the final model / adapter export assembles them."""
    from torch.distributed.checkpoint.state_dict import _get_fqns

    from axolotl.integrations.expert_parallel.shard import gather_expert_lora_full

    def gather(name, tensor):
        whole = tensor.full_tensor().contiguous()
        if "lora_B" in name:
            return gather_expert_lora_full(whole, "B", NUM_EXPERTS, ep_group)
        chunks = [torch.empty_like(whole) for _ in range(dist.get_world_size(ep_group))]
        dist.all_gather(chunks, whole, group=ep_group)
        return torch.cat(chunks)

    full = {}
    for name, param in model.named_parameters():
        if ".experts." not in name:
            continue
        (fqn,) = _get_fqns(model, name)
        full[(fqn, "weight")] = (param.requires_grad, gather(name, param))
        for key in ("exp_avg", "exp_avg_sq"):
            if key in optimizer.state.get(param, {}):
                full[(fqn, key)] = (True, gather(name, optimizer.state[param][key]))
    return full


def _io(mode):
    import transformers.trainer as hf_trainer

    from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
        patch_fsdp2_full_checkpoint,
    )

    @contextlib.contextmanager
    def checkpoint_io():
        if mode == "fixed":
            patch_fsdp2_full_checkpoint()
        yield hf_trainer

    return checkpoint_io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--dp-shard", type=int, default=1)
    parser.add_argument("--mode", choices=("fixed", "accelerate", "legacy"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--lora", action="store_true")
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

    model, optimizer, ep_group = _build(
        args.ep, args.dp_shard, 0, checkpoint_layers, args.lora
    )
    ep_rank = dist.get_rank(ep_group)
    _train(model, optimizer, 2)
    before_params, before_states = _snapshot(model, optimizer)
    full = _full_tensors(model, optimizer, ep_group)

    save_io = _io("fixed" if args.mode == "fixed" else "accelerate")
    if args.lora:  # the trainer's _save_gathered_lora_adapter
        from axolotl.integrations.expert_parallel.shard import save_ep_lora_adapter

        assert save_ep_lora_adapter(model, checkpoint, ep_group)
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
        for (fqn, key), (saved_here, value) in full.items():
            if not saved_here:
                continue  # the frozen base of a LoRA run is not checkpointed
            source = model_file if key == "weight" else optim_file["state"][fqn]
            saved = source.get(fqn if key == "weight" else key)
            if saved is None:
                report["mismatch"].append(["file", fqn, key + " missing"])
                continue
            report["file"].append([fqn, key, list(saved.shape)])
            if saved.shape != value.shape or not torch.equal(saved, value):
                report["mismatch"].append(["file", fqn, key])
        if args.lora:
            from safetensors.torch import load_file

            adapter = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
            report["adapter"] = [
                [k, list(v.shape)]
                for k, v in adapter.items()
                if ".experts." in k and "lora_" in k
            ]

    fresh, fresh_optimizer, _ = _build(
        args.ep, args.dp_shard, 1, checkpoint_layers, args.lora
    )
    load_io = _io("fixed" if args.mode in ("fixed", "legacy") else "accelerate")
    fresh_before, _ = _snapshot(fresh, fresh_optimizer)
    with load_io() as io:
        try:
            io.load_fsdp_model(
                fsdp_plugin, accelerator, fresh, checkpoint, adapter_only=True
            )
            io.load_fsdp_optimizer(
                fsdp_plugin, accelerator, fresh_optimizer, fresh, checkpoint
            )
        except (RuntimeError, ValueError) as exc:
            report["load_error"] = str(exc)

    if "load_error" in report:
        after_params, _ = _snapshot(fresh, fresh_optimizer)
        report["unchanged"] = all(
            torch.equal(after_params[name], value)
            for name, value in fresh_before.items()
        )

    if "load_error" not in report:
        after_params, after_states = _snapshot(fresh, fresh_optimizer)
        for name, value in before_params.items():
            if not torch.equal(after_params[name], value):
                report["mismatch"].append(["param", name, "weight"])
            for key, state in before_states[name].items():
                if not torch.equal(after_states[name][key], state):
                    report["mismatch"].append(["param", name, key])

    if args.mode == "fixed":
        from transformers import Trainer

        with torch.no_grad():
            for parameter in fresh.parameters():
                if parameter.requires_grad:
                    _local(parameter).zero_()
        trainer = object.__new__(Trainer)
        trainer.model = fresh
        trainer.is_fsdp_enabled = True
        trainer.is_deepspeed_enabled = False
        trainer.accelerator = accelerator
        accelerator.state = SimpleNamespace(fsdp_plugin=fsdp_plugin)
        trainer.state = SimpleNamespace(
            best_model_checkpoint=checkpoint, best_metric=0.0
        )
        trainer._load_best_model()
        best_params, _ = _snapshot(fresh, fresh_optimizer)
        report["best_model_matches"] = all(
            torch.equal(best_params[name], value)
            for name, value in before_params.items()
        )

    reports = [None] * world
    dist.all_gather_object(reports, report)
    if rank == 0:
        print("EP_CHECKPOINT_REPORT " + json.dumps(reports), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
