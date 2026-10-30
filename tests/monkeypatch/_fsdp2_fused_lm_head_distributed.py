"""CPU distributed gradient parity for TRL's fused LM head under axolotl's FSDP2 wrapping."""

from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate import FullyShardedDataParallelPlugin
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate
from transformers import Qwen3Config, Qwen3ForCausalLM
from trl.trainer.utils import add_fused_lm_head

from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model

VOCAB = 48


def make_model(kind, tp_mesh=None):
    torch.manual_seed(42)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=VOCAB,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=kind == "tied",
            attn_implementation="eager",
        )
    ).float()
    add_fused_lm_head(model)
    if tp_mesh is not None:
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            parallelize_module,
        )

        # transformers' tp_plan shards the head over the vocabulary at load, before
        # PEFT wraps it ("colwise_gather_output").
        parallelize_module(
            model.lm_head, tp_mesh, ColwiseParallel(output_layouts=Replicate())
        )
    if kind == "modules_to_save":
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=4, target_modules=["q_proj"], modules_to_save=["lm_head"]),
        )
    return model


def full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def fused_loss(model, ids):
    return model(input_ids=ids, labels=ids, fused_lm_head=True).loss


def run_case(kind, name, shape, names, fsdp_dims, policy):
    mesh = init_device_mesh("cpu", shape, mesh_dim_names=names)
    if "dp_replicate" not in fsdp_dims and len(fsdp_dims) > 1:
        mesh[fsdp_dims]._flatten(mesh_dim_name="fsdp")
        fsdp_dims = ("fsdp",)
    if "dp_replicate" in fsdp_dims and len(fsdp_dims) > 2:
        mesh[fsdp_dims[1:]]._flatten(mesh_dim_name="fsdp_shard")
        fsdp_dims = ("dp_replicate", "fsdp_shard")
    reference_mesh = mesh[fsdp_dims]
    if len(fsdp_dims) > 1:
        reference_mesh = reference_mesh._flatten(mesh_dim_name="reference_dp")
    plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        auto_wrap_policy=policy,
        min_num_params=1 if policy == "SIZE_BASED_WRAP" else None,
        transformer_cls_names_to_wrap=["Qwen3DecoderLayer"],
        reshard_after_forward=True,
        cpu_ram_efficient_loading=False,
    )
    accelerator = SimpleNamespace(
        state=SimpleNamespace(
            fsdp_plugin=plugin,
            device_mesh=mesh,
            parallelism_config=SimpleNamespace(fsdp_dim_names=fsdp_dims),
        ),
        is_main_process=dist.get_rank() == 0,
    )
    model = make_model(kind, mesh["tp"] if "tp" in names else None)
    reference = make_model(kind)
    model = fsdp2_prepare_model(accelerator, model)
    data_rank = reference_mesh.get_local_rank()
    ids = (torch.arange(12).view(2, 6) * (data_rank + 3)) % VOCAB
    fused_loss(model, ids).backward()
    fused_loss(reference, ids).backward()
    ref_params = dict(reference.named_parameters())
    group = reference_mesh.get_group()
    for key, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        expected = ref_params[key].grad
        dist.all_reduce(expected, group=group)
        expected.div_(dist.get_world_size(group))
        torch.testing.assert_close(
            full(parameter.grad),
            expected,
            atol=1e-6,
            rtol=1e-4,
            msg=lambda m, key=key: f"{kind} {name} {policy} {key}: {m}",
        )
    if dist.get_rank() == 0:
        print(f"PASS {kind} {name} {policy}", flush=True)


def main():
    dist.init_process_group("gloo")
    try:
        for kind in ("untied", "tied", "modules_to_save"):
            for policy in ("TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"):
                for name, shape, names, fsdp_dims in (
                    ("fsdp", (4,), ("dp_shard",), ("dp_shard",)),
                    (
                        "hsdp",
                        (2, 2),
                        ("dp_replicate", "dp_shard"),
                        ("dp_replicate", "dp_shard"),
                    ),
                    ("fsdp-cp", (2, 2), ("dp_shard", "cp"), ("dp_shard", "cp")),
                    ("fsdp-tp", (2, 2), ("dp_shard", "tp"), ("dp_shard",)),
                    ("fsdp-ep", (2, 2), ("dp_shard", "ep"), ("dp_shard", "ep")),
                    (
                        "hsdp-tp",
                        (2, 1, 2),
                        ("dp_replicate", "dp_shard", "tp"),
                        ("dp_replicate", "dp_shard"),
                    ),
                ):
                    if kind == "tied" and "tp" in names:
                        # Sharding a tied head over the vocabulary also shards the
                        # embedding, which then needs a plan of its own; that holds
                        # with or without the fused head.
                        continue
                    run_case(kind, name, shape, names, fsdp_dims, policy)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
