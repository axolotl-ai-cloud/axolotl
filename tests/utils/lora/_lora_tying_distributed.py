"""CPU distributed gradient and checkpoint parity for tied embedding LoRA."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate import FullyShardedDataParallelPlugin
from peft import LoraConfig, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from transformers import Qwen2Config, Qwen2ForCausalLM

from axolotl.monkeypatch.accelerate.fsdp2 import fsdp2_prepare_model
from axolotl.utils.lora_tying import tie_lora_output_embeddings


def make_model(dtype=torch.float32):
    torch.manual_seed(42)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            tie_word_embeddings=True,
            attn_implementation="eager",
        )
    )
    model = get_peft_model(
        model.to(dtype),
        LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["embed_tokens", "lm_head", "q_proj"],
            ensure_weight_tying=True,
        ),
    )
    tie_lora_output_embeddings(model)
    with torch.no_grad():
        model.get_input_embeddings().lora_embedding_A["default"].normal_(std=0.1)
    return model


def full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def run_case(name, shape, names, fsdp_dims, policy, dtype):
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
        transformer_cls_names_to_wrap=["Qwen2DecoderLayer"],
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
    model = make_model(dtype)
    reference = make_model(dtype)
    if "tp" in names:
        from torch.distributed.tensor import Replicate
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            parallelize_module,
        )

        projection = model.base_model.model.model.layers[0].self_attn.q_proj
        for linear in (projection.base_layer, projection.lora_B["default"]):
            parallelize_module(
                linear, mesh["tp"], ColwiseParallel(output_layouts=Replicate())
            )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = fsdp2_prepare_model(accelerator, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.01)
    data_rank = reference_mesh.get_local_rank()
    ids = (torch.arange(4).view(1, 4) + data_rank + 1) % 32
    group = reference_mesh.get_group()
    for _ in range(2):
        optimizer.zero_grad()
        ref_optimizer.zero_grad()
        for microbatch in range(2):
            model.set_requires_gradient_sync(microbatch == 1)
            batch = (ids + microbatch) % 32
            model(batch).logits.square().mean().backward()
            reference(batch).logits.square().mean().backward()
        ref_params = dict(reference.named_parameters())
        for key, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            expected = ref_params[key]
            dist.all_reduce(expected.grad, group=group)
            expected.grad.div_(dist.get_world_size(group))
            torch.testing.assert_close(
                full(parameter.grad), expected.grad, atol=1e-6, rtol=1e-4
            )
        optimizer.step()
        ref_optimizer.step()
        for key, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(
                    full(parameter), ref_params[key], atol=1e-6, rtol=1e-4
                )
    from safetensors.torch import load_file

    from axolotl.integrations.expert_parallel.shard import save_fsdp2_lora_adapter

    output = Path(sys.argv[1]) / f"{name}-{policy}-{dtype}"
    assert save_fsdp2_lora_adapter(model, str(output))
    dist.barrier()
    saved = load_file(output / "adapter_model.safetensors")
    embeddings = reference.get_input_embeddings()
    for head_name, weight in (
        ("lora_A", embeddings.lora_embedding_B["default"]),
        ("lora_B", embeddings.lora_embedding_A["default"]),
    ):
        torch.testing.assert_close(
            saved[f"base_model.model.lm_head.{head_name}.weight"],
            weight.t(),
            atol=1e-6,
            rtol=1e-4,
        )
    if dist.get_rank() == 0:
        print(f"PASS {name} {policy} {dtype}", flush=True)


def main():
    dist.init_process_group("gloo")
    try:
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
                (
                    "hsdp-cp",
                    (2, 1, 2),
                    ("dp_replicate", "dp_shard", "cp"),
                    ("dp_replicate", "dp_shard", "cp"),
                ),
            ):
                for dtype in (torch.float32, torch.bfloat16):
                    run_case(name, shape, names, fsdp_dims, policy, dtype)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
