"""A tiny Qwen3.6-style modelopt MIXED_PRECISION checkpoint and adapters, built on CPU.

Mirrors nvidia/Qwen3.6-35B-A3B-NVFP4: ``Qwen3_5MoeForConditionalGeneration`` with the text
weights under ``model.language_model.``; W4A16 NVFP4 routed experts (per expert), shared
expert and ``lm_head``; static per-tensor FP8 attention (Gated DeltaNet ``linear_attn`` and
full ``self_attn``); bf16 vision tower, router and an ``mtp.`` head.
"""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

E, H, I, R, ALPHA = 4, 64, 32, 4, 8
VOCAB = 128
LAYERS = (0, 1, 2, 3)
LAYER_TYPES = ("linear_attention", "full_attention") * 2
TEXT = "model.language_model"
INDEX = "model.safetensors.index.json"
FP8_LINEARS = {
    "linear_attention": (
        "linear_attn.in_proj_qkv",
        "linear_attn.in_proj_z",
        "linear_attn.out_proj",
    ),
    "full_attention": (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
    ),
}
SHARED = ("gate_proj", "up_proj", "down_proj")


def config() -> dict:
    from transformers import Qwen3_5MoeConfig

    text = {
        "vocab_size": VOCAB,
        "hidden_size": H,
        "num_hidden_layers": len(LAYERS),
        "layer_types": list(LAYER_TYPES),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "moe_intermediate_size": I,
        "shared_expert_intermediate_size": I,
        "num_experts": E,
        "num_experts_per_tok": 2,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "full_attention_interval": 2,
        "tie_word_embeddings": False,
    }
    vision = {
        "depth": 1,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": H,
    }
    return Qwen3_5MoeConfig(
        text_config=text, vision_config=vision, tie_word_embeddings=False
    ).to_dict()


def quantize_nvfp4(weight, scale_2=None):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    if scale_2 is None:
        scale_2 = weight.abs().max() / (6.0 * 448.0)
    scale_2 = torch.as_tensor(scale_2, dtype=torch.float32).reshape(()).clamp_min(1e-12)
    nv = NVFP4Tensor.to_nvfp4(
        weight.float(), per_tensor_scale=scale_2, is_swizzled_scales=False
    )
    return {
        "weight": nv.qdata.clone(),
        "weight_scale": nv.scale.to(torch.float8_e4m3fn).clone(),
        "weight_scale_2": nv.per_tensor_scale.reshape(()).clone(),
    }


def quantize_fp8(weight):
    scale = (weight.abs().max().float() / 448.0).reshape(())
    return {
        "weight": (weight.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn),
        "weight_scale": scale.clone(),
    }


def reference_state() -> dict[str, torch.Tensor]:
    """A randomly initialized bf16 model's weights, as the quantizer's input."""
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    torch.manual_seed(0)
    model = Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(**config()))
    state = {k: v.detach().to(torch.bfloat16) for k, v in model.state_dict().items()}
    generator = torch.Generator().manual_seed(1)
    for key, value in state.items():
        # the default init leaves the projections nearly zero-mean noise of 0.02, fine for
        # scales, but make the experts distinct enough that gate/up per-tensor scales differ
        if key.endswith(("gate_up_proj", "down_proj")) and value.ndim == 3:
            noise = torch.randn(value.shape, generator=generator)
            state[key] = (noise * value.shape[-1] ** -0.5).to(torch.bfloat16)
    return state


def build_base(directory: Path) -> Path:
    """Two shards; layer 1's experts and attention straddle the boundary."""
    pytest.importorskip("torchao")
    directory.mkdir(parents=True)
    state = reference_state()
    tensors: dict[str, torch.Tensor] = {}
    generator = torch.Generator().manual_seed(2)

    def input_scale():
        return torch.rand((), generator=generator) + 0.5

    for key, value in state.items():
        if not key.startswith(f"{TEXT}.layers.") and key != "lm_head.weight":
            tensors[key] = value
    for layer, layer_type in zip(LAYERS, LAYER_TYPES, strict=True):
        stem = f"{TEXT}.layers.{layer}"
        for key, value in state.items():
            if key.startswith(f"{stem}.") and not any(
                key.startswith(f"{stem}.{module}.")
                for module in (
                    *FP8_LINEARS[layer_type],
                    "mlp.experts",
                    "mlp.shared_expert",
                )
            ):
                tensors[key] = value
        for module in FP8_LINEARS[layer_type]:
            for leaf, t in quantize_fp8(state[f"{stem}.{module}.weight"]).items():
                tensors[f"{stem}.{module}.{leaf}"] = t
            tensors[f"{stem}.{module}.input_scale"] = input_scale()
        for proj in SHARED:
            module = f"{stem}.mlp.shared_expert.{proj}"
            for leaf, t in quantize_nvfp4(state[f"{module}.weight"]).items():
                tensors[f"{module}.{leaf}"] = t
            tensors[f"{module}.input_scale"] = input_scale()
        gate_up = state[f"{stem}.mlp.experts.gate_up_proj"]
        down = state[f"{stem}.mlp.experts.down_proj"]
        for expert in range(E):
            # gate and up get different per-tensor scales, as modelopt exports them
            for proj, weight in (
                ("gate_proj", gate_up[expert, :I]),
                ("up_proj", gate_up[expert, I:] * 0.5),
                ("down_proj", down[expert]),
            ):
                module = f"{stem}.mlp.experts.{expert}.{proj}"
                for leaf, t in quantize_nvfp4(weight).items():
                    tensors[f"{module}.{leaf}"] = t
                tensors[f"{module}.input_scale"] = input_scale()
    for leaf, t in quantize_nvfp4(state["lm_head.weight"]).items():
        tensors[f"lm_head.{leaf}"] = t
    tensors["lm_head.input_scale"] = input_scale()
    tensors["mtp.fc.weight"] = torch.randn(H, 2 * H, generator=generator).to(
        torch.bfloat16
    )
    tensors["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.randn(
        E, 2 * I, H, generator=generator
    ).to(torch.bfloat16)

    first = {
        k: v
        for k, v in tensors.items()
        if k.startswith(("model.visual.", f"{TEXT}.embed_tokens", f"{TEXT}.layers.0."))
        or k.startswith(f"{TEXT}.layers.1.linear_attn")
        or k.startswith(f"{TEXT}.layers.1.self_attn.q_proj")
        or (
            k.startswith(f"{TEXT}.layers.1.mlp.experts.")
            and int(k.split(".experts.")[1].split(".")[0]) < 2
        )
    }
    shards = {
        "model-00001-of-00002.safetensors": first,
        "model-00002-of-00002.safetensors": {
            k: v for k, v in tensors.items() if k not in first
        },
    }
    weight_map = {}
    for name, part in shards.items():
        save_file(part, directory / name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(part, name))
    (directory / INDEX).write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    quantized: dict[str, dict] = {}
    for layer, layer_type in zip(LAYERS, LAYER_TYPES, strict=True):
        stem = f"{TEXT}.layers.{layer}"
        for module in FP8_LINEARS[layer_type]:
            quantized[f"{stem}.{module}"] = {"quant_algo": "FP8"}
        nvfp4 = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
        quantized[f"{stem}.mlp.experts"] = nvfp4
        for proj in SHARED:
            quantized[f"{stem}.mlp.shared_expert.{proj}"] = nvfp4
    quantized["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    cfg = config()
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    cfg["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": quantized,
        "ignore": ["mtp*"],
        "producer": {"name": "modelopt", "version": "0.37.0"},
    }
    (directory / "config.json").write_text(json.dumps(cfg))
    return directory


# (out_features, in_features) of each adapted linear
LINEAR_SHAPES = {
    "in_proj_qkv": (128, H),
    "in_proj_z": (64, H),
    "out_proj": (H, 64),
    "q_proj": (128, H),
    "k_proj": (32, H),
    "v_proj": (32, H),
    "o_proj": (H, 64),
    "gate_proj": (I, H),
    "up_proj": (I, H),
    "down_proj": (H, I),
}


def lora_modules() -> list[str]:
    """Every linear the example config adapts: FP8 attention and the NVFP4 shared expert."""
    modules = []
    for layer, layer_type in zip(LAYERS, LAYER_TYPES, strict=True):
        stem = f"{TEXT}.layers.{layer}"
        modules += [f"{stem}.{m}" for m in FP8_LINEARS[layer_type]]
        modules += [f"{stem}.mlp.shared_expert.{p}" for p in SHARED]
    return modules


def save_adapter(
    directory: Path, step: int, *, merge_aware: bool = True, alpha: int = ALPHA
) -> Path:
    """Attention, shared-expert and fused expert LoRA in every layer, as PEFT saves it."""
    generator = torch.Generator().manual_seed(1000 + step)

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.2).to(torch.bfloat16)

    tensors = {}
    for module in lora_modules():
        rows, columns = LINEAR_SHAPES[module.rsplit(".", 1)[1]]
        stem = f"base_model.model.{module}"
        tensors[f"{stem}.lora_A.weight"] = randn(R, columns)
        tensors[f"{stem}.lora_B.weight"] = randn(rows, R)
    for layer in LAYERS:
        stem = f"base_model.model.{TEXT}.layers.{layer}.mlp.experts"
        tensors[f"{stem}.lora_A.weight"] = randn(R * E, I)
        tensors[f"{stem}.lora_B.weight"] = randn(H, R * E)
        tensors[f"{stem}.base_layer.lora_A.weight"] = randn(R * E, H)
        tensors[f"{stem}.base_layer.lora_B.weight"] = randn(2 * I, R * E)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "adapter_model.safetensors")
    adapter_config = {
        "peft_type": "LORA",
        "r": R,
        "lora_alpha": alpha,
        "target_modules": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    }
    if merge_aware:
        adapter_config["nvfp4_merge_aware"] = True
    (directory / "adapter_config.json").write_text(json.dumps(adapter_config))
    return directory
