"""A tiny modelopt-NVFP4 MoE base and Beam-shaped adapters, built on CPU."""

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import load_file, save_file

E, H, I, R, ALPHA = 4, 64, 16, 4, 8
LAYERS = (0, 1, 2)
LORA_EXPERT_LAYERS = (0, 1)
PROJECTIONS = (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I)))
INDEX = "model.safetensors.index.json"


def quantize(weight):
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    scale = (weight.abs().max() / (6.0 * 448.0)).reshape(()).clamp_min(1e-12)
    nv = NVFP4Tensor.to_nvfp4(
        weight.float(), per_tensor_scale=scale, is_swizzled_scales=False
    )
    return (
        nv.qdata.clone(),
        nv.scale.to(torch.float8_e4m3fn).clone(),
        nv.per_tensor_scale.reshape(()).clone(),
    )


def experts(layer):
    generator = torch.Generator().manual_seed(100 + layer)
    tensors = {}
    for proj, (rows, columns) in PROJECTIONS:
        for expert in range(E):
            weight = torch.randn(rows, columns, generator=generator) * columns**-0.5
            qdata, scale, scale_2 = quantize(weight)
            stem = f"model.layers.{layer}.mlp.experts.{expert}.{proj}"
            tensors[f"{stem}.weight"] = qdata
            tensors[f"{stem}.weight_scale"] = scale
            tensors[f"{stem}.weight_scale_2"] = scale_2
            tensors[f"{stem}.input_scale"] = torch.rand((), generator=generator)
    return tensors


def dense(layer):
    generator = torch.Generator().manual_seed(200 + layer)
    stem = f"model.layers.{layer}"
    return {
        f"{stem}.self_attn.q_proj.weight": torch.randn(H, H, generator=generator).to(
            torch.bfloat16
        ),
        f"{stem}.self_attn.o_proj.weight": torch.randn(H, H, generator=generator).to(
            torch.bfloat16
        ),
        f"{stem}.mlp.gate.weight": torch.randn(E, H, generator=generator),
        f"{stem}.input_layernorm.weight": torch.ones(H, dtype=torch.bfloat16),
    }


def build_base(directory: Path) -> Path:
    """Two shards; layer 1's experts are split across the boundary, layer 2 has no LoRA."""
    pytest.importorskip("torchao")
    directory.mkdir(parents=True)
    generator = torch.Generator().manual_seed(7)
    layer1 = experts(1)

    def split(first):
        return {
            key: value
            for key, value in layer1.items()
            if (int(key.split(".experts.")[1].split(".")[0]) < 2) == first
        }

    shards = {
        "model-00001-of-00002.safetensors": {
            "model.embed_tokens.weight": torch.randn(32, H, generator=generator).to(
                torch.bfloat16
            ),
            **dense(0),
            **experts(0),
            **dense(1),
            **split(True),
        },
        "model-00002-of-00002.safetensors": {
            **split(False),
            **dense(2),
            **experts(2),
            "model.norm.weight": torch.ones(H, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, H, generator=generator).to(
                torch.bfloat16
            ),
        },
    }
    weight_map = {}
    for name, tensors in shards.items():
        save_file(tensors, directory / name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(tensors, name))
    (directory / INDEX).write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "synthetic-test",
                "num_experts": E,
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": ["*self_attn*", "*mlp.gate", "lm_head"],
                },
            }
        )
    )
    return directory


def save_adapter(
    directory: Path, step: int, *, merge_aware: bool = False, alpha: int = ALPHA
) -> Path:
    """Attention q_proj LoRA in every layer plus fused expert LoRA in layers 0-1."""
    generator = torch.Generator().manual_seed(1000 + step)

    def randn(*shape):
        return (torch.randn(*shape, generator=generator) * 0.2).to(torch.bfloat16)

    tensors = {}
    for layer in LAYERS:
        stem = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        tensors[f"{stem}.lora_A.weight"] = randn(R, H)
        tensors[f"{stem}.lora_B.weight"] = randn(H, R)
    for layer in LORA_EXPERT_LAYERS:
        stem = f"base_model.model.model.layers.{layer}.mlp.experts"
        tensors[f"{stem}.lora_A.weight"] = randn(R * E, I)
        tensors[f"{stem}.lora_B.weight"] = randn(H, R * E)
        tensors[f"{stem}.base_layer.lora_A.weight"] = randn(R * E, H)
        tensors[f"{stem}.base_layer.lora_B.weight"] = randn(2 * I, R * E)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(tensors, directory / "adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "r": R,
        "lora_alpha": alpha,
        "target_modules": ["q_proj"],
        "target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    }
    if merge_aware:
        config["nvfp4_merge_aware"] = True
    (directory / "adapter_config.json").write_text(json.dumps(config))
    return directory


def load_checkpoint(directory: Path) -> dict[str, torch.Tensor]:
    weights = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        weights.update(load_file(path))
    return weights


def apply_delta(path: Path, previous: dict[str, torch.Tensor]):
    """Reconstruct a delta as the hot-load server does, checking each Adler-32."""
    result = {key: tensor.clone() for key, tensor in previous.items()}
    index = json.loads((path / INDEX).read_text())
    assert "delta_encoding" in index["metadata"]
    for name in set(index["weight_map"].values()):
        with safe_open(path / name, framework="numpy") as shard:
            for key in shard.keys():
                delta = np.frombuffer(
                    zstandard.ZstdDecompressor().decompress(shard.get_tensor(key)),
                    dtype=np.uint8,
                )
                target = result[key].reshape(-1).view(torch.uint8).numpy()
                target[:] ^= delta
                assert shard.metadata()[key] == f"{zlib.adler32(target):08x}"
    return result


def assert_bytes_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        assert actual[key].dtype == tensor.dtype and actual[key].shape == tensor.shape
        assert torch.equal(
            actual[key].reshape(-1).view(torch.uint8),
            tensor.reshape(-1).view(torch.uint8),
        ), key
