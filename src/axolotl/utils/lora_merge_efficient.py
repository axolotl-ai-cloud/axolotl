"""
Memory-efficient LoRA merging implementation inspired by qlora-pipe.
Processes model shards individually without loading the full model into memory.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors.torch
import torch
from peft import LoraConfig
from tqdm import tqdm

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def find_lora_weights(
    lora_state: Dict[str, torch.Tensor], key: str
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Find corresponding LoRA A and B weights for a given key.
    """
    clean_key = key.strip(".weight")
    clean_key = re.sub(r"^(base_model\.model\.|language_model\.)", "", clean_key)

    lora_a = None
    lora_b = None

    for lora_key, lora_weight in lora_state.items():
        if clean_key in lora_key:
            if "lora_A" in lora_key:
                lora_a = lora_weight
            elif "lora_B" in lora_key:
                lora_b = lora_weight

    if lora_a is not None and lora_b is not None:
        return lora_a, lora_b
    return None, None


def get_model_shards(model_path: Path) -> list[Path]:
    """Find all model shards in the given path."""
    shards = list[Path]()

    patterns = ["model*.safetensors", "model*.bin", "pytorch_model*.bin"]

    for pattern in patterns:
        shards.extend(model_path.glob(pattern))
        if shards:
            break

    return sorted(shards)


def copy_non_model_files(
    input_path: Path, output_path: Path, model_shards: list[Path]
) -> None:
    """
    Copy all non-model files to the output directory.

    Args:
        input_path: Source directory
        output_path: Destination directory
        model_shards: List of model shard files to skip
    """
    LOG.info("Copying non-model files to output directory...")

    shard_names = {shard.name for shard in model_shards}

    for filepath in input_path.glob("*"):
        if filepath.is_dir():
            continue
        if filepath.name in shard_names:
            continue
        if filepath.suffix == ".gguf":
            continue
        if filepath.name.startswith("model") and filepath.suffix == ".safetensors":
            continue

        LOG.debug(f"Copying {filepath.name} to output")
        shutil.copy(filepath, output_path)


def merge_lora_sharded_efficient(
    base_model_path: Union[str, Path],
    lora_adapter_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cuda",
    safe_tensors: bool = True,
) -> None:
    """
    Memory-efficient LoRA merging that processes shards individually
    without loading the full model into memory.
    """
    base_model_path = Path(base_model_path)
    lora_adapter_path = Path(lora_adapter_path)
    output_path = Path(output_path)

    if "/" in str(base_model_path) and not base_model_path.exists():
        from huggingface_hub import snapshot_download

        base_model_path = Path(snapshot_download(str(base_model_path)))

    os.makedirs(output_path, exist_ok=True)

    config_file = lora_adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"LoRA config not found: {config_file}")

    lora_config = LoraConfig.from_json_file(config_file)
    scale = lora_config.lora_alpha / lora_config.r

    LOG.info(f"LoRA scale factor: {scale}")

    lora_file = lora_adapter_path / "adapter_model.safetensors"
    if not lora_file.exists():
        lora_file = lora_adapter_path / "adapter_model.bin"
        if not lora_file.exists():
            raise FileNotFoundError(
                f"LoRA adapter weights not found in {lora_adapter_path}"
            )

    LOG.info(f"Loading LoRA weights from {lora_file}")

    if lora_file.suffix == ".safetensors":
        lora_state = safetensors.torch.load_file(lora_file)
    else:
        lora_state = torch.load(lora_file, map_location="cpu", weights_only=True)

    if device != "cpu":
        LOG.info(f"Moving LoRA weights to {device}")
        for key, value in tqdm(lora_state.items(), desc="Moving LoRA to device"):
            lora_state[key] = value.to(device)

    model_shards = get_model_shards(base_model_path)
    if not model_shards:
        raise FileNotFoundError(f"No model shards found in {base_model_path}")

    LOG.info(f"Found {len(model_shards)} model shards")
    copy_non_model_files(base_model_path, output_path, model_shards)

    merged_count = 0
    total_tensors = 0

    for shard_path in tqdm(model_shards, desc="Merging shards"):
        merged_tensors = {}
        metadata = {}

        if shard_path.suffix == ".safetensors":
            with safetensors.safe_open(shard_path, framework="pt", device=device) as f:
                if hasattr(f, "metadata") and f.metadata():
                    metadata = f.metadata()

                for key in f.keys():
                    total_tensors += 1
                    tensor = f.get_tensor(key)
                    lora_a, lora_b = find_lora_weights(lora_state, key)

                    if lora_a is not None and lora_b is not None:
                        merged_count += 1
                        LOG.debug(
                            f"Merging LoRA for {key}: {lora_a.shape}, {lora_b.shape}"
                        )

                        original_dtype = tensor.dtype
                        tensor_fp32 = tensor.to(torch.float32)

                        delta = scale * (
                            lora_b.to(torch.float32) @ lora_a.to(torch.float32)
                        )

                        merged_tensor = (tensor_fp32 + delta).to(original_dtype)
                        merged_tensors[key] = merged_tensor
                    else:
                        merged_tensors[key] = tensor
        else:
            state_dict = torch.load(
                shard_path, map_location=device
            )  # nosec B614: loading trusted model weights
            for key, tensor in state_dict.items():
                total_tensors += 1
                lora_a, lora_b = find_lora_weights(lora_state, key)

                if lora_a is not None and lora_b is not None:
                    merged_count += 1
                    original_dtype = tensor.dtype
                    tensor_fp32 = tensor.to(torch.float32)
                    delta = scale * (
                        lora_b.to(torch.float32) @ lora_a.to(torch.float32)
                    )
                    merged_tensors[key] = (tensor_fp32 + delta).to(original_dtype)
                else:
                    merged_tensors[key] = tensor

        output_shard_path = output_path / shard_path.name
        if safe_tensors and shard_path.suffix == ".safetensors":
            safetensors.torch.save_file(
                merged_tensors, output_shard_path, metadata=metadata
            )
        else:
            if safe_tensors:
                output_shard_path = output_shard_path.with_suffix(".safetensors")
            torch.save(merged_tensors, output_shard_path)

        del merged_tensors
        if device != "cpu":
            torch.cuda.empty_cache()

    LOG.info(f"Applied LoRA to {merged_count}/{total_tensors} tensors")
