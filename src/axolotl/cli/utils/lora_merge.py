import gc
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors
import safetensors.torch
import torch
from huggingface_hub import snapshot_download
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
    clean_key = key[:-7] if key.endswith(".weight") else key

    a_key = f"base_model.model.{clean_key}.lora_A.weight"
    b_key = f"base_model.model.{clean_key}.lora_B.weight"

    lora_a = lora_state.get(a_key)
    lora_b = lora_state.get(b_key)

    if lora_a is not None and lora_b is not None:
        return lora_a, lora_b
    return None, None


def get_model_shards(model_path: Path) -> list[Path]:
    """Find all model shards in the given path."""
    shards: list[Path] = []

    patterns = ["model*.safetensors", "pytorch_model*.bin"]

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
        if (
            filepath.name.startswith("model") and filepath.suffix == ".safetensors"
        ) or (filepath.name.startswith("pytorch_model") and filepath.suffix == ".bin"):
            continue
        if filepath.suffix == ".gguf":
            continue

        LOG.debug(f"Copying {filepath.name} to output")
        shutil.copy2(filepath, output_path)


def _merge_tensor_with_lora(
    tensor: torch.Tensor,
    key: str,
    lora_state: Dict[str, torch.Tensor],
    scale: float,
    lora_config_dict: Dict,
    device: str,
) -> torch.Tensor:
    """
    Helper function to merge a single tensor with its corresponding LoRA weights.

    Args:
        tensor: Base model tensor
        key: Tensor key/name
        lora_state: Dictionary containing LoRA weights
        scale: LoRA scaling factor (alpha/r)
        lora_config_dict: LoRA configuration dictionary
        device: Device to perform computations on

    Returns:
        Merged tensor with LoRA applied
    """
    lora_a, lora_b = find_lora_weights(lora_state, key)

    if lora_a is not None and lora_b is not None:
        LOG.debug(f"Merging LoRA for {key}: {lora_a.shape}, {lora_b.shape}")

        original_dtype = tensor.dtype
        base_fp32 = tensor.to(device).to(torch.float32)
        a_fp32 = lora_a.to(device).to(torch.float32)
        b_fp32 = lora_b.to(device).to(torch.float32)
        delta = scale * (b_fp32 @ a_fp32)
        if bool(
            lora_config_dict.get("fan_in_fan_out", False)
            or lora_config_dict.get("lora_fan_in_fan_out", False)
        ):
            delta = delta.T
        merged_tensor = (base_fp32 + delta).to(original_dtype).detach().cpu()
        del base_fp32, a_fp32, b_fp32, delta
        return merged_tensor, True
    else:
        return tensor.detach().cpu(), False


def merge_lora_sharded_efficient(
    base_model_path: Union[str, Path],
    lora_adapter_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cpu",
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
        base_model_path = Path(snapshot_download(str(base_model_path)))

    os.makedirs(output_path, exist_ok=True)

    config_file = lora_adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"LoRA config not found: {config_file}")

    lora_config_dict = LoraConfig.from_json_file(str(config_file))
    if not lora_config_dict.get("r") or lora_config_dict["r"] <= 0:
        raise ValueError("LoRA config 'r' must be > 0")

    unsupported_methods = []

    # Check for DoRA (Weight-Decomposed LoRA)
    if lora_config_dict.get("use_dora", False):
        unsupported_methods.append("DoRA (Weight-Decomposed LoRA)")

    # Check for AdaLoRA (Adaptive LoRA)
    if lora_config_dict.get("use_adalora", False):
        unsupported_methods.append("AdaLoRA (Adaptive LoRA)")

    # Check for VeRA (Vector-based Random Matrix Adaptation)
    if lora_config_dict.get("use_vera", False):
        unsupported_methods.append("VeRA (Vector-based Random Matrix Adaptation)")

    # Check for other advanced LoRA variants by task_type
    task_type = lora_config_dict.get("task_type", "")
    if task_type and task_type not in [
        "CAUSAL_LM",
        "SEQ_2_SEQ_LM",
        "TOKEN_CLS",
        "SEQ_CLS",
        "QUESTION_ANS",
    ]:
        unsupported_methods.append(f"Task type: {task_type}")

    # Check for rank adaptation patterns (AdaLoRA indicators)
    if any(
        key in lora_config_dict
        for key in ["rank_pattern", "alpha_pattern", "target_rank"]
    ):
        unsupported_methods.append("AdaLoRA (rank adaptation detected)")

    # Check for advanced initialization methods
    init_lora_weights = lora_config_dict.get("init_lora_weights", "")
    if init_lora_weights and init_lora_weights not in [
        "gaussian",
        "loftq",
        True,
        False,
    ]:
        unsupported_methods.append(f"Advanced initialization: {init_lora_weights}")

    if unsupported_methods:
        methods_str = ", ".join(unsupported_methods)
        raise NotImplementedError(
            f"Memory-efficient LoRA merge only supports standard LoRA. "
            f"Detected unsupported methods: {methods_str}. "
            f"Please use the legacy merge method for advanced LoRA variants."
        )

    scale = float(lora_config_dict["lora_alpha"]) / float(lora_config_dict["r"])

    LOG.debug(f"LoRA scale factor: {scale}")

    lora_file = lora_adapter_path / "adapter_model.safetensors"
    if not lora_file.exists():
        lora_file = lora_adapter_path / "adapter_model.bin"
        if not lora_file.exists():
            raise FileNotFoundError(
                f"LoRA adapter weights not found in {lora_adapter_path}"
            )

    LOG.debug(f"Loading LoRA weights from {lora_file}")

    if lora_file.suffix == ".safetensors":
        lora_state = safetensors.torch.load_file(lora_file)
    else:
        lora_state = torch.load(lora_file, map_location="cpu", weights_only=True)  # nosec B614
    LOG.debug("Keeping LoRA weights on CPU; will move per-tensor during merge")

    model_shards = get_model_shards(base_model_path)
    if not model_shards:
        raise FileNotFoundError(f"No model shards found in {base_model_path}")

    LOG.debug(f"Found {len(model_shards)} model shards in {base_model_path}")
    copy_non_model_files(base_model_path, output_path, model_shards)

    merged_count = 0
    total_tensors = 0

    for shard_path in tqdm(model_shards, desc="Merging shards"):
        merged_tensors = {}
        metadata = {}

        if shard_path.suffix == ".safetensors":
            with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
                if hasattr(f, "metadata") and f.metadata():
                    metadata = f.metadata()

                for key in f.keys():
                    total_tensors += 1
                    tensor = f.get_tensor(key)
                    merged_tensor, was_merged = _merge_tensor_with_lora(
                        tensor, key, lora_state, scale, lora_config_dict, device
                    )
                    merged_tensors[key] = merged_tensor
                    if was_merged:
                        merged_count += 1
        else:
            state_dict = torch.load(  # nosec B614: loading trusted model weights
                shard_path, map_location="cpu", weights_only=True
            )
            for key, tensor in state_dict.items():
                total_tensors += 1
                merged_tensor, was_merged = _merge_tensor_with_lora(
                    tensor, key, lora_state, scale, lora_config_dict, device
                )
                merged_tensors[key] = merged_tensor
                if was_merged:
                    merged_count += 1

        output_shard_path = output_path / shard_path.name
        merged_tensors = {k: v.detach().cpu() for k, v in merged_tensors.items()}

        if safe_tensors:
            if not str(output_shard_path).endswith(".safetensors"):
                output_shard_path = output_path / (shard_path.stem + ".safetensors")
            safetensors.torch.save_file(
                merged_tensors, output_shard_path, metadata=metadata
            )
        else:
            if shard_path.suffix == ".safetensors":
                safetensors.torch.save_file(
                    merged_tensors, output_shard_path, metadata=metadata
                )
            else:
                torch.save(merged_tensors, output_shard_path)

        del merged_tensors
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    LOG.info(f"Applied LoRA to {merged_count}/{total_tensors} tensors")
