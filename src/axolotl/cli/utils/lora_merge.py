import gc
import math
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


def _simulate_nf4_roundtrip(
    tensor: torch.Tensor,
    blocksize: Optional[int] = None,
    compress_statistics: bool = True,
) -> torch.Tensor:
    """
    Simulate NF4 quantization roundtrip to match QLoRA training dynamics.

    During QLoRA training, base weights are quantized to NF4 and dequantized on-the-fly
    for each forward pass. The LoRA adapters learn to compensate for the quantization
    noise in the dequantized weights. To match this at merge time, we apply the same
    quantize → dequantize roundtrip so the merged result reflects what the model saw
    during training.

    Args:
        tensor: Base model weight tensor (fp16/bf16/fp32)
        blocksize: NF4 quantization block size (default: bitsandbytes default)
        compress_statistics: Whether to use double quantization

    Returns:
        Tensor after NF4 quantize → dequantize roundtrip, in original dtype
    """
    import bitsandbytes.functional as bnb_F

    original_dtype = tensor.dtype
    original_shape = tensor.shape

    # bitsandbytes requires float32 input for quantization and contiguous+CUDA tensor
    flat = tensor.reshape(-1).to(torch.float32).contiguous().cuda()

    quant_kwargs = {
        "quant_type": "nf4",
        "compress_statistics": compress_statistics,
    }
    if blocksize is not None:
        quant_kwargs["blocksize"] = blocksize

    quantized, quant_state = bnb_F.quantize_4bit(flat, **quant_kwargs)
    dequantized = bnb_F.dequantize_4bit(quantized, quant_state, quant_type="nf4")

    return dequantized.reshape(original_shape).to(original_dtype).cpu()


def find_lora_weights(
    lora_state: Dict[str, torch.Tensor],
    key: str,
    weight_renamings: Optional[Dict[str, str]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Find corresponding LoRA A and B weights for a given key.

    Also tries keys after applying weight renamings (from transformers v5
    conversion mappings) in case the checkpoint key names differ from the
    runtime model key names used by the LoRA adapter.
    """
    import re

    clean_key = key[:-7] if key.endswith(".weight") else key

    # Try the direct key first
    a_key = f"base_model.model.{clean_key}.lora_A.weight"
    b_key = f"base_model.model.{clean_key}.lora_B.weight"

    lora_a = lora_state.get(a_key)
    lora_b = lora_state.get(b_key)

    if lora_a is not None and lora_b is not None:
        return lora_a, lora_b

    # Try renamed keys (checkpoint format → runtime format)
    if weight_renamings:
        for src_pattern, tgt_pattern in weight_renamings.items():
            renamed_key = re.sub(src_pattern, tgt_pattern, clean_key)
            if renamed_key != clean_key:
                a_key = f"base_model.model.{renamed_key}.lora_A.weight"
                b_key = f"base_model.model.{renamed_key}.lora_B.weight"
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


def _find_dora_magnitude(
    lora_state: Dict[str, torch.Tensor],
    key: str,
    weight_renamings: Optional[Dict[str, str]] = None,
) -> Optional[torch.Tensor]:
    """
    Find DoRA magnitude vector for a given key.
    """
    import re

    clean_key = key[:-7] if key.endswith(".weight") else key
    mag_key = f"base_model.model.{clean_key}.lora_magnitude_vector"
    result = lora_state.get(mag_key)
    if result is not None:
        return result

    if weight_renamings:
        for src_pattern, tgt_pattern in weight_renamings.items():
            renamed_key = re.sub(src_pattern, tgt_pattern, clean_key)
            if renamed_key != clean_key:
                mag_key = f"base_model.model.{renamed_key}.lora_magnitude_vector"
                result = lora_state.get(mag_key)
                if result is not None:
                    return result

    return None


def _should_nf4_roundtrip(
    key: str,
    tensor: torch.Tensor,
    simulate_nf4: bool,
    simulate_nf4_experts: bool,
) -> bool:
    """Determine if a tensor should undergo NF4 quantization roundtrip."""
    if tensor.ndim < 2:
        return False
    if simulate_nf4:
        return True
    if simulate_nf4_experts and tensor.ndim >= 3 and "expert" in key.lower():
        return True
    return False


def _merge_tensor_with_lora(
    tensor: torch.Tensor,
    key: str,
    lora_state: Dict[str, torch.Tensor],
    scale: float,
    lora_config_dict: Dict,
    device: str,
    simulate_nf4: bool = False,
    simulate_nf4_experts: bool = False,
    nf4_blocksize: Optional[int] = None,
    nf4_double_quant: bool = True,
    use_dora: bool = False,
    weight_renamings: Optional[Dict[str, str]] = None,
) -> tuple[torch.Tensor, bool]:
    """
    Helper function to merge a single tensor with its corresponding LoRA weights.

    Args:
        tensor: Base model tensor
        key: Tensor key/name
        lora_state: Dictionary containing LoRA weights
        scale: LoRA scaling factor (alpha/r)
        lora_config_dict: LoRA configuration dictionary
        device: Device to perform computations on
        simulate_nf4: Whether to simulate NF4 quantization roundtrip for all weights
        simulate_nf4_experts: Whether to simulate NF4 roundtrip for MoE expert tensors only
        nf4_blocksize: Block size for NF4 quantization
        nf4_double_quant: Whether to use double quantization
        use_dora: Whether to apply DoRA (Weight-Decomposed LoRA) merging
        weight_renamings: Optional key renamings from transformers conversion mapping

    Returns:
        Tuple of (merged tensor, whether LoRA was applied)
    """
    lora_a, lora_b = find_lora_weights(lora_state, key, weight_renamings)

    do_nf4 = _should_nf4_roundtrip(key, tensor, simulate_nf4, simulate_nf4_experts)

    if lora_a is not None and lora_b is not None:
        LOG.debug(f"Merging LoRA for {key}: {lora_a.shape}, {lora_b.shape}")

        original_dtype = tensor.dtype

        # Simulate NF4 quantization roundtrip to match QLoRA training dynamics
        if do_nf4:
            tensor = _simulate_nf4_roundtrip(
                tensor,
                blocksize=nf4_blocksize,
                compress_statistics=nf4_double_quant,
            )

        base_fp32 = tensor.to(device).to(torch.float32)
        a_fp32 = lora_a.to(device).to(torch.float32)
        b_fp32 = lora_b.to(device).to(torch.float32)
        delta = scale * (b_fp32 @ a_fp32)
        if bool(
            lora_config_dict.get("fan_in_fan_out", False)
            or lora_config_dict.get("lora_fan_in_fan_out", False)
        ):
            delta = delta.T

        if use_dora:
            # DoRA: Weight = m * (W + delta) / ||W + delta||_col
            # where m is the learned magnitude vector
            magnitude = _find_dora_magnitude(lora_state, key, weight_renamings)
            if magnitude is not None:
                merged_weight = base_fp32 + delta
                # Column-wise norm (norm over input dimension = dim 1 for [out, in])
                weight_norm = merged_weight.norm(p=2, dim=1, keepdim=True)
                weight_norm = weight_norm.clamp(min=1e-8)
                mag = magnitude.to(device).to(torch.float32).unsqueeze(1)
                merged_tensor = (
                    (mag * merged_weight / weight_norm)
                    .to(original_dtype)
                    .detach()
                    .cpu()
                )
                del base_fp32, a_fp32, b_fp32, delta, merged_weight, weight_norm, mag
                return merged_tensor, True
            else:
                LOG.warning(
                    f"DoRA enabled but magnitude vector not found for {key}, "
                    f"falling back to standard LoRA merge"
                )

        merged_tensor = (base_fp32 + delta).to(original_dtype).detach().cpu()
        del base_fp32, a_fp32, b_fp32, delta
        return merged_tensor, True
    else:
        if do_nf4:
            tensor = _simulate_nf4_roundtrip(
                tensor,
                blocksize=nf4_blocksize,
                compress_statistics=nf4_double_quant,
            )
        return tensor.detach().cpu(), False


def _get_conversion_info(base_model_path: Path) -> tuple[Dict[str, str], list]:
    """
    Load the model's config.json and check if transformers has WeightRenaming
    or WeightConverter mappings for this model type.

    Returns:
        - dict of {source_pattern: target_pattern} for simple renamings
        - list of WeightConverter objects for fuse/unfuse operations
    """
    import json as _json

    config_path = base_model_path / "config.json"
    if not config_path.exists():
        return {}, []

    try:
        with open(config_path) as f:
            model_config = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return {}, []

    model_type = model_config.get("model_type")
    if not model_type:
        return {}, []

    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming
    except ImportError:
        return {}, []

    conversions = get_checkpoint_conversion_mapping(model_type)
    if not conversions:
        return {}, []

    renamings = {}
    weight_converters = []
    for conv in conversions:
        if isinstance(conv, WeightRenaming):
            # WeightRenaming stores patterns as lists internally
            src_list = (
                conv.source_patterns
                if isinstance(conv.source_patterns, list)
                else [conv.source_patterns]
            )
            tgt_list = (
                conv.target_patterns
                if isinstance(conv.target_patterns, list)
                else [conv.target_patterns]
            )
            if len(src_list) == 1 and len(tgt_list) == 1:
                renamings[src_list[0]] = tgt_list[0]
        elif isinstance(conv, WeightConverter):
            weight_converters.append(conv)

    return renamings, weight_converters


def _fuse_and_unfuse_with_merge(
    shard_tensors: Dict[str, torch.Tensor],
    weight_converters: list,
    lora_state: Dict[str, torch.Tensor],
    scale: float,
    lora_config_dict: Dict,
    device: str,
    simulate_nf4: bool = False,
    simulate_nf4_experts: bool = False,
    nf4_blocksize: Optional[int] = None,
    nf4_double_quant: bool = True,
    use_dora: bool = False,
    weight_renamings: Optional[Dict[str, str]] = None,
) -> tuple[Dict[str, torch.Tensor], int, set]:
    """
    For tensors matching WeightConverter patterns (MoE expert weights):
    1. Fuse checkpoint-format tensors into runtime-format (e.g., per-expert → fused 3D)
    2. Apply NF4 roundtrip + LoRA merge on the fused tensor
    3. Unfuse back to checkpoint format for saving

    Returns:
        - Updated tensor dict
        - Count of merged LoRA targets
        - Set of keys that were processed (fused/merged/unfused) and should be
          skipped by the per-tensor merge pass to avoid double NF4 roundtrip
    """
    import re

    from transformers.core_model_loading import Concatenate, MergeModulelist

    result = dict(shard_tensors)  # Start with all tensors
    merged_count = 0
    processed_keys: set = set()  # Keys that were fuse/unfuse processed

    for converter in weight_converters:
        src_patterns = (
            converter.source_patterns
            if isinstance(converter.source_patterns, list)
            else [converter.source_patterns]
        )
        tgt_patterns = (
            converter.target_patterns
            if isinstance(converter.target_patterns, list)
            else [converter.target_patterns]
        )

        # Build regex for each source pattern
        pattern_regexes = []
        for pat in src_patterns:
            regex_str = re.escape(pat).replace(r"\.\*\.", r"\.(\d+)\.")
            regex_str = (
                regex_str.rstrip(r"\$") if regex_str.endswith(r"\$") else regex_str
            )
            pattern_regexes.append(re.compile(r"(.*\.)?" + regex_str + "$"))

        # Group matching keys by layer prefix and source pattern
        # {layer_prefix: {pat_idx: {expert_idx: (key, tensor)}}}
        layer_groups: Dict[str, Dict[int, Dict[int, tuple[str, torch.Tensor]]]] = {}

        for key in list(result.keys()):
            for pat_idx, pat_regex in enumerate(pattern_regexes):
                match = pat_regex.match(key)
                if match:
                    prefix = match.group(1) or ""
                    # Extract expert index from the matched portion
                    remaining = key[len(prefix) :]
                    expert_match = re.search(r"\.(\d+)\.", remaining)
                    expert_idx = int(expert_match.group(1)) if expert_match else 0

                    layer_groups.setdefault(prefix, {}).setdefault(pat_idx, {})[
                        expert_idx
                    ] = (key, result[key])
                    break

        # Process each layer group
        for prefix, pat_groups in layer_groups.items():
            # Check we have all source patterns for this layer
            if not pat_groups:
                continue

            # Step 1: Fuse — MergeModulelist (stack experts) per source pattern
            fused_per_pattern = {}
            original_keys_per_pattern: Dict[int, list[str]] = {}
            num_experts = None

            for pat_idx in sorted(pat_groups.keys()):
                expert_data = pat_groups[pat_idx]
                sorted_indices = sorted(expert_data.keys())
                if num_experts is None:
                    num_experts = len(sorted_indices)

                sorted_tensors = [expert_data[idx][1] for idx in sorted_indices]
                original_keys_per_pattern[pat_idx] = [
                    expert_data[idx][0] for idx in sorted_indices
                ]
                fused_per_pattern[src_patterns[pat_idx]] = torch.stack(
                    sorted_tensors, dim=0
                )

            # Apply remaining operations (Concatenate)
            fused_tensor = None
            has_concat = False
            concat_dim = 1  # default

            for op in converter.operations:
                if isinstance(op, MergeModulelist):
                    pass  # Already handled
                elif isinstance(op, Concatenate):
                    has_concat = True
                    concat_dim = op.dim
                    tensors_to_cat = [
                        fused_per_pattern[sp]
                        for sp in src_patterns
                        if sp in fused_per_pattern
                    ]
                    if len(tensors_to_cat) > 1:
                        fused_tensor = torch.cat(tensors_to_cat, dim=concat_dim)
                    elif tensors_to_cat:
                        fused_tensor = tensors_to_cat[0]

            if not has_concat and len(fused_per_pattern) == 1:
                fused_tensor = next(iter(fused_per_pattern.values()))

            if fused_tensor is None:
                continue

            # Step 2: Build the fused key name and merge LoRA
            fused_key = prefix + tgt_patterns[0]

            # Apply NF4 roundtrip on the fused tensor (matching training dynamics)
            do_nf4 = _should_nf4_roundtrip(
                fused_key, fused_tensor, simulate_nf4, simulate_nf4_experts
            )
            if do_nf4:
                fused_tensor = _simulate_nf4_roundtrip(
                    fused_tensor,
                    blocksize=nf4_blocksize,
                    compress_statistics=nf4_double_quant,
                )

            # Try to find and merge LoRA weights for the fused key
            lora_a, lora_b = find_lora_weights(lora_state, fused_key, weight_renamings)
            if lora_a is not None and lora_b is not None:
                LOG.debug(
                    f"Merging LoRA for fused key {fused_key}: {lora_a.shape}, {lora_b.shape}"
                )
                original_dtype = fused_tensor.dtype
                base_fp32 = fused_tensor.to(device).to(torch.float32)
                a_fp32 = lora_a.to(device).to(torch.float32)
                b_fp32 = lora_b.to(device).to(torch.float32)
                delta = scale * (b_fp32 @ a_fp32)

                if use_dora:
                    magnitude = _find_dora_magnitude(
                        lora_state, fused_key, weight_renamings
                    )
                    if magnitude is not None:
                        merged_weight = base_fp32 + delta
                        weight_norm = merged_weight.norm(
                            p=2, dim=1, keepdim=True
                        ).clamp(min=1e-8)
                        mag = magnitude.to(device).to(torch.float32).unsqueeze(1)
                        fused_tensor = (
                            (mag * merged_weight / weight_norm)
                            .to(original_dtype)
                            .detach()
                            .cpu()
                        )
                        del (
                            base_fp32,
                            a_fp32,
                            b_fp32,
                            delta,
                            merged_weight,
                            weight_norm,
                            mag,
                        )
                    else:
                        fused_tensor = (
                            (base_fp32 + delta).to(original_dtype).detach().cpu()
                        )
                        del base_fp32, a_fp32, b_fp32, delta
                else:
                    fused_tensor = (base_fp32 + delta).to(original_dtype).detach().cpu()
                    del base_fp32, a_fp32, b_fp32, delta
                merged_count += 1

            # Step 3: Unfuse back to checkpoint format for saving
            # Reverse of fuse: Chunk (reverse of Concatenate) then unstack (reverse of MergeModulelist)
            if has_concat and len(src_patterns) > 1:
                # Split along the concat dim back into per-pattern tensors
                chunks = torch.chunk(fused_tensor, len(src_patterns), dim=concat_dim)
                unstacked_per_pattern = dict(zip(src_patterns, chunks, strict=True))
            else:
                unstacked_per_pattern = {src_patterns[0]: fused_tensor}

            # Unstack (reverse of MergeModulelist): split 3D → list of 2D per-expert
            for pat_idx, src_pat in enumerate(src_patterns):
                if src_pat not in unstacked_per_pattern:
                    continue
                stacked = unstacked_per_pattern[src_pat]
                if pat_idx in original_keys_per_pattern:
                    orig_keys = original_keys_per_pattern[pat_idx]
                    # Remove old keys from result
                    for ok in orig_keys:
                        result.pop(ok, None)
                    # Split and save with original key names
                    expert_tensors = torch.unbind(stacked, dim=0)
                    for orig_key, expert_tensor in zip(
                        orig_keys, expert_tensors, strict=True
                    ):
                        result[orig_key] = expert_tensor.detach().cpu()
                        processed_keys.add(orig_key)

    return result, merged_count, processed_keys


def merge_lora_sharded_efficient(
    base_model_path: Union[str, Path],
    lora_adapter_path: Union[str, Path],
    output_path: Union[str, Path],
    device: str = "cpu",
    safe_tensors: bool = True,
    simulate_nf4: bool = False,
    simulate_nf4_experts: bool = False,
    nf4_blocksize: Optional[int] = None,
    nf4_double_quant: bool = True,
) -> None:
    """
    Memory-efficient LoRA merging that processes shards individually
    without loading the full model into memory.

    Args:
        simulate_nf4: Apply NF4 roundtrip to ALL weight tensors (for QLoRA)
        simulate_nf4_experts: Apply NF4 roundtrip only to MoE expert tensors
            (for quantize_moe_experts). Expert tensors are identified by having
            "expert" in the key name and ndim >= 3.
    """
    base_model_path = Path(base_model_path)
    lora_adapter_path = Path(lora_adapter_path)
    output_path = Path(output_path)

    if "/" in str(base_model_path) and not base_model_path.exists():
        base_model_path = Path(snapshot_download(str(base_model_path)))

    # Check for weight conversion requirements (transformers v5)
    weight_renamings, weight_converters = _get_conversion_info(base_model_path)
    if weight_renamings:
        LOG.debug(f"Found {len(weight_renamings)} weight renamings for this model type")
    if weight_converters:
        LOG.debug(
            f"Found {len(weight_converters)} weight converters (fuse/unfuse) for this model type. "
            f"Will fuse→merge→unfuse within each shard."
        )

    os.makedirs(output_path, exist_ok=True)

    config_file = lora_adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"LoRA config not found: {config_file}")

    lora_config_dict = LoraConfig.from_json_file(str(config_file))
    if not lora_config_dict.get("r") or lora_config_dict["r"] <= 0:
        raise ValueError("LoRA config 'r' must be > 0")

    use_dora = bool(lora_config_dict.get("use_dora", False))

    unsupported_methods = []

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

    use_rslora = bool(lora_config_dict.get("use_rslora", False))
    if use_rslora:
        scale = float(lora_config_dict["lora_alpha"]) / math.sqrt(
            float(lora_config_dict["r"])
        )
    else:
        scale = float(lora_config_dict["lora_alpha"]) / float(lora_config_dict["r"])

    LOG.debug(f"LoRA scale factor: {scale} (rslora={use_rslora})")

    if simulate_nf4:
        LOG.info(
            "NF4 simulation enabled: base weights will undergo quantize→dequantize "
            "roundtrip before LoRA merge to match QLoRA training dynamics"
        )

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

        # Load all tensors from the shard
        if shard_path.suffix == ".safetensors":
            with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
                if hasattr(f, "metadata") and f.metadata():
                    metadata = f.metadata()
                shard_tensors = {key: f.get_tensor(key) for key in f.keys()}
        else:
            shard_tensors = torch.load(  # nosec B614: loading trusted model weights
                shard_path, map_location="cpu", weights_only=True
            )

        total_tensors += len(shard_tensors)

        # Step 1: Handle fused weight conversions (MoE experts) if applicable
        fused_keys: set = set()
        if weight_converters:
            shard_tensors, fused_merged, fused_keys = _fuse_and_unfuse_with_merge(
                shard_tensors,
                weight_converters,
                lora_state,
                scale,
                lora_config_dict,
                device,
                simulate_nf4=simulate_nf4,
                simulate_nf4_experts=simulate_nf4_experts,
                nf4_blocksize=nf4_blocksize,
                nf4_double_quant=nf4_double_quant,
                use_dora=use_dora,
                weight_renamings=weight_renamings,
            )
            merged_count += fused_merged

        # Step 2: Merge remaining (non-fused) tensors with LoRA
        # Skip keys already processed by fuse/unfuse to avoid double NF4 roundtrip
        for key, tensor in shard_tensors.items():
            if key in fused_keys:
                merged_tensors[key] = tensor.detach().cpu()
                continue
            merged_tensor, was_merged = _merge_tensor_with_lora(
                tensor,
                key,
                lora_state,
                scale,
                lora_config_dict,
                device,
                simulate_nf4=simulate_nf4,
                simulate_nf4_experts=simulate_nf4_experts,
                nf4_blocksize=nf4_blocksize,
                nf4_double_quant=nf4_double_quant,
                use_dora=use_dora,
                weight_renamings=weight_renamings,
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

        del merged_tensors, shard_tensors
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if merged_count == 0:
        LOG.warning(
            "No LoRA weights were matched to base model tensors. "
            "This may indicate a key name mismatch between the checkpoint format "
            "and the LoRA adapter. Consider using merge_lora_method: legacy."
        )
    LOG.info(f"Applied LoRA to {merged_count}/{total_tensors} tensors")
