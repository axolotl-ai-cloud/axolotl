import gc
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

import safetensors
import safetensors.torch
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig
from peft.utils.other import get_pattern_key
from tqdm import tqdm

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _resolve_lora_alpha_for_key(
    weight_key: str,
    lora_config_dict: Dict,
    weight_renamings: Optional[Dict[str, str]] = None,
) -> Optional[int]:
    # Mirror PEFT's get_pattern_key matching so merge uses the same per-module alpha as training.
    alpha_pattern = lora_config_dict.get("alpha_pattern") or {}
    if not alpha_pattern:
        return None
    module_path = (
        weight_key.rsplit(".weight", 1)[0]
        if weight_key.endswith(".weight")
        else weight_key
    )
    pattern_keys = list(alpha_pattern.keys())
    matched_key = get_pattern_key(pattern_keys, module_path)
    if matched_key in alpha_pattern:
        return alpha_pattern[matched_key]
    # Fall back to renamed path so alpha lookup follows the same key resolution as find_lora_weights.
    if weight_renamings:
        import re

        for src_pattern, tgt_pattern in weight_renamings.items():
            renamed = re.sub(src_pattern, tgt_pattern, module_path)
            if renamed != module_path:
                matched_key = get_pattern_key(pattern_keys, renamed)
                if matched_key in alpha_pattern:
                    return alpha_pattern[matched_key]
    return None


def _build_layer_type_map(
    base_model_path: Path, trust_remote_code: bool = False
) -> dict[str, str]:
    """Build a map of module_name -> layer_type using a meta-device model.

    Instantiates the model architecture on the meta device (zero memory)
    to inspect which modules are Linear vs Conv1d/Conv2d/Conv3d.
    This avoids relying on weight tensor ndim heuristics.
    """
    import json as _json

    import torch.nn as nn
    from transformers import AutoConfig

    config_path = base_model_path / "config.json"
    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            model_config = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return {}

    architectures = model_config.get("architectures", [])
    if not architectures:
        return {}

    try:
        config = AutoConfig.from_pretrained(
            str(base_model_path), trust_remote_code=trust_remote_code
        )
    except Exception:
        LOG.debug("Could not load config for layer type introspection")
        return {}

    # Determine the right Auto class from architectures
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
    )

    auto_classes = [AutoModelForCausalLM, AutoModel]
    try:
        from transformers import AutoModelForImageTextToText

        auto_classes.insert(0, AutoModelForImageTextToText)
    except ImportError:
        pass

    model = None
    for auto_cls in auto_classes:
        try:
            with torch.device("meta"):
                model = auto_cls.from_config(
                    config, trust_remote_code=trust_remote_code
                )
            break
        except Exception:  # noqa: BLE001
            LOG.debug(
                "Could not instantiate meta model with %s, trying next",
                auto_cls.__name__,
            )

    if model is None:
        LOG.debug("Could not instantiate meta model for layer type introspection")
        return {}

    layer_types = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            layer_types[name] = "Conv3d"
        elif isinstance(module, nn.Conv2d):
            layer_types[name] = "Conv2d"
        elif isinstance(module, nn.Conv1d):
            layer_types[name] = "Conv1d"
        elif isinstance(module, nn.Linear):
            layer_types[name] = "Linear"

    del model
    LOG.debug(
        f"Layer type map: {len(layer_types)} modules "
        f"({sum(1 for v in layer_types.values() if 'Conv' in v)} conv layers)"
    )
    return layer_types


def _simulate_nf4_roundtrip(
    tensor: torch.Tensor,
    blocksize: Optional[int] = None,
    compress_statistics: bool = True,
    device: Optional[Union[str, torch.device]] = None,
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
        device: Device for quantization computation.  bitsandbytes requires a
            CUDA device; defaults to "cuda" when available.

    Returns:
        Tensor after NF4 quantize → dequantize roundtrip, in original dtype
    """
    import bitsandbytes.functional as bnb_F

    quant_device: torch.device
    if device is None:
        quant_device = torch.device("cuda")
    elif isinstance(device, str):
        quant_device = torch.device(device)
    else:
        quant_device = device

    if quant_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "NF4 simulation requires CUDA but no GPU is available. "
            "Either run on a machine with a GPU or disable NF4 simulation."
        )

    original_dtype = tensor.dtype
    original_shape = tensor.shape

    # bitsandbytes requires float32 input for quantization and contiguous+CUDA tensor
    flat = tensor.reshape(-1).to(torch.float32).contiguous().to(quant_device)

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


def _find_param_wrapper_lora(
    lora_state: Dict[str, torch.Tensor],
    key: str,
    tensor_shape: Optional[tuple] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str]]:
    """
    Find LoRA weights from a ParamWrapper (lora_target_parameters) that targets
    a parent module containing this weight as a sub-parameter.

    For example, base weight key 'model.layers.0.mlp.experts.down_proj' may have
    LoRA at 'base_model.model.model.layers.0.mlp.experts.lora_A.weight' (targeting
    the 'experts' module with 'down_proj' as the parameter_name).

    When tensor_shape is provided, validates that the LoRA dimensions match the
    target tensor (important when multiple ParamWrappers are nested and each
    nesting level has different LoRA dimensions).

    Returns (lora_A, lora_B, parameter_name) or (None, None, None).
    """
    clean_key = key[:-7] if key.endswith(".weight") else key
    # Strip trailing parameter name to get the parent module path
    # e.g., "model.layers.0.mlp.experts.down_proj" → parent="model.layers.0.mlp.experts", param="down_proj"
    parts = clean_key.rsplit(".", 1)
    if len(parts) != 2:
        return None, None, None

    parent_key, param_name = parts

    # PEFT's ParamWrapper nesting: when multiple parameters are targeted on
    # the same module, it nests wrappers. The outer wrapper's LoRA is at
    # parent.lora_A/B and inner wrappers use parent.base_layer.lora_A/B,
    # parent.base_layer.base_layer.lora_A/B, etc.
    prefixes_to_try = [
        f"base_model.model.{parent_key}",
    ]
    # Walk up .base_layer nesting levels (typically 1-2 deep)
    for depth in range(1, 4):
        bl = ".base_layer" * depth
        prefixes_to_try.append(f"base_model.model.{parent_key}{bl}")

    # Both 3D orientations exist: gpt-oss-style [E, in, out] pairs with
    # (A_in, B_out) = (shape[1], shape[2]); Qwen3-style [E, out, in] with
    # (A_in, B_out) = (shape[2], shape[1]). Exhaust every nesting level in the
    # exact orientation before falling back to the transposed one, so a
    # transposed outer LoRA cannot shadow an exact inner match.
    orientations: tuple = (None,)
    if tensor_shape is not None and len(tensor_shape) >= 3:
        orientations = (
            (tensor_shape[1], tensor_shape[2]),
            (tensor_shape[2], tensor_shape[1]),
        )

    for orientation in orientations:
        for prefix in prefixes_to_try:
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            lora_a = lora_state.get(a_key)
            lora_b = lora_state.get(b_key)
            if lora_a is None or lora_b is None:
                continue

            # When tensor_shape is given, verify dimensions match before returning.
            # This prevents returning a mismatched LoRA from a different nesting level.
            if orientation is not None and tensor_shape is not None:
                num_experts = tensor_shape[0]
                if not (
                    lora_a.shape[0] == lora_b.shape[1]
                    and lora_a.shape[0] % num_experts == 0
                    and (lora_a.shape[1], lora_b.shape[0]) == orientation
                ):
                    continue  # Dimensions don't match, try next nesting level

            return lora_a, lora_b, param_name

    return None, None, None


def _build_peft_layer_and_get_delta(
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_config_dict: Dict,
    base_tensor: torch.Tensor,
    adapter_name: str = "default",
    is_param_wrapper: bool = False,
    magnitude: Optional[torch.Tensor] = None,
    layer_type: Optional[str] = None,
    lora_alpha_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Use PEFT's own layer classes to compute the LoRA delta weight.

    Instead of re-implementing the merge math for every LoRA variant, this
    constructs a lightweight PEFT layer, loads the A/B weights, and calls
    ``get_delta_weight`` (or ``merge`` for DoRA) which handles standard LoRA,
    RSLoRA, DoRA, and ParamWrapper (expert-blocked) LoRA.

    Returns the delta tensor (same shape as base_tensor).
    """
    import warnings

    import torch.nn as nn

    r_total = lora_a.shape[0]
    in_features = lora_a.shape[1]
    out_features = lora_b.shape[0]
    # Per-module override from alpha_pattern wins over the global alpha so merge matches training scale.
    if lora_alpha_override is not None:
        lora_alpha = lora_alpha_override
    else:
        lora_alpha = lora_config_dict.get("lora_alpha", lora_config_dict.get("r", 1))
    use_rslora = bool(lora_config_dict.get("use_rslora", False))
    use_dora = bool(lora_config_dict.get("use_dora", False))

    if is_param_wrapper:
        from peft.tuners.lora.layer import ParamWrapper

        num_experts = base_tensor.shape[0]
        r = r_total // num_experts

        class _FakeModule(nn.Module):
            pass

        fake = _FakeModule()
        fake.register_parameter(
            "weight", nn.Parameter(base_tensor.clone(), requires_grad=False)
        )

        # ParamWrapper rejects dropout/fan_in_fan_out/lora_bias/use_dora, so
        # build a minimal config with only the fields it accepts.
        pw_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            fan_in_fan_out=False,
            use_rslora=use_rslora,
            use_dora=False,
            lora_bias=False,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            layer = ParamWrapper(
                fake,
                adapter_name=adapter_name,
                parameter_name="weight",
                config=pw_config,
                r=r,
                lora_alpha=lora_alpha,
            )
        layer.lora_A[adapter_name].weight.data = lora_a
        layer.lora_B[adapter_name].weight.data = lora_b
        delta = layer.get_delta_weight(adapter_name)
        # peft >=0.19.1 may return delta with transposed dims for 3D params
        if delta.shape != base_tensor.shape and delta.ndim == 3:
            delta = delta.transpose(1, 2).contiguous()
        return delta
    elif (
        layer_type and "Conv" in layer_type or (layer_type is None and lora_a.ndim > 2)
    ):
        # Conv layer detected via model introspection (or ndim fallback)

        from peft.tuners.lora import layer as peft_lora_layer

        # Determine conv type from layer_type map or fall back to ndim
        if layer_type and "Conv" in layer_type:
            conv_type: str = layer_type
        else:
            ndim = lora_a.ndim
            _conv_map = {3: "Conv1d", 4: "Conv2d", 5: "Conv3d"}
            if ndim not in _conv_map:
                raise ValueError(
                    f"Unsupported LoRA weight dimensionality {ndim} for conv layer"
                )
            conv_type = _conv_map[ndim]
            LOG.warning(
                f"Using ndim-based fallback for conv detection (ndim={ndim}). "
                f"Consider providing layer_type from meta-device introspection."
            )

        conv_cls_map = {"Conv1d": nn.Conv1d, "Conv2d": nn.Conv2d, "Conv3d": nn.Conv3d}
        ConvCls = conv_cls_map[conv_type]
        PeftConvCls = getattr(peft_lora_layer, conv_type)

        # Reconstruct conv parameters from base tensor and lora_a shapes
        # base_tensor: [out_channels, in_channels/groups, *kernel_size]
        # lora_a:      [r, in_channels/groups, *kernel_size]
        # lora_b:      [out_channels, r, *ones]
        out_channels = base_tensor.shape[0]
        in_channels = base_tensor.shape[1]
        kernel_size = tuple(base_tensor.shape[2:])
        stride = (1,) * (base_tensor.ndim - 2)
        padding = (0,) * (base_tensor.ndim - 2)

        base_layer = ConvCls(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        base_layer.weight.data = base_tensor.clone()

        conv_config = LoraConfig(
            r=r_total,
            lora_alpha=lora_alpha,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        layer = PeftConvCls(
            base_layer,
            adapter_name=adapter_name,
            config=conv_config,
            r=r_total,
            lora_alpha=lora_alpha,
        )
        layer.lora_A[adapter_name].weight.data = lora_a
        layer.lora_B[adapter_name].weight.data = lora_b

        if use_dora:
            if magnitude is None:
                raise ValueError(
                    f"DoRA merge requires a magnitude vector but none was found "
                    f"for conv layer (adapter={adapter_name}). Check that the "
                    f"adapter checkpoint contains lora_magnitude_vector weights."
                )
            mag_layer = layer.lora_magnitude_vector[adapter_name]
            mag_layer.weight = nn.Parameter(magnitude)
            layer.merge(adapter_names=[adapter_name])
            return base_layer.weight.data - base_tensor

        return layer.get_delta_weight(adapter_name)
    else:
        from peft.tuners.lora.layer import Linear as LoraLinear

        base_layer = nn.Linear(in_features, out_features, bias=False)
        base_layer.weight.data = base_tensor.clone()

        fan_in_fan_out = bool(
            lora_config_dict.get("fan_in_fan_out", False)
            or lora_config_dict.get("lora_fan_in_fan_out", False)
        )

        linear_config = LoraConfig(
            r=r_total,
            lora_alpha=lora_alpha,
            fan_in_fan_out=fan_in_fan_out,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )
        layer = LoraLinear(
            base_layer,
            adapter_name=adapter_name,
            config=linear_config,
            r=r_total,
            lora_alpha=lora_alpha,
        )
        layer.lora_A[adapter_name].weight.data = lora_a
        layer.lora_B[adapter_name].weight.data = lora_b

        if use_dora:
            # DoRA merges magnitude normalization into the weight directly.
            # Use PEFT's merge() which handles DoRA internally, then
            # compute the delta as merged_weight - original_weight.
            if magnitude is None:
                raise ValueError(
                    f"DoRA merge requires a magnitude vector but none was found "
                    f"for linear layer (adapter={adapter_name}). Check that the "
                    f"adapter checkpoint contains lora_magnitude_vector weights."
                )
            mag_layer = layer.lora_magnitude_vector[adapter_name]
            mag_layer.weight = nn.Parameter(magnitude)
            layer.merge(adapter_names=[adapter_name])
            return base_layer.weight.data - base_tensor

        return layer.get_delta_weight(adapter_name)


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
        # Skip weight-map index files — they reference shard filenames that may
        # change during the merge (e.g. .bin → .safetensors).  A correct index
        # is regenerated after all shards have been written.
        if filepath.name.endswith(".index.json"):
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


def _dequant_block_fp8(w: torch.Tensor, si: torch.Tensor, dev: str) -> torch.Tensor:
    """FineGrainedFP8 (block-fp8): ``W`` e4m3 * ``scale_inv`` fp32 (128x128 blocks). 2D or fused-3D
    (block axes = last two dims)."""
    wf = w.to(dev).float()
    s = si.to(dev).float()
    *lead, N, K = w.shape
    sr, sc = s.shape[-2], s.shape[-1]
    # FineGrainedFP8 dims are always block-aligned; a ragged grid would need a fixed 128-block
    # pad/slice (not a floor-spread), so refuse rather than silently mis-scale.
    if N % sr or K % sc:
        raise ValueError(
            f"block-fp8 weight {tuple(w.shape)} is not aligned to its scale grid "
            f"({sr}x{sc}); FineGrainedFP8 requires block-aligned dims"
        )
    bn, bk = N // sr, K // sc
    return (wf.reshape(*lead, sr, bn, sc, bk) * s.reshape(*lead, sr, 1, sc, 1)).reshape(
        *lead, N, K
    )


# Standard OCP-MX / NVFP4 FP4 E2M1 codebook, indexed by raw 4-bit nibble (sign|2-exp|1-mantissa).
_FP4_E2M1_LUT = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


_MX_BLOCK = 32  # OCP-MX block width (one e8m0 scale per 32 elements)


def _mx_block_scale(sb_bytes: torch.Tensor, N: int, K: int, dev: str) -> torch.Tensor:
    """e8m0 biased-exponent bytes [.., N, nb] -> per-element multiplier [.., N, K]. Each scale covers
    a fixed 32-wide MX block; a ragged final block (K not a multiple of 32) is trimmed."""
    s = torch.exp2(sb_bytes.to(dev).float() - 127.0)
    if s.shape[-1] * _MX_BLOCK == K:  # perfectly aligned -> memory-efficient expand
        return s[..., None].expand(*s.shape, _MX_BLOCK).reshape(*s.shape[:-1], K)
    return s.repeat_interleave(_MX_BLOCK, dim=-1)[..., :K]


def _dequant_mxfp8(w: torch.Tensor, s: torch.Tensor, dev: str) -> torch.Tensor:
    """OCP-MX fp8: ``W`` e4m3 * ``2^(e8m0_byte-127)`` (32-wide blocks along the last dim). ``s`` is the
    e8m0 scale (``float8_e8m0fnu`` or raw ``uint8``). Ragged-K safe."""
    wf = w.to(dev).float()
    *lead, N, K = w.shape
    scale = _mx_block_scale(s.view(torch.uint8), N, K, dev).reshape(*lead, N, K)
    return wf * scale


def _unpack_fp4(packed: torch.Tensor, dev: str) -> torch.Tensor:
    """Unpack a packed-e2m1 uint8 tensor [.., K/2] -> fp32 values [.., K] via the OCP LUT. Low nibble =
    element 2i, high nibble = element 2i+1 (torchao / OCP convention)."""
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=dev)
    b = packed.to(dev)
    lo = (b & 0xF).long()
    hi = ((b >> 4) & 0xF).long()
    nib = torch.stack([lo, hi], dim=-1).reshape(*b.shape[:-1], b.shape[-1] * 2)
    return lut[nib]


def _dequant_mxfp4(w: torch.Tensor, s: torch.Tensor, dev: str) -> torch.Tensor:
    """OCP-MX fp4: packed e2m1 ``W`` [.., K/2] uint8 * ``2^(e8m0-127)`` (32-wide blocks). Ragged-K safe.
    Matches the codebook + low/high nibble order the ScatterMoE MX forward uses, so the merged weight
    equals what the model computes."""
    vals = _unpack_fp4(w, dev)
    *lead, N, K = vals.shape
    scale = _mx_block_scale(s.view(torch.uint8), N, K, dev).reshape(*lead, N, K)
    return vals * scale


def _dequant_nvfp4(w, scale, scale2, dev: str) -> torch.Tensor:
    """NVFP4: packed e2m1 ``W`` [..,K/2] uint8 + e4m3 block-16 ``scale`` [..,K/16] + optional per-tensor
    ``scale_2`` (two-level; single-level when absent -> per_tensor_scale=1). Uses torchao's own
    dequantize (matches the loader). Auto-detects swizzled scales by shape and passes the flag. Raises
    if torchao is unavailable (caller degrades)."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    wt = w.to(dev)
    st = scale.to(dev)
    p = scale2.to(dev).float().reshape(()) if scale2 is not None else None
    *lead, N, K = wt.shape
    # a padded swizzled scale has more elements than the plain block-16 grid; torchao must unswizzle it
    expect = 1
    for d in (*lead, N, (K * 2) // 16):
        expect *= d
    swizzled = st.numel() != expect
    nv = NVFP4Tensor(
        wt, st, 16, torch.bfloat16, per_tensor_scale=p, is_swizzled_scales=swizzled
    )
    return nv.dequantize(torch.bfloat16)


def _detect_quant_format(key, w, shard_tensors, e8m0):
    """Return (fmt, {suffix: scale_tensor}) for a quantized weight ``key``, else (None, {}).
    fmt in {block_fp8, mxfp8, nvfp4, mxfp4}."""
    si = shard_tensors.get(key + "_scale_inv")
    sc = shard_tensors.get(key + "_scale")
    sc2 = shard_tensors.get(key + "_scale_2")
    is_e8m0 = sc is not None and (
        sc.dtype == torch.uint8 or (e8m0 is not None and sc.dtype == e8m0)
    )
    if w.dtype == torch.float8_e4m3fn and si is not None:
        return "block_fp8", {"_scale_inv": si}
    if w.dtype == torch.float8_e4m3fn and is_e8m0:
        return "mxfp8", {"_scale": sc}
    if w.dtype == torch.uint8 and sc is not None and sc.dtype == torch.float8_e4m3fn:
        scales = {"_scale": sc}
        if sc2 is not None:
            scales["_scale_2"] = sc2
        return "nvfp4", scales
    if w.dtype == torch.uint8 and is_e8m0:
        return "mxfp4", {"_scale": sc}
    return None, {}


def _dequant_by_format(fmt, w, scales, dev):
    if fmt == "block_fp8":
        return _dequant_block_fp8(w, scales["_scale_inv"], dev)
    if fmt == "mxfp8":
        return _dequant_mxfp8(w, scales["_scale"], dev)
    if fmt == "nvfp4":
        return _dequant_nvfp4(w, scales["_scale"], scales.get("_scale_2"), dev)
    return _dequant_mxfp4(w, scales["_scale"], dev)  # mxfp4


def _requant_by_format(fmt, w_bf16, scales, dev, nvfp4_scale_mode="reuse"):
    """Re-quantize a merged bf16 weight back to its original format. Returns
    ``{"": qweight, "_scale*": scale_tensors}`` matching the original scale dtypes/shapes so the
    merged checkpoint loads exactly like the base did.

    nvfp4 has two scale modes (shared quantizer in sonicmoe ``nvfp4_quant``):

    - ``reuse`` (default, unprepared adapters): keep the base scales verbatim and only re-round
      the codes. Recomputing scales shifts the whole dequant grid, re-rounding EVERY element and
      burying a small LoRA delta under uncorrelated noise, while on the original grid only
      elements the delta pushes across a code boundary change. It also keeps gate/up outer scales
      equal, which the loader's fuse otherwise reconciles by folding ratios into block scales.
    - ``fresh`` (merge-aware adapters): recompute block scales from the merged weight with the
      SAME quantizer the training fake-quant used, so the written grid is bitwise the grid the
      adapter trained against; ``_scale_2`` is passed through (the expert writer supplies the
      fused-max pts training saw)."""
    w = w_bf16.to(dev).float()
    if fmt == "block_fp8":
        si = scales["_scale_inv"]
        *lead, N, K = w.shape
        sr, sc = si.shape[-2], si.shape[-1]
        bn, bk = N // sr, K // sc
        wb = w.reshape(*lead, sr, bn, sc, bk)
        amax = wb.abs().amax(dim=(-3, -1), keepdim=True).clamp_min(1e-12)
        scale_inv = amax / 448.0
        q = (wb / scale_inv).clamp_(-448, 448).to(torch.float8_e4m3fn)
        return {
            "": q.reshape(*lead, N, K).cpu(),
            "_scale_inv": scale_inv.reshape(*lead, sr, sc).to(si.dtype).cpu(),
        }
    if fmt == "mxfp8":
        q, ebyte = _quant_mx(w, 32)
        s = scales["_scale"]
        return {"": q.cpu(), "_scale": ebyte.view(s.dtype).cpu()}
    if fmt == "mxfp4":
        packed, ebyte = _quant_mxfp4(w)
        s = scales["_scale"]
        return {"": packed.cpu(), "_scale": ebyte.view(s.dtype).cpu()}
    # nvfp4: shared merge-identity quantizer (see docstring for the two scale modes)
    from axolotl.integrations.kernels.libs.sonicmoe.nvfp4_quant import (
        quantize_nvfp4_merge,
    )

    sc = scales["_scale"]
    sc2 = scales.get("_scale_2")
    pts = sc2.to(dev).float().reshape(()) if sc2 is not None else None
    if nvfp4_scale_mode == "fresh":
        packed, sc_out = quantize_nvfp4_merge(w, pts, scale_mode="fresh")
    else:
        packed, sc_out = quantize_nvfp4_merge(
            w,
            1.0 if pts is None else pts,
            scale_mode="reuse",
            base_block_scale=sc.to(dev),
        )
    out = {"": packed.cpu(), "_scale": sc_out.cpu()}
    if sc2 is not None:
        out["_scale_2"] = sc2.cpu()
    return out


def _quant_mx(w_f32, block):
    """bf16/f32 -> (e4m3 qdata, uint8 e8m0 exponent byte), FLOOR e8m0, 32-wide blocks along last dim."""
    *lead, N, K = w_f32.shape
    nb = K // block
    wb = w_f32.reshape(*lead, N, nb, block)
    amax = wb.abs().amax(-1).clamp_min(1e-12)
    exp = torch.floor(torch.log2(amax)) - 8.0
    q = (wb / torch.exp2(exp)[..., None]).clamp_(-448, 448).to(torch.float8_e4m3fn)
    ebyte = (exp + 127.0).clamp_(0, 254).to(torch.uint8)
    return q.reshape(*lead, N, K), ebyte


def _quant_mxfp4(w_f32):
    """f32 -> (packed e2m1 uint8 [.., K/2], uint8 e8m0 [.., K/32]); nearest-codebook, low/high nibble."""
    lut = torch.tensor(_FP4_E2M1_LUT, dtype=torch.float32, device=w_f32.device)
    *lead, N, K = w_f32.shape
    nb = K // _MX_BLOCK
    wb = w_f32.reshape(*lead, N, nb, _MX_BLOCK)
    amax = wb.abs().amax(-1).clamp_min(1e-6)
    exp = torch.floor(torch.log2(amax / 6.0))
    wn = (wb / torch.exp2(exp)[..., None]).reshape(*lead, N, K)
    idx = (wn.unsqueeze(-1) - lut).abs().argmin(-1).to(torch.uint8)  # [.., N, K]
    packed = idx[..., 0::2] | (idx[..., 1::2] << 4)
    ebyte = (exp + 127.0).clamp_(0, 254).to(torch.uint8)
    return packed, ebyte


def _resolve_nvfp4_scale_mode(lora_config_dict, override_quantizer: bool = False):
    """Read the merge-aware quantizer-identity metadata from adapter_config.json.

    Returns the nvfp4 requant scale mode: ``reuse`` for unprepared adapters,
    ``fresh`` for merge-aware ones. A merge-aware adapter trained against one
    quantizer and merged with another silently voids the retention guarantee,
    so any identity mismatch (missing torchao, different encoder version,
    unknown scale mode / pts policy) is a hard error; ``override_quantizer``
    downgrades the encoder-version check to a warning.
    """
    meta = lora_config_dict.get("nvfp4_merge_aware")
    if meta is None or meta is False:
        return "reuse"
    if meta is True:
        meta = {}
    if not isinstance(meta, dict):
        raise ValueError(
            f"adapter_config.json nvfp4_merge_aware must be a dict, got {meta!r}"
        )
    try:
        import torchao
    except ImportError as ex:
        raise RuntimeError(
            "this adapter was trained merge-aware (nvfp4_merge_aware) and its merge "
            "requires torchao (the training quantizer); pip install torchao"
        ) from ex
    scale_mode = meta.get("scale_mode", "fresh")
    if scale_mode != "fresh":
        raise ValueError(
            f"unsupported nvfp4_merge_aware scale_mode {scale_mode!r} (expected 'fresh')"
        )
    pts_policy = meta.get("pts_policy", "base_fused_max")
    if pts_policy != "base_fused_max":
        raise ValueError(
            f"unsupported nvfp4_merge_aware pts_policy {pts_policy!r} "
            "(expected 'base_fused_max')"
        )
    recorded = meta.get("encoder")
    current = f"torchao-{torchao.__version__}"
    if recorded and recorded != current:
        msg = (
            f"merge-aware adapter was trained with encoder {recorded!r} but this "
            f"environment has {current!r}; the written grid may not be the grid the "
            "adapter trained against. Install the matching torchao version, or pass "
            "--override-quantizer to merge anyway."
        )
        if not override_quantizer:
            raise RuntimeError(msg)
        LOG.warning("%s (overridden)", msg)
    return scale_mode


def _key_has_lora(key, shape, lora_state, weight_renamings):
    a, b = find_lora_weights(lora_state, key, weight_renamings)
    if a is not None and b is not None:
        return True
    if len(shape) >= 3:
        pa, pb, _ = _find_param_wrapper_lora(lora_state, key, tuple(shape))
        return pa is not None and pb is not None
    return False


def _dequantize_quantized_shard(
    shard_tensors: Dict[str, torch.Tensor],
    device: str,
    lora_state: Optional[Dict[str, torch.Tensor]] = None,
    weight_renamings: Optional[Dict[str, str]] = None,
    dequant_all: bool = True,
) -> tuple[Dict[str, torch.Tensor], bool, bool, Dict]:
    """Dequantize quantized weights to bf16 so the LoRA delta folds into the true value.

    Returns ``(new_shard, any_dequantized, any_left_quantized, requant_plan)``.

    ``dequant_all=True`` dequantizes EVERY quantized weight to bf16 (the ``--dequant`` merge: output is
    bf16). Default (``dequant_all=False``) is FORMAT-PRESERVING: only LoRA-targeted quantized weights
    are dequantized (so the delta can fold correctly); ``requant_plan[key] = (fmt, scales)`` records how
    to re-quantize each back to its original format after the merge, and quantized weights with no LoRA
    are left untouched (same dtype + scales). ``any_left_quantized`` = a quantized weight was detected
    but could not be dequantized (unsupported format / torchao missing) -> keep the config.

    Formats (detected by scale sibling): block-fp8 (``_scale_inv`` fp32, 128x128), mxfp8 (``_scale``
    e8m0/32), nvfp4 (``_scale`` e4m3/16 [+ ``_scale_2``]), mxfp4 (``_scale`` e8m0/32). Covers 2D linears
    and fused-3D experts; native per-expert-unfused nvfp4/mxfp4 is not handled here."""
    dev = device if (device != "cpu" and torch.cuda.is_available()) else "cpu"
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    out: Dict[str, torch.Tensor] = dict(shard_tensors)
    drop: set = set()
    plan: Dict = {}
    did = False
    left = False
    lora_state = lora_state or {}
    for key, w in shard_tensors.items():
        if key in drop or key.endswith(("_scale", "_scale_inv", "_scale_2")):
            continue
        if w.ndim not in (2, 3):
            continue
        fmt, scales = _detect_quant_format(key, w, shard_tensors, e8m0)
        if fmt is None:
            # a low-bit weight with a scale sibling but an unrecognized format -> keep the config
            if w.dtype in (torch.float8_e4m3fn, torch.uint8) and (
                shard_tensors.get(key + "_scale_inv") is not None
                or shard_tensors.get(key + "_scale") is not None
            ):
                left = True
            continue
        # format-preserving: only touch quantized weights a LoRA actually targets
        if not dequant_all and not _key_has_lora(
            key, w.shape, lora_state, weight_renamings
        ):
            continue
        try:
            deq = _dequant_by_format(fmt, w, scales, dev)
        except Exception as ex:  # torchao missing / shape mismatch -> leave quantized
            LOG.warning("%s dequant skipped for %s: %s", fmt, key, ex)
            left = True
            continue
        out[key] = deq.to(torch.bfloat16).cpu()
        did = True
        drop.update(key + suf for suf in scales)
        if not dequant_all:
            plan[key] = (fmt, scales)  # re-quantize after the LoRA fold
    for sk in drop:
        out.pop(sk, None)
    return out, did, left, plan


_FUSED_EXPERT_LORA_RE = re.compile(r"\.experts\.(?:base_layer\.)?lora_[AB]\.weight$")
_PER_EXPERT_WEIGHT_RE = re.compile(
    r"\.experts\.\d+\.(?:gate_proj|up_proj|down_proj|w1|w2|w3|gate_up_proj)\.weight$"
)


def _detect_per_expert_unfused_mismatch(model_shards, lora_state) -> bool:
    """True if the adapter carries a FUSED expert LoRA (``experts.lora_A``, targeting a fused
    ``experts.gate_up_proj`` param) but the base stores experts PER-EXPERT and unfused
    (``experts.0.gate_proj.weight`` ...). In that case the fused adapter keys match no base tensor,
    so the expert LoRA would be *silently dropped* by the shard-by-shard merge. Peeks shard keys
    only (no tensor loads)."""
    if not any(_FUSED_EXPERT_LORA_RE.search(k) for k in lora_state):
        return False
    for shard in model_shards:
        try:
            if str(shard).endswith(".safetensors"):
                with safetensors.safe_open(shard, framework="pt") as f:
                    keys = list(f.keys())
            else:
                keys = list(
                    torch.load(shard, map_location="meta", weights_only=True).keys()  # nosec B614
                )
        except Exception:  # noqa: BLE001  # nosec B112 - unreadable shard: skip the peek, not fatal
            continue
        if any(_PER_EXPERT_WEIGHT_RE.search(k) for k in keys):
            return True
    return False


def _per_expert_weights_are_packed(model_shards) -> bool:
    """True if any per-expert unfused expert weight is packed uint8 (NVFP4/MXFP4 qdata) — the layout
    the expert-merge writer understands. Reads safetensors headers only."""
    for shard in model_shards:
        try:
            if str(shard).endswith(".safetensors"):
                with safetensors.safe_open(shard, framework="pt") as f:
                    for k in f.keys():
                        if _PER_EXPERT_WEIGHT_RE.search(k) and ".experts." in k:
                            if f.get_slice(k).get_dtype() == "U8":
                                return True
            else:
                tensors = torch.load(shard, map_location="meta", weights_only=True)  # nosec B614
                for k, t in tensors.items():
                    if _PER_EXPERT_WEIGHT_RE.search(k) and t.dtype == torch.uint8:
                        return True
        except Exception:  # noqa: BLE001  # nosec B112 - unreadable shard: skip the peek, not fatal
            continue
    return False


_EXPERT_TRIPLE_RE = re.compile(
    r"^(?P<prefix>.*\.experts)\.(?P<e>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)\."
    r"(?P<leaf>weight|weight_scale|weight_scale_2)$"
)
_EXPERT_LEAVES = ("weight", "weight_scale", "weight_scale_2")
# (fused runtime param name, per-expert checkpoint proj names, concatenated on the row axis in order)
_EXPERT_FUSED_GROUPS = (
    ("gate_up_proj", ("gate_proj", "up_proj")),
    ("gate_up_proj", ("w1", "w3")),
    ("down_proj", ("down_proj",)),
    ("down_proj", ("w2",)),
)


class _Nvfp4ExpertMergeWriter:
    """Folds a FUSED expert LoRA (PEFT ParamWrapper over ``experts.gate_up_proj``/``down_proj``)
    into a base that stores experts PER-EXPERT unfused as modelopt NVFP4
    (``experts.<i>.<proj>.{weight,weight_scale,weight_scale_2}``), where no base key matches the
    adapter and the shard merge would otherwise drop the expert LoRA.

    ``consume`` claims the per-expert quantized tensors of LoRA-targeted layers out of each shard
    (buffering across shard boundaries, since a layer's expert list can be split). When a fused
    group is complete it dequantizes each expert (torchao, the same path training saw), fuses
    to the runtime 3D layout (stack experts, concat gate-then-up rows), folds the ParamWrapper
    delta, then unfuses and re-quantizes each expert back to NVFP4, so the merged checkpoint
    keeps the base's exact per-expert layout. Under ``dequant=True`` it emits the merged FUSED
    bf16 param instead (matching the bf16 fuse pass convention).

    ``scale_mode`` selects the requant grid (see ``_requant_by_format``): ``reuse`` keeps each
    expert's base grid; ``fresh`` (merge-aware adapters) rebuilds the FUSED grid the way the
    loader's ``fuse_nvfp4_experts`` did (one per-expert pts = max across projections, per-proj
    ratios folded into block scales), merges on it, and re-quantizes with the shared training
    quantizer, emitting the fused-max pts as every projection's ``weight_scale_2``, so the next
    load fuses exactly and the written grid is bitwise the grid training snapped to.
    """

    def __init__(
        self,
        lora_state: Dict[str, torch.Tensor],
        lora_config_dict: Dict,
        expected_num_experts: int,
        device: str,
        dequant: bool = False,
        scale_mode: str = "reuse",
    ):
        self.lora_state = lora_state
        self.lora_config_dict = lora_config_dict
        self.num_experts = expected_num_experts
        self.dequant = dequant
        self.scale_mode = scale_mode
        self._dev = device if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        # prefix -> proj -> expert_idx -> {leaf: tensor}
        self.pending: Dict[str, Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = {}
        self._prefix_lora: Dict[str, bool] = {}
        self.merged_groups = 0
        self._delta_mag_sum = 0.0
        self._base_mag_sum = 0.0

    def _prefix_has_fused_lora(self, prefix: str) -> bool:
        has = self._prefix_lora.get(prefix)
        if has is None:
            has = any(
                f"base_model.model.{prefix}{'.base_layer' * d}.lora_A.weight"
                in self.lora_state
                for d in range(4)
            )
            self._prefix_lora[prefix] = has
        return has

    @torch.no_grad()
    def consume(
        self, shard_tensors: Dict[str, torch.Tensor]
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
        """Claim this shard's per-expert NVFP4 tensors for LoRA-targeted expert modules and emit
        the merged tensors of every layer group that is now complete. Returns
        ``(remaining_shard_tensors, emitted_tensors, merged_lora_count)``."""
        remaining = dict(shard_tensors)
        for key in list(remaining):
            m = _EXPERT_TRIPLE_RE.match(key)
            if m is None or not self._prefix_has_fused_lora(m["prefix"]):
                continue
            # only the packed-uint8 layout is understood; a float weight (bf16 base) flows through
            if m["leaf"] == "weight" and remaining[key].dtype != torch.uint8:
                continue
            self.pending.setdefault(m["prefix"], {}).setdefault(
                m["proj"], {}
            ).setdefault(int(m["e"]), {})[m["leaf"]] = remaining.pop(key)

        emitted: Dict[str, torch.Tensor] = {}
        merged = 0
        for prefix in list(self.pending):
            for fused_name, members in _EXPERT_FUSED_GROUPS:
                if not self._group_complete(prefix, members):
                    continue
                out, n = self._process_group(prefix, fused_name, members)
                emitted.update(out)
                merged += n
                for mp in members:
                    del self.pending[prefix][mp]
            if not self.pending[prefix]:
                del self.pending[prefix]
        self.merged_groups += merged
        return remaining, emitted, merged

    def _group_complete(self, prefix: str, members: tuple) -> bool:
        projs = self.pending[prefix]
        if not any(mp in projs for mp in members):
            return False
        expected = set(range(self.num_experts))
        return all(
            mp in projs
            and set(projs[mp]) == expected
            and all(
                all(leaf in projs[mp][e] for leaf in _EXPERT_LEAVES) for e in expected
            )
            for mp in members
        )

    def _process_group(
        self, prefix: str, fused_name: str, members: tuple
    ) -> tuple[Dict[str, torch.Tensor], int]:
        projs = self.pending[prefix]
        E = self.num_experts
        # fused shape from the packed qdata headers (rows N, packed K/2 -> K), no dequant needed
        row_counts = [projs[mp][0]["weight"].shape[0] for mp in members]
        k_dim = projs[members[0]][0]["weight"].shape[1] * 2
        fused_key = f"{prefix}.{fused_name}"
        lora_a, lora_b, _ = _find_param_wrapper_lora(
            self.lora_state, fused_key, tensor_shape=(E, sum(row_counts), k_dim)
        )
        emitted: Dict[str, torch.Tensor] = {}
        if lora_a is None or lora_b is None:
            LOG.warning(
                "expert-merge writer: no shape-matching fused LoRA for %s; "
                "passing its per-expert tensors through unchanged",
                fused_key,
            )
            for mp in members:
                for e in range(E):
                    for leaf, t in projs[mp][e].items():
                        emitted[f"{prefix}.{e}.{mp}.{leaf}"] = t
            return emitted, 0

        dev = self._dev
        fresh = self.scale_mode == "fresh"
        pts_fused = None
        if fresh:
            # mirror fuse_nvfp4_experts: one per-expert pts (max across projs), each
            # proj's ratio folded into its block scales -> dequant here == the fused
            # weight training fake-quantized against
            pts_all = [
                torch.stack(
                    [
                        projs[mp][e]["weight_scale_2"].to(dev).float().reshape(())
                        for e in range(E)
                    ]
                ).view(-1, 1, 1)
                for mp in members
            ]
            pts_fused = pts_all[0]
            for pts_i in pts_all[1:]:
                pts_fused = torch.maximum(pts_fused, pts_i)
            from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

            per_proj = []
            for i, mp in enumerate(members):
                qd = torch.stack([projs[mp][e]["weight"].to(dev) for e in range(E)])
                sc = torch.stack(
                    [projs[mp][e]["weight_scale"].to(dev) for e in range(E)]
                )
                if not torch.allclose(pts_all[i], pts_fused):
                    sc = (sc.float() * (pts_all[i] / pts_fused)).to(torch.float8_e4m3fn)
                nv = NVFP4Tensor(qd, sc, 16, torch.bfloat16, per_tensor_scale=pts_fused)
                per_proj.append(nv.dequantize(torch.bfloat16))
        else:
            per_proj = [
                torch.stack(
                    [
                        _dequant_nvfp4(
                            projs[mp][e]["weight"],
                            projs[mp][e]["weight_scale"],
                            projs[mp][e]["weight_scale_2"],
                            dev,
                        )
                        for e in range(E)
                    ],
                    dim=0,
                )
                for mp in members
            ]
        fused = per_proj[0] if len(per_proj) == 1 else torch.cat(per_proj, dim=1)
        del per_proj
        delta = _build_peft_layer_and_get_delta(
            lora_a.to(dev),
            lora_b.to(dev),
            self.lora_config_dict,
            fused,
            is_param_wrapper=True,
        )
        merged_t = (fused.to(torch.float32) + delta.to(torch.float32)).to(
            torch.bfloat16
        )
        if not fresh and not self.dequant:
            self._delta_mag_sum += float(delta.float().abs().mean())
            self._base_mag_sum += float(fused.float().abs().mean())
        del fused, delta

        if self.dequant:
            emitted[fused_key] = merged_t.detach().cpu()
            return emitted, 1

        start = 0
        for mp, rows in zip(members, row_counts, strict=True):
            part = merged_t[:, start : start + rows, :]
            start += rows
            for e in range(E):
                leaves = projs[mp][e]
                wkey = f"{prefix}.{e}.{mp}.weight"
                sc2 = leaves["weight_scale_2"]
                if pts_fused is not None:
                    # clone: gate/up share pts_fused storage, safetensors refuses aliases
                    sc2 = pts_fused[e].reshape(sc2.shape).to(sc2.dtype).cpu().clone()
                requant = _requant_by_format(
                    "nvfp4",
                    part[e],
                    {"_scale": leaves["weight_scale"], "_scale_2": sc2},
                    dev,
                    nvfp4_scale_mode=self.scale_mode,
                )
                emitted[wkey] = requant.pop("")
                for suf, t in requant.items():
                    emitted[wkey + suf] = t
        return emitted, 1

    def assert_drained(self) -> None:
        if self.pending:
            detail = {
                prefix: {mp: len(ed) for mp, ed in projs.items()}
                for prefix, projs in self.pending.items()
            }
            raise RuntimeError(
                f"expert-merge writer: expert groups never completed (experts seen per projection: "
                f"{detail}; expected {self.num_experts} per projection with weight/weight_scale/"
                f"weight_scale_2 each). The base checkpoint is missing per-expert tensors."
            )
        if self._base_mag_sum > 0:
            ratio = self._delta_mag_sum / self._base_mag_sum
            if ratio < 0.02:
                LOG.warning(
                    "NEAR-NO-OP expert merge: mean |LoRA delta| is %.2f%% of mean |base weight|, "
                    "far below the NVFP4 grid step (~25-50%% of the block max), so re-rounding "
                    "onto the base grid erases most of the adapter. Train with "
                    "nvfp4_merge_aware: true, merge with --dequant (bf16, fully preserves the "
                    "adapter), or serve base + adapter unmerged.",
                    100.0 * ratio,
                )


def _update_config_vocab_size(output_path: Path, vocab_size: int) -> None:
    """After carrying a resized ``embed_tokens`` into the merge, set ``vocab_size`` in the merged
    ``config.json`` (and any nested text config) so the checkpoint loads with the enlarged vocab."""
    import json as _json

    cfg_path = output_path / "config.json"
    if not cfg_path.exists():
        return
    cfg = _json.loads(cfg_path.read_text())
    changed = cfg.get("vocab_size") != vocab_size
    cfg["vocab_size"] = vocab_size
    for sub in ("text_config", "llm_config"):
        if isinstance(cfg.get(sub), dict):
            cfg[sub]["vocab_size"] = vocab_size
            changed = True
    if changed:
        cfg_path.write_text(_json.dumps(cfg, indent=2))
        LOG.info(
            "Set merged config.json vocab_size=%d (resized embeddings carried)",
            vocab_size,
        )


def _find_full_override(
    lora_state: Dict[str, torch.Tensor], key: str
) -> Optional[torch.Tensor]:
    """Return the adapter's FULL-weight override for a base ``key`` if present, else None.

    PEFT saves trainable non-LoRA modules — ``modules_to_save`` and resized ``embed_tokens`` /
    ``lm_head`` — as plain ``base_model.model.<key>`` tensors (no ``lora_A``/``lora_B``). These must
    REPLACE the base weight at merge, not be ignored (otherwise a resized vocab / trained head is
    silently dropped). Matches ``.weight`` overrides only; ignores any ``.lora_`` tensor."""
    if not key.endswith(".weight") or ".lora_" in key:
        return None
    for cand in (
        "base_model.model." + key,
        "base_model.model.model." + key,
        "base_model.model." + key.replace("modules_to_save.", ""),
    ):
        t = lora_state.get(cand)
        if t is not None and ".lora_" not in cand:
            return t
    # PEFT modules_to_save layout: ...<module>.modules_to_save.default.weight
    ms = (
        "base_model.model." + key[: -len(".weight")] + ".modules_to_save.default.weight"
    )
    return lora_state.get(ms)


def _strip_quantization_config(output_path: Path) -> None:
    """After a block-fp8 -> bf16 merge, remove ``quantization_config`` from the merged ``config.json``
    so the merged checkpoint loads as bf16 (not FineGrainedFP8) and set ``torch_dtype`` to bfloat16."""
    import json as _json

    cfg_path = output_path / "config.json"
    if not cfg_path.exists():
        return
    cfg = _json.loads(cfg_path.read_text())
    changed = cfg.pop("quantization_config", None) is not None
    if cfg.get("torch_dtype") not in (None, "bfloat16"):
        cfg["torch_dtype"] = "bfloat16"
        changed = True
    if changed:
        cfg_path.write_text(_json.dumps(cfg, indent=2))
        LOG.info(
            "Stripped quantization_config from merged config.json (block-fp8 -> bf16 merge)"
        )


_QUANT_DTYPES = {
    torch.float8_e4m3fn,
    getattr(torch, "float8_e5m2", torch.float8_e4m3fn),
    torch.uint8,
    torch.int8,
}
_WARNED_UNDEQUANT: set = set()


def _warn_if_quant_undequantized(key: str, tensor: torch.Tensor, do_nf4: bool) -> None:
    """Loudly warn (once/key) if a LoRA delta is about to be folded into a still-quantized weight
    that the shard dequant did NOT convert to a real dtype (an unhandled format — e.g. mxfp4, a
    per-tensor-fp8 with a scalar scale, or a per-expert-unfused nvfp4/mxfp4 layout). Folding into raw
    packed bytes silently corrupts the weight; NF4 is handled by the simulate_nf4 roundtrip and is
    exempt."""
    if do_nf4 or tensor.dtype not in _QUANT_DTYPES or key in _WARNED_UNDEQUANT:
        return
    _WARNED_UNDEQUANT.add(key)
    LOG.warning(
        "LoRA merge: '%s' is still %s (a quantized format the merge did not dequantize) yet a LoRA "
        "delta targets it — folding into raw quantized data is WRONG. This format is unsupported by "
        "the efficient merge (handled: bf16, nf4-sim, block-fp8, mxfp8, fused-nvfp4). For per-expert "
        "nvfp4/mxfp4 experts use the nvfp4 expert-merge writer; otherwise use merge_method: legacy.",
        key,
        tensor.dtype,
    )


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
    layer_type_map: Optional[Dict[str, str]] = None,
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
        _warn_if_quant_undequantized(key, tensor, do_nf4)

        original_dtype = tensor.dtype

        # Simulate NF4 quantization roundtrip to match QLoRA training dynamics
        if do_nf4:
            tensor = _simulate_nf4_roundtrip(
                tensor,
                blocksize=nf4_blocksize,
                compress_statistics=nf4_double_quant,
                device=device,
            )

        magnitude = (
            _find_dora_magnitude(lora_state, key, weight_renamings)
            if use_dora
            else None
        )

        # Look up layer type from meta-device model introspection
        _layer_type = None
        if layer_type_map:
            mod_path = key.rsplit(".weight", 1)[0] if key.endswith(".weight") else key
            _layer_type = layer_type_map.get(mod_path)
            # Try common prefix variations (e.g. with/without "model." prefix)
            if _layer_type is None:
                for prefix in [
                    "model.",
                    "model.language_model.",
                    "model.language_model.model.",
                ]:
                    _layer_type = layer_type_map.get(prefix + mod_path)
                    if _layer_type:
                        break

        delta = _build_peft_layer_and_get_delta(
            lora_a.to(device),
            lora_b.to(device),
            lora_config_dict,
            tensor.to(device),
            magnitude=magnitude.to(device) if magnitude is not None else None,
            layer_type=_layer_type,
            lora_alpha_override=_resolve_lora_alpha_for_key(
                key, lora_config_dict, weight_renamings
            ),
        )
        merged_tensor = (
            (tensor.to(device).to(torch.float32) + delta.to(torch.float32))
            .to(original_dtype)
            .detach()
            .cpu()
        )
        return merged_tensor, True
    else:
        # Try ParamWrapper LoRA (lora_target_parameters) — the LoRA targets a
        # parent module and this weight is a sub-parameter of that module.
        if tensor.ndim >= 3:
            pw_a, pw_b, param_name = _find_param_wrapper_lora(
                lora_state, key, tensor_shape=tuple(tensor.shape)
            )
            if pw_a is not None and pw_b is not None:
                LOG.debug(
                    f"Merging ParamWrapper LoRA for {key} "
                    f"(param={param_name}): {pw_a.shape}, {pw_b.shape}"
                )
                _warn_if_quant_undequantized(key, tensor, do_nf4)
                if do_nf4:
                    tensor = _simulate_nf4_roundtrip(
                        tensor,
                        blocksize=nf4_blocksize,
                        compress_statistics=nf4_double_quant,
                        device=device,
                    )
                original_dtype = tensor.dtype
                delta = _build_peft_layer_and_get_delta(
                    pw_a.to(device),
                    pw_b.to(device),
                    lora_config_dict,
                    tensor.to(device),
                    is_param_wrapper=True,
                    lora_alpha_override=_resolve_lora_alpha_for_key(
                        key, lora_config_dict, weight_renamings
                    ),
                )
                merged = (
                    (tensor.to(device).to(torch.float32) + delta.to(torch.float32))
                    .to(original_dtype)
                    .detach()
                    .cpu()
                )
                return merged, True

        if do_nf4:
            tensor = _simulate_nf4_roundtrip(
                tensor,
                blocksize=nf4_blocksize,
                compress_statistics=nf4_double_quant,
                device=device,
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


def _get_expected_num_experts(base_model_path: Path) -> Optional[int]:
    """Expert count from config.json, used to detect expert lists split across shards."""
    import json as _json

    config_path = base_model_path / "config.json"
    if not config_path.exists():
        return None
    try:
        cfg = _json.loads(config_path.read_text())
    except (OSError, _json.JSONDecodeError):
        return None
    for sub in (cfg, cfg.get("text_config"), cfg.get("llm_config")):
        if not isinstance(sub, dict):
            continue
        for key in (
            "num_experts",
            "num_local_experts",
            "n_routed_experts",
            "num_routed_experts",
        ):
            val = sub.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


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
    layer_type_map: Optional[Dict[str, str]] = None,
    expected_num_experts: Optional[int] = None,
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

        is_expert_list = any(
            isinstance(op, MergeModulelist) for op in converter.operations
        )

        # Process each layer group
        for prefix, pat_groups in layer_groups.items():
            # Check we have all source patterns for this layer
            if not pat_groups:
                continue

            # Shards are processed one at a time, so a layer's expert list can be
            # split across shard boundaries. Fusing a partial list either crashes
            # (gate/up count mismatch in torch.cat) or silently fuses a subset of
            # experts under the fused key. Fusing still-quantized tensors is also
            # wrong: it stacks raw qdata and orphans the per-expert scale siblings.
            # Skip fusion in both cases; the per-tensor pass carries the tensors
            # through unchanged.
            index_sets = [set(g.keys()) for g in pat_groups.values()]
            complete = (
                len(pat_groups) == len(pattern_regexes)
                and all(s == index_sets[0] for s in index_sets[1:])
                and index_sets[0] == set(range(len(index_sets[0])))
                and (
                    not is_expert_list
                    or expected_num_experts is None
                    or len(index_sets[0]) == expected_num_experts
                )
            )
            skip_reason = None
            if not complete:
                expected_str = (
                    f", expected {expected_num_experts}"
                    if is_expert_list and expected_num_experts is not None
                    else ""
                )
                skip_reason = (
                    "expert list incomplete in this shard (found "
                    f"{[len(pat_groups[p]) for p in sorted(pat_groups)]} experts "
                    f"per pattern{expected_str})"
                )
            elif any(
                t.dtype in _QUANT_DTYPES
                or k + "_scale" in result
                or k + "_scale_inv" in result
                for g in pat_groups.values()
                for (k, t) in g.values()
            ):
                skip_reason = "tensors are still quantized (raw qdata cannot be fused)"
            if skip_reason:
                LOG.info(
                    "Skipping fuse for '%s%s': %s; leaving per-expert tensors "
                    "unchanged",
                    prefix,
                    tgt_patterns[0],
                    skip_reason,
                )
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
                    device=device,
                )

            # Try to find and merge LoRA weights for the fused key
            lora_a, lora_b = find_lora_weights(lora_state, fused_key, weight_renamings)
            if lora_a is not None and lora_b is not None:
                LOG.debug(
                    f"Merging LoRA for fused key {fused_key}: {lora_a.shape}, {lora_b.shape}"
                )
                original_dtype = fused_tensor.dtype
                magnitude = (
                    _find_dora_magnitude(lora_state, fused_key, weight_renamings)
                    if use_dora
                    else None
                )
                # Look up layer type for the fused key
                _layer_type = None
                if layer_type_map:
                    mod_path = (
                        fused_key.rsplit(".weight", 1)[0]
                        if fused_key.endswith(".weight")
                        else fused_key
                    )
                    _layer_type = layer_type_map.get(mod_path)
                    if _layer_type is None:
                        for prefix in [
                            "model.",
                            "model.language_model.",
                            "model.language_model.model.",
                        ]:
                            _layer_type = layer_type_map.get(prefix + mod_path)
                            if _layer_type:
                                break

                delta = _build_peft_layer_and_get_delta(
                    lora_a.to(device),
                    lora_b.to(device),
                    lora_config_dict,
                    fused_tensor.to(device),
                    magnitude=magnitude.to(device) if magnitude is not None else None,
                    layer_type=_layer_type,
                    lora_alpha_override=_resolve_lora_alpha_for_key(
                        fused_key, lora_config_dict, weight_renamings
                    ),
                )
                fused_tensor = (
                    (
                        fused_tensor.to(device).to(torch.float32)
                        + delta.to(torch.float32)
                    )
                    .to(original_dtype)
                    .detach()
                    .cpu()
                )
                merged_count += 1

            # Step 3: Save in fused format (runtime format) so that the merged
            # model can be loaded directly without needing WeightConverter
            # fusion during from_pretrained (which can OOM for large MoE models).
            # Remove the original per-expert keys and save the fused tensor
            # under the runtime key name.
            for pat_idx in sorted(original_keys_per_pattern.keys()):
                for ok in original_keys_per_pattern[pat_idx]:
                    result.pop(ok, None)
                    processed_keys.add(ok)

            result[fused_key] = fused_tensor.detach().cpu()
            processed_keys.add(fused_key)

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
    trust_remote_code: bool = False,
    dequant: bool = False,
    override_quantizer: bool = False,
) -> None:
    """
    Memory-efficient LoRA merging that processes shards individually
    without loading the full model into memory.

    dequant: if True, dequantize every quantized weight and write a bf16 checkpoint (strips
        quantization_config). Default False = FORMAT-PRESERVING: LoRA-targeted quantized weights are
        dequantized, the delta folded, then re-quantized back to the SAME format (fp8 stays fp8,
        nvfp4 stays nvfp4), so a large quantized base does not double in size.
    override_quantizer: proceed despite a quantizer-identity mismatch on a merge-aware
        adapter (encoder version drift); see ``_resolve_nvfp4_scale_mode``.

    Args:
        simulate_nf4: Apply NF4 roundtrip to ALL weight tensors (for QLoRA)
        simulate_nf4_experts: Apply NF4 roundtrip only to MoE expert tensors
            (for quantize_moe_experts). Expert tensors are identified by having
            "expert" in the key name and ndim >= 3.
        trust_remote_code: Whether to trust remote code when loading model
            config for layer-type introspection. Defaults to False for safety.
    """
    base_model_path = Path(base_model_path)
    lora_adapter_path = Path(lora_adapter_path)
    output_path = Path(output_path)

    config_file = lora_adapter_path / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"LoRA config not found: {config_file}")

    lora_config_dict = LoraConfig.from_json_file(str(config_file))
    if not lora_config_dict.get("r") or lora_config_dict["r"] <= 0:
        raise ValueError("LoRA config 'r' must be > 0")

    if dequant and lora_config_dict.get("nvfp4_merge_aware"):
        raise ValueError(
            "--dequant on a merge-aware adapter: the bf16 dequant merge writes "
            "the raw un-snapped effective weight, which is NOT the function "
            "training optimized (it can score worse than the base model). Merge "
            "without --dequant; the format-preserving NVFP4 merge is lossless "
            "for merge-aware adapters."
        )

    nvfp4_scale_mode = _resolve_nvfp4_scale_mode(lora_config_dict, override_quantizer)

    if "/" in str(base_model_path) and not base_model_path.exists():
        base_model_path = Path(snapshot_download(str(base_model_path)))

    # Check for weight conversion requirements (transformers v5)
    weight_renamings, weight_converters = _get_conversion_info(base_model_path)
    if weight_renamings:
        LOG.debug(f"Found {len(weight_renamings)} weight renamings for this model type")
    expected_num_experts = None
    if weight_converters:
        LOG.debug(
            f"Found {len(weight_converters)} weight converters (fuse/unfuse) for this model type. "
            f"Will fuse→merge→unfuse within each shard."
        )
        expected_num_experts = _get_expected_num_experts(base_model_path)

    os.makedirs(output_path, exist_ok=True)

    if nvfp4_scale_mode == "fresh":
        LOG.info(
            "merge-aware adapter detected: expert weights re-quantize with fresh "
            "scales (bitwise the grid training fake-quantized against)"
        )

    use_dora = bool(lora_config_dict.get("use_dora", False))

    # Build layer type map via meta-device model introspection
    layer_type_map = _build_layer_type_map(
        base_model_path, trust_remote_code=trust_remote_code
    )
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

    # PEFT writes peft_type=ADALORA and target_r for AdaLoRA; rank_pattern/alpha_pattern alone are plain LoRA.
    if (
        lora_config_dict.get("peft_type") == "ADALORA"
        or lora_config_dict.get("target_r") is not None
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

    expert_writer = None
    if _detect_per_expert_unfused_mismatch(model_shards, lora_state):
        try:
            import torchao  # noqa: F401

            has_torchao = True
        except ImportError:
            has_torchao = False
        if not has_torchao or not _per_expert_weights_are_packed(model_shards):
            LOG.warning(
                "MERGE INCOMPLETE: the adapter has a FUSED expert LoRA (experts.gate_up_proj/down_proj) "
                "but this base stores experts PER-EXPERT and unfused (experts.<i>.gate_proj.weight ...) "
                "in a layout the nvfp4 expert-merge writer cannot handle (%s), so the EXPERT LoRA is "
                "being DROPPED (non-expert LoRA still merges). Use merge_method: legacy (loads the "
                "full model so PEFT fuses the experts itself).",
                "torchao is not installed"
                if not has_torchao
                else "expert weights are not packed uint8 NVFP4",
            )
        else:
            n_experts = expected_num_experts or _get_expected_num_experts(
                base_model_path
            )
            if n_experts is None:
                raise RuntimeError(
                    "The adapter has a fused expert LoRA over a per-expert unfused base, but "
                    "config.json exposes no expert count (num_experts/num_local_experts/"
                    "n_routed_experts), so the expert-merge writer cannot validate layer "
                    "completeness across shards. Add the expert count to config.json or use "
                    "merge_method: legacy."
                )
            expert_writer = _Nvfp4ExpertMergeWriter(
                lora_state,
                lora_config_dict,
                n_experts,
                device,
                dequant=dequant,
                scale_mode=nvfp4_scale_mode,
            )
            LOG.info(
                "Adapter has a FUSED expert LoRA over a PER-EXPERT unfused NVFP4 base: using "
                "the expert-merge writer (dequant -> fuse -> fold delta -> unfuse -> requant)."
            )

    merged_count = 0
    total_tensors = 0
    block_fp8_dequantized = False
    left_quantized = False
    resized_vocab = None
    # Track weight_map for index regeneration: {tensor_key: shard_filename}
    weight_map: Dict[str, str] = {}

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

        # Per-expert unfused NVFP4 experts with a fused adapter: the writer claims those tensors
        # (buffered across shard boundaries) and emits the merged per-expert keys itself, so the
        # dequant/fuse/per-tensor passes below never see them.
        if expert_writer is not None:
            shard_tensors, expert_emitted, expert_merged = expert_writer.consume(
                shard_tensors
            )
            merged_tensors.update(expert_emitted)
            merged_count += expert_merged

        # Step 0: dequantize quantized weights so the LoRA delta folds into the TRUE weight (a raw read
        # misses the block scale). Default (dequant=False) touches only LoRA-targeted weights and plans
        # to re-quantize them back to their original format after the fold; dequant=True dequants all.
        shard_tensors, _shard_deq, _shard_left, requant_plan = (
            _dequantize_quantized_shard(
                shard_tensors,
                device,
                lora_state=lora_state,
                weight_renamings=weight_renamings,
                dequant_all=dequant,
            )
        )
        block_fp8_dequantized = block_fp8_dequantized or _shard_deq
        left_quantized = left_quantized or _shard_left

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
                layer_type_map=layer_type_map,
                expected_num_experts=expected_num_experts,
            )
            merged_count += fused_merged

        # Step 2: Merge remaining (non-fused) tensors with LoRA
        # Skip keys already processed by fuse/unfuse to avoid double NF4 roundtrip
        for key, tensor in shard_tensors.items():
            if key in fused_keys:
                merged_tensors[key] = tensor.detach().cpu()
                continue
            # Full-weight override (modules_to_save / resized embed_tokens+lm_head): replace, don't
            # fold LoRA. Track a vocab resize so the merged config.json stays consistent.
            override = _find_full_override(lora_state, key)
            if override is not None:
                merged_tensors[key] = override.to(tensor.dtype).detach().cpu()
                merged_count += 1
                # a resized vocab shows up as a row-count change on embed_tokens AND/OR lm_head
                # (they may be untied); catch either so the merged config.json stays consistent.
                if (
                    key.endswith(("embed_tokens.weight", "lm_head.weight"))
                    and override.shape[0] != tensor.shape[0]
                ):
                    resized_vocab = int(override.shape[0])
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
                layer_type_map=layer_type_map,
            )
            merged_tensors[key] = merged_tensor
            if was_merged:
                merged_count += 1

        # Step 3 (format-preserving): re-quantize each merged weight back to its original format
        # (fresh block scales) so the output dtype matches the input (fp8 -> fp8, nvfp4 -> nvfp4).
        for key, (fmt, scales) in requant_plan.items():
            if key not in merged_tensors:
                continue
            try:
                requant = _requant_by_format(
                    fmt,
                    merged_tensors[key],
                    scales,
                    device,
                    nvfp4_scale_mode=nvfp4_scale_mode,
                )
            except (
                Exception
            ) as ex:  # torchao missing etc. -> keep the bf16 merge for this weight
                LOG.warning("%s re-quant skipped for %s: %s", fmt, key, ex)
                left_quantized = True
                continue
            merged_tensors[key] = requant.pop("")
            for suf, t in requant.items():
                merged_tensors[key + suf] = t

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

        for tensor_key in merged_tensors:
            weight_map[tensor_key] = output_shard_path.name

        del merged_tensors, shard_tensors
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if expert_writer is not None:
        expert_writer.assert_drained()
        if expert_writer.merged_groups == 0:
            LOG.warning(
                "MERGE INCOMPLETE: the expert-merge writer matched no fused expert LoRA to any "
                "per-expert layer group (unrecognized checkpoint naming or shape mismatch); the "
                "expert LoRA was NOT merged."
            )

    # Regenerate weight-map index if the model was sharded
    if len(model_shards) > 1 and weight_map:
        import json as _json

        index_name = (
            "model.safetensors.index.json"
            if safe_tensors
            else "pytorch_model.bin.index.json"
        )
        index = {
            "metadata": {"total_size": total_tensors},
            "weight_map": weight_map,
        }
        with open(output_path / index_name, "w") as f:
            _json.dump(index, f, indent=2)
        LOG.debug(f"Wrote weight-map index: {index_name}")

    # Only the --dequant merge emits a bf16 checkpoint; the format-preserving default keeps the
    # quantized dtypes (and their quantization_config) intact.
    if dequant and block_fp8_dequantized and not left_quantized:
        _strip_quantization_config(output_path)
    elif block_fp8_dequantized and left_quantized:
        LOG.warning(
            "Merged some quantized weights to bf16 but others were left quantized (unsupported "
            "format); keeping quantization_config so the checkpoint still loads its quantized tensors."
        )
    if resized_vocab is not None:
        _update_config_vocab_size(output_path, resized_vocab)

    if merged_count == 0:
        LOG.warning(
            "No LoRA weights were matched to base model tensors. "
            "This may indicate a key name mismatch between the checkpoint format "
            "and the LoRA adapter. Consider using merge_method: legacy."
        )
    LOG.info(f"Applied LoRA to {merged_count}/{total_tensors} tensors")
