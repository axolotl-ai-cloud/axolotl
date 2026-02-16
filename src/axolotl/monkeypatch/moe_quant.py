"""
Post-load quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, many MoE models store expert weights as fused 3D nn.Parameter
tensors instead of individual nn.Linear modules. BnB 4-bit quantization only targets
nn.Linear, so these expert weights are skipped during model loading, causing OOM.

This module provides a post-load fixup that quantizes those skipped parameters using
bitsandbytes.nn.parametrize.replace_parameter_4bit (requires bitsandbytes >= 0.48.0).
PEFT's target_parameters / ParamWrapper can then apply LoRA on top of these quantized
params via stacked parametrizations.
"""

import bitsandbytes as bnb
import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def find_unquantized_expert_params(model):
    """Find 3D+ nn.Parameter tensors that BnB quantization skipped.

    Returns:
        List of (module, param_name) tuples to quantize.
    """
    params_to_quantize = []
    for _, module in model.named_modules():
        if isinstance(module, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            continue
        for param_name, param in module.named_parameters(recurse=False):
            if param.ndim >= 3 and any(
                kw in param_name for kw in ("experts", "gate_up_proj", "down_proj")
            ):
                params_to_quantize.append((module, param_name))
    return params_to_quantize


def quantize_moe_expert_params(model, quant_type=None, compress_statistics=None):
    """Quantize 3D nn.Parameter expert weights that BnB skips during model loading.

    Reads quant_type and compress_statistics from the model's quantization_config
    when not explicitly provided, so that the same settings used for nn.Linear
    quantization are applied to the MoE expert parameters.
    """
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    params_to_quantize = find_unquantized_expert_params(model)
    if not params_to_quantize:
        return False

    # Derive settings from model's BnB config if not explicitly provided
    if quant_type is None or compress_statistics is None:
        bnb_config = getattr(model.config, "quantization_config", None)
        if bnb_config is not None:
            if quant_type is None:
                quant_type = getattr(bnb_config, "bnb_4bit_quant_type", "nf4")
            if compress_statistics is None:
                compress_statistics = getattr(
                    bnb_config, "bnb_4bit_use_double_quant", True
                )
    # Final defaults
    if quant_type is None:
        quant_type = "nf4"
    if compress_statistics is None:
        compress_statistics = True

    count = 0
    for module, param_name in params_to_quantize:
        replace_parameter_4bit(
            module,
            param_name,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
        )
        count += 1

    torch.cuda.empty_cache()
    LOG.info(
        "Quantized %d MoE expert parameters to 4-bit (quant_type=%s, compress_statistics=%s)",
        count,
        quant_type,
        compress_statistics,
    )
    return True
