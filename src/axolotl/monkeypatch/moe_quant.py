"""
Post-load quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, many MoE models store expert weights as fused 3D nn.Parameter
tensors instead of individual nn.Linear modules. BnB 4-bit quantization only targets
nn.Linear, so these expert weights are skipped during model loading, causing OOM.

This module provides a post-load fixup that quantizes those skipped parameters using
bitsandbytes.nn.parametrize.replace_parameter_4bit (requires bitsandbytes >= 0.48.0).
"""

import torch

from axolotl.common.architectures import MOE_EXPERT_PARAMS
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def quantize_moe_expert_params(
    model, model_config_type, quant_type="nf4", compress_statistics=True
):
    """Quantize 3D nn.Parameter expert weights that BnB skips during model loading."""
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    target_params = MOE_EXPERT_PARAMS.get(model_config_type)
    if not target_params:
        return

    count = 0
    for module_name, module in model.named_modules():
        for param_name in target_params:
            if hasattr(module, param_name):
                param = getattr(module, param_name)
                if isinstance(param, torch.nn.Parameter) and param.ndim >= 2:
                    replace_parameter_4bit(
                        module,
                        param_name,
                        compress_statistics=compress_statistics,
                        quant_type=quant_type,
                    )
                    count += 1

    LOG.info("Quantized %d MoE expert parameters to 4-bit", count)
