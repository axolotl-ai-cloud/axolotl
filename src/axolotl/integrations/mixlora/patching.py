# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MixLoRA model patching: replaces FFN layers with MixLoRA MoE blocks.

Finds all SwiGLU-style FFNs (gate_proj, up_proj, down_proj) in a model's
transformer layers and replaces them with MixLoraFFN modules that contain
the original frozen FFN + router + LoRA experts.
"""

import torch.nn as nn

from axolotl.integrations.mixlora.model import MixLoraFFN
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Default values for MixLoRA config
MIXLORA_DEFAULTS = {
    "mixlora_num_experts": 8,
    "mixlora_top_k": 2,
    "mixlora_router_aux_loss_coef": 0.01,
    "mixlora_router_init_range": 0.02,
    "mixlora_jitter_noise": 0.0,
}


def _is_swiglu_ffn(module: nn.Module) -> bool:
    """Check if a module is a SwiGLU-style FFN with gate_proj, up_proj, down_proj."""
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and isinstance(module.gate_proj, nn.Linear)
        and isinstance(module.up_proj, nn.Linear)
        and isinstance(module.down_proj, nn.Linear)
    )


def _find_ffn_modules(model: nn.Module) -> list[tuple[nn.Module, str, nn.Module]]:
    """Find all FFN modules in the model that can be replaced with MixLoRA.

    Looks for modules with gate_proj/up_proj/down_proj attributes (SwiGLU pattern),
    typically named 'mlp' in transformer layers.

    Returns:
        List of (parent_module, attribute_name, ffn_module) tuples.
    """
    ffn_modules = []

    for name, module in model.named_modules():
        if _is_swiglu_ffn(module):
            # Find the parent module and the attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name

            ffn_modules.append((parent, attr_name, module))

    return ffn_modules


def patch_model_with_mixlora(model: nn.Module, cfg: DictDefault) -> nn.Module:
    """Replace FFN layers with MixLoRA MoE blocks.

    Iterates over all transformer layers, finds FFN modules with the SwiGLU
    pattern (gate_proj, up_proj, down_proj), and replaces them with MixLoraFFN
    modules that wrap the original frozen FFN with a router and LoRA experts.

    Args:
        model: The model to patch (may be a PeftModel wrapping a base model).
        cfg: Axolotl configuration with MixLoRA settings.

    Returns:
        The patched model (modified in-place).
    """
    # Resolve config with defaults (use `is not None` to avoid masking falsy values like 0)
    num_experts = getattr(cfg, "mixlora_num_experts", None)
    if num_experts is None:
        num_experts = MIXLORA_DEFAULTS["mixlora_num_experts"]
    top_k = getattr(cfg, "mixlora_top_k", None)
    if top_k is None:
        top_k = MIXLORA_DEFAULTS["mixlora_top_k"]
    router_init_range = getattr(cfg, "mixlora_router_init_range", None)
    if router_init_range is None:
        router_init_range = MIXLORA_DEFAULTS["mixlora_router_init_range"]
    jitter_noise = getattr(cfg, "mixlora_jitter_noise", None)
    if jitter_noise is None:
        jitter_noise = MIXLORA_DEFAULTS["mixlora_jitter_noise"]

    # Expert LoRA config (falls back to main LoRA config)
    lora_r = getattr(cfg, "mixlora_expert_lora_r", None)
    if lora_r is None:
        lora_r = cfg.lora_r
    lora_alpha = getattr(cfg, "mixlora_expert_lora_alpha", None)
    if lora_alpha is None:
        lora_alpha = cfg.lora_alpha
    lora_dropout = getattr(cfg, "mixlora_expert_lora_dropout", None)
    if lora_dropout is None:
        lora_dropout = cfg.lora_dropout if cfg.lora_dropout is not None else 0.0

    # Find all FFN modules
    ffn_modules = _find_ffn_modules(model)

    if not ffn_modules:
        LOG.warning(
            "MixLoRA: No SwiGLU FFN modules found (gate_proj, up_proj, down_proj). "
            "MixLoRA patching had no effect."
        )
        return model

    LOG.info(
        f"MixLoRA: Patching {len(ffn_modules)} FFN layers with "
        f"{num_experts} experts (top-{top_k}), "
        f"expert LoRA r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}"
    )

    for parent, attr_name, original_ffn in ffn_modules:
        mixlora_ffn = MixLoraFFN(
            original_ffn=original_ffn,
            num_experts=num_experts,
            top_k=top_k,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            router_init_range=router_init_range,
            jitter_noise=jitter_noise,
        )

        # Move to the same device/dtype as the original
        device = next(original_ffn.parameters()).device
        dtype = next(original_ffn.parameters()).dtype
        # Only move trainable params (router + experts) to the device
        # The original FFN is already on the correct device
        mixlora_ffn.router = mixlora_ffn.router.to(device=device, dtype=dtype)
        mixlora_ffn.experts = mixlora_ffn.experts.to(device=device, dtype=dtype)

        setattr(parent, attr_name, mixlora_ffn)

    LOG.info("MixLoRA: Patching complete")
    return model
