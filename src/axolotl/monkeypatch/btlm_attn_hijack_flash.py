"""
Flash attention monkey patch for cerebras btlm model
"""

import importlib
import logging
from typing import Optional, Tuple

import torch
from accelerate import init_empty_weights
from flash_attn.flash_attn_interface import flash_attn_func
from transformers import AutoConfig, AutoModelForCausalLM

LOG = logging.getLogger("axolotl")


def replace_btlm_attn_with_flash_attn(model_name="cerebras/btlm-3b-8k-base"):
    # this is a wonky hack to get the remotely loaded module
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    # we need to load the model here in order for modeling_btlm to be available
    with init_empty_weights():
        AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    module_name = model_config.__class__.__module__.replace(
        ".configuration_btlm", ".modeling_btlm"
    )
    modeling_btlm = importlib.import_module(module_name)
    modeling_btlm.BTLMAttention._attn = (  # pylint: disable=protected-access
        flashattn_attn
    )


def flashattn_attn(
    self,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    value: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
    head_mask: Optional[torch.Tensor] = None,
    position_bias: Optional[torch.Tensor] = None,  # pylint: disable=unused-argument
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    softmax_scale = (
        1 / (key.size(-1) ** self.attn_scale_power) if self.scale_attn_weights else None
    )

    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # Perform Flash attention
    attn_output = flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,  # Assuming you have this attribute
        softmax_scale=softmax_scale,  # Set this if you have specific scaling in mind
        causal=not self.is_cross_attention,  # Assuming you have this attribute
        return_attn_probs=False,  # Set this based on your needs
    )

    # Optional: Apply head mask if it's not None
    if head_mask is not None:
        attn_output *= head_mask

    attn_output = attn_output.permute(0, 2, 1, 3)

    return attn_output, None  # We don't have explicit attn_weights in Flash attention
