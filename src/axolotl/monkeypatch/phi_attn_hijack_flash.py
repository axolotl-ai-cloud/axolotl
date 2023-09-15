"""
Flash attention monkey patch for phi mixformers model
"""

import importlib
import logging

from flash_attn.flash_attn_interface import (
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
)
from transformers import AutoConfig

LOG = logging.getLogger("axolotl")


def replace_phi_attn_with_flash_attn(model_name: str):
    # this is a wonky hack to get the remotely loaded module
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    module_name = model_config.__class__.__module__.replace(
        ".configuration_mixformer_sequential", ".modeling_mixformer_sequential"
    )
    modeling_phi = importlib.import_module(module_name)
    modeling_phi.SelfAttention.forward = flash_self_attn_forward
    modeling_phi.CrossAttention.forward = flash_cross_attn_forward
    modeling_phi.MixFormerSequentialForCausalLM._no_split_modules = ["ParallelBlock"]


def flash_self_attn_forward(self, qkv, causal=None, key_padding_mask=None):
    causal = self.causal if causal is None else causal
    return flash_attn_qkvpacked_func(
        qkv, dropout_p=self.drop.p, softmax_scale=self.softmax_scale, causal=causal
    )


def flash_cross_attn_forward(self, q, kv, causal=None, key_padding_mask=None):
    causal = self.causal if causal is None else causal
    return flash_attn_kvpacked_func(
        q, kv, dropout_p=self.drop.p, softmax_scale=self.softmax_scale, causal=causal
    )
