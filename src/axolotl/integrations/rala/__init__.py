"""Definition of RALA plugin."""

import logging

from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.rala.auto.llama.modeling_rala import LlamaRALAAttention

LOG = logging.getLogger(__name__)


class RalaPlugin(BasePlugin):
    """
    Plugin for Rala integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.rala.args.RalaArgs"

    def pre_model_load(self, cfg):
        """Apply differential attention patch before model loading if enabled."""
        if cfg.rala_attention:
            LLAMA_ATTENTION_CLASSES["rala"] = LlamaRALAAttention

            from axolotl.monkeypatch.attention.differential import (
                patch_llama_attention_classes,
            )

            patch_llama_attention_classes()

    def set_attn_config(self, cfg, model_kwargs, model_config):
        if cfg.rala_attention:
            model_kwargs["attn_implementation"] = "rala"
