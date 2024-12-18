"""Definition of differential transformer plugin."""

import logging

from axolotl.integrations.base import BasePlugin

LOG = logging.getLogger(__name__)


class DifferentialTransformerPlugin(BasePlugin):
    """
    Plugin for differential transformer integration with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.differential_transformer.args.DifferentialTransformerArgs"

    def pre_model_load(self, cfg):
        """Apply differential attention patch before model loading if enabled."""
        if cfg.differential_attention:
            from axolotl.monkeypatch.attention.differential import (
                patch_llama_attention_classes,
            )

            patch_llama_attention_classes()
