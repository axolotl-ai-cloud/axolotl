"""DenseMixer plugin for Axolotl"""

import importlib

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class DenseMixerPlugin(BasePlugin):
    """
    Plugin for DenseMixer
    """

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.densemixer.args.DenseMixerArgs"

    def pre_model_load(self, cfg):
        """Apply densemixer patches before model loading if enabled."""
        if cfg.dense_mixer:
            if not importlib.util.find_spec("densemixer"):
                raise RuntimeError(
                    "DenseMixer is not installed. Install it with `pip install densemixer`"
                )

            from densemixer.patching import (
                apply_olmoe_patch,
                apply_qwen2_moe_patch,
                apply_qwen3_moe_patch,
            )

            LOG.info(
                f"Applying DenseMixer patches for model type: {cfg.model_config_type}"
            )

            if cfg.model_config_type == "olmoe":
                apply_olmoe_patch()
            if cfg.model_config_type == "qwen2_moe":
                apply_qwen2_moe_patch()
            if cfg.model_config_type == "qwen3_moe":
                apply_qwen3_moe_patch()
