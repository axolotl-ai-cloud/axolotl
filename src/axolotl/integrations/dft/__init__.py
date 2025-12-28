"""Composable DFT (Dynamic Fine-Tuning) plugin for Axolotl."""

from __future__ import annotations

from transformers import Trainer

from axolotl.integrations.base import BasePlugin
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class DFTPlugin(BasePlugin):
    """Enable ms-swift style DFT loss in Axolotl SFT training."""

    def get_input_args(self) -> str:
        return "axolotl.integrations.dft.args.DFTArgs"

    def get_training_args_mixin(self) -> str:
        return "axolotl.integrations.dft.args.DFTTrainingArgsMixin"

    def get_training_args(self, cfg: DictDefault) -> dict:
        if not cfg.enable_dft_loss:
            return {}
        return {"enable_dft_loss": True}

    def get_trainer_cls(self, cfg: DictDefault) -> None:
        return None

    def post_trainer_create(self, cfg: DictDefault, trainer: Trainer) -> None:
        if not cfg.enable_dft_loss:
            return
        if cfg.rl or cfg.reward_model or cfg.process_reward_model:
            LOG.warning("DFTPlugin is intended for SFT; skipping for RL/Reward paths.")
            return

        from .patch import patch_compute_loss_for_dft

        patch_compute_loss_for_dft(trainer, cfg)
        LOG.info("DFTPlugin: patched trainer.compute_loss")

