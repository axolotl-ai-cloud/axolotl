"""
Module for the Plugin for LM Eval Harness
"""

import subprocess  # nosec

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.lm_eval.cli import build_lm_eval_command, get_model_path

from .args import LMEvalArgs as LMEvalArgs


class LMEvalPlugin(BasePlugin):
    """
    Plugin for LM Evaluation Harness integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.lm_eval.LMEvalArgs"

    def post_train_unload(self, cfg):
        if cfg.lm_eval_post_train:
            for lm_eval_args in build_lm_eval_command(
                cfg.lm_eval_tasks,
                bfloat16=cfg.bfloat16 or cfg.bf16,
                flash_attention=cfg.flash_attention,
                output_dir=cfg.output_dir,
                batch_size=cfg.lm_eval_batch_size,
                wandb_project=cfg.wandb_project,
                wandb_entity=cfg.wandb_entity,
                wandb_name=cfg.wandb_name,
                model=get_model_path(cfg),
            ):
                subprocess.run(  # nosec
                    lm_eval_args,
                    check=True,
                )
