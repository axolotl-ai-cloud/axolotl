"""
Module for the Plugin for LM Eval Harness
"""

import subprocess  # nosec

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.lm_eval.cli import build_lm_eval_command

from .args import LMEvalArgs  # pylint: disable=unused-import. # noqa: F401


class LMEvalPlugin(BasePlugin):
    """
    Plugin for LM Evaluation Harness integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.lm_eval.LMEvalArgs"

    def post_train_unload(self, cfg):
        if cfg.lm_eval_post_train:
            # pylint: disable=duplicate-code
            for lm_eval_args in build_lm_eval_command(
                cfg.lm_eval_tasks,
                bfloat16=cfg.bfloat16 or cfg.bf16,
                flash_attention=cfg.flash_attention,
                output_dir=cfg.output_dir,
                batch_size=cfg.lm_eval_batch_size,
                wandb_project=cfg.wandb_project,
                wandb_entity=cfg.wandb_entity,
                wandb_name=cfg.wandb_name,
                model=cfg.lm_eval_model or cfg.hub_model_id,
            ):
                subprocess.run(  # nosec
                    lm_eval_args,
                    check=True,
                )
