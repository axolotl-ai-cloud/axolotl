"""
Module for the Plugin for LM Eval Harness
"""
import subprocess  # nosec
from datetime import datetime

from axolotl.integrations.base import BasePlugin

from .args import LMEvalArgs  # pylint: disable=unused-import. # noqa: F401


class LMEvalPlugin(BasePlugin):
    """
    Plugin for LM Evaluation Harness integraton with Axolotl.
    """

    def get_input_args(self):
        return "axolotl.integrations.lm_eval.LMEvalArgs"

    def post_train_unload(self, cfg):
        tasks = ",".join(cfg.lm_eval_tasks)
        fa2 = ",attn_implementation=flash_attention_2" if cfg.flash_attention else ""
        dtype = ",dtype=bfloat16" if cfg.bf16 else ",dtype=float16"
        output_path = cfg.output_dir
        output_path += "" if cfg.output_dir.endswith("/") else "/"
        output_path += "lm_eval_results/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        subprocess.run(  # nosec
            [
                "lm_eval",
                "--model",
                "hf",
                "--model_args",
                f"pretrained={cfg.output_dir}{fa2}{dtype}",
                "--tasks",
                tasks,
                "--batch_size",
                str(cfg.lm_eval_batch_size),
                "--output_path",
                output_path,
            ],
            check=True,
        )
