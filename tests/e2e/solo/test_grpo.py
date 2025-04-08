"""
E2E tests for GRPO
"""

import logging
import os
import unittest

import transformers

from axolotl.cli.args import PreprocessCliArgs
from axolotl.common.datasets import load_preference_datasets
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import with_temp_dir

LOG = logging.getLogger("axolotl.tests.e2e")
os.environ["WANDB_DISABLED"] = "true"


class TestGRPO(unittest.TestCase):
    """
    Test case for GRPO training using single GPU
    """

    def _utils_write_rewards(self):
        # write cfg to yaml file
        with open("rewards.py", "w", encoding="utf-8") as fout:
            fout.write(
                """import random
def rand_reward_func(completions, **kwargs) -> list[float]:
    return [random.uniform(0, 1) for _ in completions]

def oai_gsm8k_transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        label = example["answer"].split("####")[-1].strip().replace(",", "")
        return {
            "prompt": [{"role": "user", "content": example["question"]},],
            "answer": label,
        }
    return transform_fn, {"remove_columns": ["question"]}
"""
            )

    @with_temp_dir
    def test_custom_rewards_fn_preprocess(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "rl": "grpo",
                "trl": [
                    {
                        "beta": 0.001,
                        "max_completion_length": 256,
                        "use_vllm": True,
                        "num_generations": 4,
                        "reward_funcs": [
                            "rewards.rand_reward_func"
                        ],  # format: '{file_name}.{fn_name}'
                        "reward_weights": [1.0],
                    },
                ],
                "datasets": [
                    {
                        "path": "openai/gsm8k",
                        "name": "main",
                        "type": "rewards.oai_gsm8k_transform",
                    },
                ],
                "output_dir": temp_dir,
            }
        )

        self._utils_write_rewards()

        cfg = validate_config(cfg)
        normalize_config(cfg)
        parser = transformers.HfArgumentParser(PreprocessCliArgs)
        cli_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

        load_preference_datasets(cfg=cfg, cli_args=cli_args)
