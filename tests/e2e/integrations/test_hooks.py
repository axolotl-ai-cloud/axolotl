"""
e2e tests to make sure all the hooks are fired on the plugin
"""

import os
from pathlib import Path

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.integrations.base import BasePlugin
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from ..utils import check_model_output_exists


class LogHooksPlugin(BasePlugin):
    """
    fixture to capture in a log file each hook that was fired
    """

    base_dir = Path("/tmp/axolotl-log-hooks")

    def __init__(self):
        self.base_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.remove(self.base_dir.joinpath("plugin_hooks.log"))
        except FileNotFoundError:
            pass

    def pre_model_load(self, cfg):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("pre_model_load\n")

    def post_model_build(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("post_model_build\n")

    def pre_lora_load(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("pre_lora_load\n")

    def post_lora_load(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("post_lora_load\n")

    def post_model_load(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("post_model_load\n")

    def create_optimizer(self, cfg, trainer):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("create_optimizer\n")

    def get_trainer_cls(self, cfg):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("get_trainer_cls\n")

    def create_lr_scheduler(
        self, cfg, trainer, optimizer, num_training_steps
    ):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("create_lr_scheduler\n")

    def add_callbacks_pre_trainer(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("add_callbacks_pre_trainer\n")
        return []

    def add_callbacks_post_trainer(
        self, cfg, trainer
    ):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("add_callbacks_post_trainer\n")
        return []

    def post_train(self, cfg, model):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("post_train\n")

    def post_train_unload(self, cfg):  # pylint: disable=unused-argument
        with open(
            self.base_dir.joinpath("plugin_hooks.log"), "a", encoding="utf-8"
        ) as f:
            f.write("post_train_unload\n")


class TestPluginHooks:
    """
    e2e tests to make sure all the hooks are fired during the training
    """

    def test_plugin_hooks(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "plugins": [
                    "tests.e2e.integrations.test_hooks.LogHooksPlugin",
                ],
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.02,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "flash_attention": True,
                "bf16": "auto",
            }
        )

        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

        with open(
            "/tmp/axolotl-log-hooks" + "/plugin_hooks.log", "r", encoding="utf-8"
        ) as f:
            file_contents = f.readlines()
            file_contents = "\n".join(file_contents)
            assert "pre_model_load" in file_contents
            assert "post_model_build" in file_contents
            assert "pre_lora_load" in file_contents
            assert "post_lora_load" in file_contents
            assert "post_model_load" in file_contents
            # assert "create_optimizer" in file_contents  # not implemented yet
            assert "get_trainer_cls" in file_contents
            assert "create_lr_scheduler" in file_contents
            assert "add_callbacks_pre_trainer" in file_contents
            assert "add_callbacks_post_trainer" in file_contents
            assert "post_train" in file_contents
            # assert "post_train_unload" in file_contents  # not called from test train call

        try:
            os.remove("/tmp/axolotl-log-hooks" + "/plugin_hooks.log")
        except FileNotFoundError:
            pass
