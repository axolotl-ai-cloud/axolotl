"""
Simple end-to-end test for Liger integration
"""

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_model_output_exists, require_torch_2_4_1


class LigerIntegrationTestCase:
    """
    e2e tests for liger integration with Axolotl
    """

    @require_torch_2_4_1
    def test_llama_wo_flce(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_glu_activation": True,
                "liger_cross_entropy": True,
                "liger_fused_linear_cross_entropy": False,
                "sequence_len": 1024,
                "val_set_size": 0.05,
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
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 5,
            }
        )
        # pylint: disable=duplicate-code
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)

    @require_torch_2_4_1
    def test_llama_w_flce(self, temp_dir):
        # pylint: disable=duplicate-code
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "plugins": [
                    "axolotl.integrations.liger.LigerPlugin",
                ],
                "liger_rope": True,
                "liger_rms_norm": True,
                "liger_glu_activation": True,
                "liger_cross_entropy": False,
                "liger_fused_linear_cross_entropy": True,
                "sequence_len": 1024,
                "val_set_size": 0.05,
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
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 5,
            }
        )
        # pylint: disable=duplicate-code
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
