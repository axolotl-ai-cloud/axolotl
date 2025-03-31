"""
e2e tests for kd trainer support in Axolotl
"""

from pathlib import Path

import pytest

from axolotl.cli.args import TrainerCliArgs
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import check_tensorboard, require_torch_2_5_1


@pytest.fixture(name="kd_min_cfg")
def min_cfg(temp_dir):
    return {
        "base_model": "osllmai-community/Llama-3.2-1B",
        "tokenizer_config": "axolotl-ai-co/Llama-3.3-70B-Instruct-tokenizer",
        "plugins": [
            "axolotl.integrations.kd.KDPlugin",
            "axolotl.integrations.liger.LigerPlugin",
        ],
        "liger_rms_norm": True,
        "liger_glu_activation": True,
        "torch_compile": True,
        "chat_template": "llama3",
        "kd_trainer": True,
        "kd_ce_alpha": 0.1,
        "kd_alpha": 0.9,
        "kd_temperature": 1.0,
        "dataloader_prefetch_factor": 8,
        "dataloader_num_workers": 4,
        "dataloader_pin_memory": True,
        "datasets": [
            {
                "path": "axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample",
                "type": "axolotl.integrations.kd.chat_template",
                "field_messages": "messages_combined",
                "split": "train",
                "logprobs_field": "llm_text_generation_vllm_logprobs",
                "temperature": 1.0,
                "preprocess_shards": 2,
            },
        ],
        "val_set_size": 0.0,
        "sequence_len": 2048,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 1,
        "num_epochs": 1,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.00001,
        "bf16": "auto",
        "gradient_checkpointing": True,
        "flash_attention": True,
        "special_tokens": {
            "pad_token": "<|end_of_text|>",
            "eos_token": "<|eot_id|>",
        },
        "max_steps": 5,
        "output_dir": temp_dir,
        "save_safetensors": True,
        "use_tensorboard": True,
    }


class TestKnowledgeDistillation:
    """
    Test case for Knowledge Distillation
    """

    # While this will run on torch 2.4.x without torch_compile enabled
    # the VRAM requirement is higher than what is available in CI
    @require_torch_2_5_1
    def test_llama_kd(self, temp_dir, kd_min_cfg):
        cfg = DictDefault(kd_min_cfg)
        # pylint: disable=duplicate-code
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 1.0, "Train Loss is too high"
        )

    @pytest.mark.parametrize(
        "load_in_8bit",
        [True, False],
    )
    def test_llama_lora_kd(self, temp_dir, kd_min_cfg, load_in_8bit):
        cfg = DictDefault(
            {
                "load_in_8bit": load_in_8bit,
                "torch_compile": False,
                "adapter": "lora",
                "peft_use_dora": True,
                "lora_target_linear": True,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
            }
            | kd_min_cfg
        )
        # pylint: disable=duplicate-code
        cfg = validate_config(cfg)
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 1.0, "Train Loss is too high"
        )
