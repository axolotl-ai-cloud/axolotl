"""
e2e tests for kd trainer support in Axolotl
"""
from pathlib import Path

import pytest
from e2e.utils import check_tensorboard

from axolotl.cli import load_datasets
from axolotl.common.cli import TrainerCliArgs
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="kd_min_cfg")
def min_cfg(temp_dir):
    return {
        "base_model": "unsloth/Llama-3.2-1B",
        "plugins": [
            "axolotl.integrations.kd.KDPlugin",
            "axolotl.integrations.liger.LigerPlugin",
        ],
        "liger_rms_norm": True,
        "liger_glu_activation": True,
        "torch_compile": False,
        "chat_template": "llama3",
        "kd_trainer": True,
        "kd_ce_alpha": 0.1,
        "kd_alpha": 0.9,
        "kd_temperature": 2.0,
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
        "sequence_len": 4096,
        "sample_packing": True,
        "pad_to_sequence_len": True,
        "gradient_accumulation_steps": 2,
        "micro_batch_size": 2,
        "num_epochs": 1,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "learning_rate": 0.0001,
        "bf16": "auto",
        "gradient_checkpointing": True,
        "flash_attention": True,
        "special_tokens": {
            "pad_token": "<|end_of_text|>",
            "eos_token": "<|eot_id|>",
        },
        "max_steps": 5,
        "output_dir": temp_dir,
    }


class TestKnowledgeDistillation:
    """
    Test case for Knowledge Distillation
    """

    def test_llama_kd(self, temp_dir, kd_min_cfg):
        cfg = DictDefault(kd_min_cfg)
        # pylint: disable=duplicate-code
        prepare_plugins(cfg)
        normalize_config(cfg)
        cli_args = TrainerCliArgs()
        dataset_meta = load_datasets(cfg=cfg, cli_args=cli_args)

        train(cfg=cfg, cli_args=cli_args, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "model.safetensors").exists()
        check_tensorboard(
            temp_dir + "/runs", "train/loss", 1.0, "Train Loss is too high"
        )
