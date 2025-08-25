"""
E2E tests for activation offloading
"""

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists


class TestActivationOffloading:
    """
    E2E test cases for activation offloading
    """

    @pytest.mark.parametrize(
        "adapter",
        ["lora", "qlora", None],
    )
    def test_activation_offloading(
        self,
        temp_dir,
        adapter,
    ):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "val_set_size": 0.0,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                    "eos_token": "<|im_end|>",
                },
                "datasets": [
                    {
                        "chat_template": "chatml",
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "split": "train[:10%]",
                        "field_messages": "conversations",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    },
                ],
                "num_epochs": 1,
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": True,
                "bf16": "auto",
                "save_safetensors": True,
                "gradient_checkpointing": True,
                "activation_offloading": True,
                "save_first_step": False,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
            }
        )
        if adapter == "lora":
            cfg["adapter"] = "lora"
        if adapter == "qlora":
            cfg["adapter"] = "qlora"
            cfg["load_in_4bit"] = True

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(temp_dir, cfg)
