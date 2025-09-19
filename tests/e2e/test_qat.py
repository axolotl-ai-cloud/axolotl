"""
E2E tests for QAT
"""

from pathlib import Path

from axolotl.common.datasets import load_datasets, load_preference_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault

from .utils import check_model_output_exists, check_tensorboard


class TestQATLlama:
    """
    Test case for QAT Llama models
    """

    def test_qat(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "sequence_len": 1024,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "datasets": [
                    {
                        "path": "mlabonne/FineTome-100k",
                        "type": "chat_template",
                        "field_messages": "conversations",
                        "message_property_mappings": {
                            "role": "from",
                            "content": "value",
                        },
                        "drop_system_message": True,
                        "split": "train[:1%]",
                    },
                ],
                "chat_template": "chatml",
                "qat": {
                    "quantize_embedding": True,
                    "activation_dtype": "int8",
                    "weight_dtype": "int4",
                    "group_size": 8,
                },
                "num_epochs": 1,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "max_steps": 5,
                "save_safetensors": True,
                "bf16": True,
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-5", cfg)

    def test_qat_dpo(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 2048,
                "sample_packing": False,
                "eval_sample_packing": False,
                "pad_to_sequence_len": True,
                "val_set_size": 0.01,
                "special_tokens": {
                    "pad_token": "<|endoftext|>",
                },
                "rl": "dpo",
                "chat_template": "chatml",
                "datasets": [
                    {
                        "path": "fozziethebeat/alpaca_messages_2k_dpo_test",
                        "type": "chat_template.default",
                        "field_messages": "conversation",
                        "field_chosen": "chosen",
                        "field_rejected": "rejected",
                        "message_field_role": "role",
                        "message_field_content": "content",
                        "roles": {
                            "system": ["system"],
                            "user": ["user"],
                            "assistant": ["assistant"],
                        },
                    },
                ],
                "num_epochs": 1,
                "max_steps": 5,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "warmup_steps": 0,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "use_tensorboard": True,
                "bf16": True,
                "qat": {
                    "quantize_embedding": True,
                    "activation_dtype": "int8",
                    "weight_dtype": "int4",
                    "group_size": 8,
                },
                "save_first_step": False,
            }
        )
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_preference_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        check_model_output_exists(Path(temp_dir) / "checkpoint-5", cfg)

        loss_threshold = 2.3
        check_tensorboard(
            temp_dir + "/runs",
            "train/train_loss",
            loss_threshold,
            "Train Loss (%s) is too high",
        )
