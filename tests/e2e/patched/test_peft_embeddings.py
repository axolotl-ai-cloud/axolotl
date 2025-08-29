"""
Test case for handling embeddings when using peft
"""

import torch

from axolotl.train import setup_model_and_tokenizer
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


class TestLlamaPeftEmbeddings:
    """
    test class for handling embeddings when using peft
    """

    def test_peft_embeddings_upcast(self, temp_dir):
        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "load_in_4bit": True,
                "adapter": "qlora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_target_linear": True,
                "trust_remote_code": True,
                "sequence_len": 512,
                "val_set_size": 0.01,
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
                "max_steps": 2,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "output_dir": temp_dir,
                "learning_rate": 0.00001,
                "optimizer": "adamw_8bit",
                "lr_scheduler": "cosine",
                "flash_attention": True,
                "sample_packing": False,
                "bf16": "auto",
                "save_safetensors": True,
                "embeddings_skip_upcast": True,
                "save_first_step": False,
            }
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)

        model, _, _, _ = setup_model_and_tokenizer(cfg)

        # Check if the embeddings are upcast correctly
        # only embed_tokens is a parameter that may be upcast
        assert model.base_model.model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert model.base_model.model.lm_head.weight.dtype == torch.bfloat16
