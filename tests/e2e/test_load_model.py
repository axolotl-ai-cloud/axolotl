"""Module for testing ModelLoader."""

import shutil
import tempfile

import pytest
import torch

from axolotl.utils.dict import DictDefault
from axolotl.utils.models import ModelLoader, load_model, load_tokenizer


@pytest.fixture(name="temp_dir")
def fixture_temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestLoadModelUtils:
    """
    Testing module testing ModelLoader.
    """

    def setup_method(self):
        # load config
        self.cfg = DictDefault(
            {
                "base_model": "JackFram/llama-68m",
                "model_type": "LlamaForCausalLM",
                "tokenizer_type": "LlamaTokenizer",
                "load_in_8bit": True,
                "load_in_4bit": False,
                "strict": False,
                "datasets": [
                    {
                        "path": "teknium/GPT4-LLM-Cleaned",
                        "type": "alpaca",
                    },
                ],
                "val_set_size": 0.02,
                "sequence_len": 1024,
                "sample_packing": True,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "lora_target_modules": [
                    "gate_proj",
                    "down_proj",
                    "up_proj",
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",
                ],
                "gradient_accumulation_steps": 1,
                "num_epochs": 1,
                "micro_batch_size": 2,
                "optimizer": "adamw_bnb_8bit",
                "lr_scheduler": "cosine",
                "learning_rate": 0.0002,
                "train_on_inputs": False,
                "group_by_length": False,
                "bf16": False,
                "fp16": True,
                "tf32": False,
                "gradient_checkpointing": True,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
            }
        )
        self.model_loader = (  # pylint: disable=attribute-defined-outside-init
            ModelLoader(
                cfg=self.cfg,
                tokenizer="",
            )
        )

    @pytest.mark.parametrize("embedding_modules", ["embed_tokens", "lm_head"])
    @pytest.mark.parametrize(
        "dist_dtype", [torch.bfloat16, torch.float16, torch.float32]
    )
    @pytest.mark.parametrize("before_kbit_train_or_finetune", [True, False])
    def test_convert_embedding_modules_dtype(
        self, temp_dir, embedding_modules, dist_dtype, before_kbit_train_or_finetune
    ):
        self.cfg.output_dir = temp_dir
        self.model_loader.tokenizer = load_tokenizer(self.cfg)  # pylint: disable=all
        self.model_loader.model, _ = load_model(
            self.cfg,
            self.model_loader.tokenizer,
            inference=False,
            reference_model=True,
        )
        self.model_loader.convert_embedding_modules_dtype(
            embedding_modules, dist_dtype, before_kbit_train_or_finetune
        )
        for name, module in self.model_loader.model.named_modules():
            if (
                "norm" in name
                or (before_kbit_train_or_finetune and name.endswith(".gate"))
                or (
                    any(m in name for m in embedding_modules)
                    and hasattr(module, "weight")
                )
            ):
                for _, param in module.named_parameters(recurse=False):
                    assert param.dtype == dist_dtype
