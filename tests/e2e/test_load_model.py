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
                "tokenizer_type": "LlamaTokenizer",
                "tokenizer_config": "JackFram/llama-68m",
                "sequence_len": 1024,
                "load_in_8bit": False,
                "adapter": "lora",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_linear": True,
                "val_set_size": 0.1,
                "special_tokens": {
                    "unk_token": "<unk>",
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                },
                "datasets": [
                    {
                        "path": "mhenrichsen/alpaca_2k_test",
                        "type": "alpaca",
                    },
                ],
                "num_epochs": 1,
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
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
                for _, param in module.named_parameters():
                    assert param.dtype == dist_dtype
