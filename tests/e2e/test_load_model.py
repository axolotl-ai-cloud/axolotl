"""Module for testing ModelLoader."""

import shutil
import tempfile

import pytest
import torch

from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.utils.dict import DictDefault


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
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "tokenizer_type": "AutoTokenizer",
                "tokenizer_config": "HuggingFaceTB/SmolLM2-135M",
                "sequence_len": 1024,
                "load_in_8bit": False,
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
                "micro_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "learning_rate": 0.00001,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "tensor_parallel_size": 1,
                "context_parallel_size": 1,
            }
        )
        self.model_loader = ModelLoader(
            cfg=self.cfg,
            tokenizer="",
            inference=False,
            reference_model=True,
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
        self.model_loader.tokenizer = load_tokenizer(self.cfg)
        self.model_loader.load()
        self.model_loader._convert_embedding_modules_dtype(
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
