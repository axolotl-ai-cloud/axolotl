"""Module for testing models utils file."""

import pytest
import torch

from axolotl.cli import load_cfg
from axolotl.utils.models import ModelLoader, load_model, load_tokenizer


class TestLoadModelUtils:
    """Testing module for models utils."""

    def setup_method(self) -> None:
        # load config
        self.cfg = load_cfg(  # pylint: disable=attribute-defined-outside-init
            "examples/openllama-3b/config.yml"
        )
        # flash-attn not installed in env of CI
        self.cfg.flash_attention = (
            False  # pylint: disable=attribute-defined-outside-init
        )
        self.tokenizer = load_tokenizer(self.cfg)  # pylint: disable=all
        self.model_loader = (  # pylint: disable=attribute-defined-outside-init
            ModelLoader(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )
        )
        self.model_loader.model, _ = load_model(
            self.cfg,
            self.tokenizer,
            inference=False,
            reference_model=True,
        )

    @pytest.mark.parametrize("embedding_modules", ["embed_tokens", "lm_head"])
    @pytest.mark.parametrize(
        "dist_dtype", [torch.bfloat16, torch.float16, torch.float32]
    )
    @pytest.mark.parametrize("before_kbit_train_or_finetune", [True, False])
    def test_convert_embedding_modules_dtype(
        self, embedding_modules, dist_dtype, before_kbit_train_or_finetune
    ):
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
