"""
E2E smoke tests for LLMCompressorPlugin integration
"""

from pathlib import Path

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

from tests.e2e.utils import (
    check_model_output_exists,
    require_llmcompressor,
    require_torch_2_4_1,
)

MODELS = [
    "nm-testing/llama2.c-stories42M-pruned2.4-compressed",
    "nm-testing/llama2.c-stories42M-gsm8k-sparse-only-compressed",
]


@pytest.mark.parametrize(
    "base_model", MODELS, ids=["no-checkpoint-recipe", "with-checkpoint-recipe"]
)
@pytest.mark.parametrize(
    "save_compressed", [True, False], ids=["save_compressed", "save_uncompressed"]
)
class TestLLMCompressorIntegration:
    """
    e2e tests for axolotl.integrations.llm_compressor.LLMCompressorPlugin
    """

    @require_llmcompressor
    @require_torch_2_4_1
    def test_llmcompressor_plugin(
        self, temp_dir, base_model: str, save_compressed: bool
    ):
        from llmcompressor import active_session

        # core cfg
        cfg = DictDefault(
            {
                "base_model": base_model,
                "plugins": ["axolotl.integrations.llm_compressor.LLMCompressorPlugin"],
                "sequence_len": 1024,
                "val_set_size": 0.05,
                "special_tokens": {"pad_token": "<|endoftext|>"},
                "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
                "num_epochs": 1,
                "micro_batch_size": 2,
                "gradient_accumulation_steps": 2,
                "output_dir": temp_dir,
                "learning_rate": 1e-5,
                "optimizer": "adamw_torch_fused",
                "lr_scheduler": "cosine",
                "save_safetensors": True,
                "bf16": "auto",
                "max_steps": 5,
                "llmcompressor": {
                    "recipe": {
                        "finetuning_stage": {
                            "finetuning_modifiers": {
                                "ConstantPruningModifier": {
                                    "targets": [
                                        "re:.*q_proj.weight",
                                        "re:.*k_proj.weight",
                                        "re:.*v_proj.weight",
                                        "re:.*o_proj.weight",
                                        "re:.*gate_proj.weight",
                                        "re:.*up_proj.weight",
                                        "re:.*down_proj.weight",
                                    ],
                                    "start": 0,
                                },
                            },
                        },
                    },
                    "save_compressed": save_compressed,
                },
                "save_first_step": False,
            }
        )

        prepare_plugins(cfg)
        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        try:
            train(cfg=cfg, dataset_meta=dataset_meta)
            check_model_output_exists(temp_dir, cfg)
            _check_llmcompressor_model_outputs(temp_dir, save_compressed)
        finally:
            active_session().reset()


def _check_llmcompressor_model_outputs(temp_dir, save_compressed):
    if save_compressed:
        assert (Path(temp_dir) / "recipe.yaml").exists()

        from compressed_tensors import ModelCompressor
        from compressed_tensors.config import Sparse24BitMaskConfig

        compressor = ModelCompressor.from_pretrained(temp_dir)
        assert compressor is not None
        assert isinstance(compressor.sparsity_config, Sparse24BitMaskConfig)
