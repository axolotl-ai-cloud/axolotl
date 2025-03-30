"""
Tests for loading DPO preference datasets with chatml formatting
"""

import unittest

import pytest

from axolotl.prompt_strategies.dpo import load as load_dpo
from axolotl.utils.data.rl import load_prepare_preference_datasets
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="minimal_dpo_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "tokenizer_config": "HuggingFaceTB/SmolLM2-135M",
            "rl": "dpo",
            "learning_rate": 0.000001,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "special_tokens": {
                "pad_token": "<|endoftext|>",
            },
            "sequence_len": 2048,
        }
    )


class TestDPOChatml:
    """
    Test loading DPO preference datasets with chatml formatting
    """

    @pytest.mark.skip(reason="TODO: fix hf hub offline to work with HF rate limits")
    @enable_hf_offline
    def test_default(self, minimal_dpo_cfg):
        cfg = DictDefault(
            {
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "chatml",
                        "split": "train[:1%]",
                    }
                ]
            }
            | minimal_dpo_cfg
        )

        # test that dpo.load works
        load_dpo("chatml", cfg)
        # now actually load the datasets with the strategy
        train_ds, _ = load_prepare_preference_datasets(cfg)
        assert train_ds[0]["prompt"].startswith("<|im_start|>")
        assert train_ds[0]["prompt"].endswith("<|im_start|>assistant\n")
        assert "chosen" in train_ds[0]
        assert "rejected" in train_ds[0]


if __name__ == "__main__":
    unittest.main()
