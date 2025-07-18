"""
e2e gpu test for the pytorch profiler callback
"""

from pathlib import Path

import pytest

from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="profiler_base_cfg")
def fixture_profiler_base_cfg():
    cfg = DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        tokenizer_type="AutoTokenizer",
        sequence_len=1024,
        load_in_8bit=True,
        adapter="lora",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_target_linear=True,
        val_set_size=0.02,
        special_tokens={"pad_token": "<|endoftext|>"},
        datasets=[
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        num_epochs=1,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=0.00001,
        optimizer="adamw_torch_fused",
        lr_scheduler="cosine",
    )
    return cfg


class TestProfiler:
    """
    test cases for the pytorch profiler callback
    """

    def test_profiler_saves(self, profiler_base_cfg, temp_dir):
        cfg = profiler_base_cfg | DictDefault(
            output_dir=temp_dir,
            max_steps=5,
            profiler_steps=3,
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "snapshot.pickle").exists()

    def test_profiler_saves_w_start(self, profiler_base_cfg, temp_dir):
        cfg = profiler_base_cfg | DictDefault(
            output_dir=temp_dir,
            max_steps=5,
            profiler_steps=3,
            profiler_steps_start=1,
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "snapshot.pickle").exists()

    @pytest.mark.parametrize(
        "profiler_steps_start",
        [3, 5],
    )
    def test_profiler_saves_past_end(
        self, profiler_base_cfg, temp_dir, profiler_steps_start
    ):
        cfg = profiler_base_cfg | DictDefault(
            output_dir=temp_dir,
            max_steps=5,
            profiler_steps=3,
            profiler_steps_start=profiler_steps_start,
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert (Path(temp_dir) / "snapshot.pickle").exists()

    def test_profiler_never_started(self, profiler_base_cfg, temp_dir):
        cfg = profiler_base_cfg | DictDefault(
            output_dir=temp_dir,
            max_steps=5,
            profiler_steps=3,
            profiler_steps_start=6,
        )

        cfg = validate_config(cfg)
        normalize_config(cfg)
        dataset_meta = load_datasets(cfg=cfg)

        train(cfg=cfg, dataset_meta=dataset_meta)
        assert not (Path(temp_dir) / "snapshot.pickle").exists()
