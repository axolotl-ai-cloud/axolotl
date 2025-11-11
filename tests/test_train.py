"""Test for batch size calculation for multi-gpu training."""

import pytest

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="train_base_cfg")
def fixture_train_base_cfg(min_base_cfg):
    return (
        DictDefault(
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_len=2048,
            sample_packing=True,
            num_epochs=1,
        )
        | min_base_cfg
    )


class TestTrain:
    """test class for train related tests"""

    @pytest.mark.parametrize(
        "world_size, expected_batch_size",
        [
            (1, 8),
            (4, 32),
        ],
    )
    def test_batch_size_ddp(
        self, train_base_cfg, monkeypatch, world_size, expected_batch_size
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        cfg = validate_config(train_base_cfg)
        normalize_config(cfg)
        assert cfg.batch_size == expected_batch_size
