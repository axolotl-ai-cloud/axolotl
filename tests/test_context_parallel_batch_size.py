"""Tests for batch_size calculation with context parallelism."""

import pytest

from axolotl.utils.config import (
    ensure_context_parallel_plugin,
    normalize_config,
    validate_config,
)
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="cp_base_cfg")
def fixture_cp_base_cfg(min_base_cfg):
    return (
        DictDefault(
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_len=2048,
            num_epochs=1,
            flash_attention=True,
        )
        | min_base_cfg
    )


class TestContextParallelBatchSize:
    """Verify batch_size scales by effective dp world_size when using context parallelism."""

    @pytest.mark.parametrize(
        "world_size, context_parallel_size, expected_batch_size",
        [
            (4, 1, 32),  # no CP: 2*4*4 = 32
            (4, 2, 16),  # CP=2: 2*4*(4//2) = 16
            (4, 4, 8),  # CP=4: 2*4*(4//4) = 8
            (2, 2, 8),  # CP=ws: 2*4*(2//2) = 8 (no scaling)
        ],
    )
    def test_batch_size_with_context_parallelism(
        self,
        cp_base_cfg,
        monkeypatch,
        world_size,
        context_parallel_size,
        expected_batch_size,
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        cp_base_cfg["context_parallel_size"] = context_parallel_size
        ensure_context_parallel_plugin(cp_base_cfg)
        cfg = validate_config(cp_base_cfg)
        normalize_config(cfg)
        assert cfg.batch_size == expected_batch_size
