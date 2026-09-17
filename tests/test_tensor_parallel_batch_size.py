"""Tests for batch_size calculation with tensor parallelism."""

from unittest.mock import patch

import addict
import pytest

from axolotl.utils.config import normalize_config, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="tp_base_cfg")
def fixture_tp_base_cfg(min_base_cfg):
    return (
        DictDefault(
            micro_batch_size=2,
            gradient_accumulation_steps=4,
            sequence_len=2048,
            num_epochs=1,
        )
        | min_base_cfg
    )


class TestTensorParallelBatchSize:
    """Verify batch_size scales by effective dp world_size when using tensor parallelism."""

    @pytest.mark.parametrize(
        "world_size, tensor_parallel_size, expected_batch_size",
        [
            (4, 1, 32),  # no TP: 2*4*4 = 32
            (4, 2, 16),  # TP=2: 2*4*(4//2) = 16
            (4, 4, 8),  # TP=4: 2*4*(4//4) = 8
            (2, 2, 8),  # TP=ws: 2*4*(2//2) = 8 (no scaling)
        ],
    )
    def test_batch_size_with_tensor_parallelism(
        self,
        tp_base_cfg,
        monkeypatch,
        world_size,
        tensor_parallel_size,
        expected_batch_size,
    ):
        monkeypatch.setenv("WORLD_SIZE", str(world_size))
        tp_base_cfg["tensor_parallel_size"] = tensor_parallel_size
        cfg = validate_config(tp_base_cfg)
        # Mock load_model_config to avoid downloading the model and to bypass
        # the tie_word_embeddings validation that blocks TP > 1.
        with patch(
            "axolotl.utils.config.load_model_config",
            return_value=addict.Dict({"model_type": "llama"}),
        ):
            normalize_config(cfg)
        assert cfg.batch_size == expected_batch_size
