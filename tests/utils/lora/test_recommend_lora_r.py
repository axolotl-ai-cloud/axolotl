"""Tests for the recommend_lora_r utility and its integration in load_datasets."""

import pytest

from axolotl.utils.lora import recommend_lora_r


class TestRecommendLoraR:
    """Unit tests for the recommend_lora_r heuristic."""

    @pytest.mark.parametrize(
        "dataset_size, expected_rank",
        [
            (0, 8),
            (1, 8),
            (999, 8),
            (1_000, 16),
            (5_000, 16),
            (9_999, 16),
            (10_000, 32),
            (50_000, 32),
            (99_999, 32),
            (100_000, 64),
            (500_000, 64),
            (1_000_000, 64),
        ],
    )
    def test_rank_thresholds(self, dataset_size, expected_rank):
        assert recommend_lora_r(dataset_size) == expected_rank

    def test_returns_power_of_two(self):
        for size in [0, 500, 5_000, 50_000, 500_000]:
            rank = recommend_lora_r(size)
            assert rank > 0 and (rank & (rank - 1)) == 0, (
                f"rank {rank} is not a power of two"
            )

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            recommend_lora_r(-1)

    def test_boundary_exactly_at_threshold(self):
        # Boundaries should map to the higher rank
        assert recommend_lora_r(1_000) == 16
        assert recommend_lora_r(10_000) == 32
        assert recommend_lora_r(100_000) == 64

    def test_rank_monotonically_non_decreasing(self):
        sizes = [0, 999, 1_000, 9_999, 10_000, 99_999, 100_000, 1_000_000]
        ranks = [recommend_lora_r(s) for s in sizes]
        for i in range(len(ranks) - 1):
            assert ranks[i] <= ranks[i + 1], (
                f"rank decreased from size {sizes[i]} to {sizes[i+1]}"
            )


class TestLoadDatasetsLoraRAutoSet:
    """Integration-style tests for the auto-set behavior in load_datasets."""

    def _make_cfg(self, adapter, lora_r=None):
        from axolotl.utils.dict import DictDefault

        return DictDefault(
            {
                "adapter": adapter,
                "lora_r": lora_r,
                "processor_type": None,
            }
        )

    def test_auto_sets_lora_r_for_lora(self):
        from unittest.mock import MagicMock, patch

        cfg = self._make_cfg("lora")
        fake_dataset = MagicMock()
        fake_dataset.__len__ = lambda self: 5_000

        with (
            patch("axolotl.common.datasets.load_tokenizer"),
            patch("axolotl.common.datasets.load_processor"),
            patch(
                "axolotl.common.datasets.prepare_datasets",
                return_value=(fake_dataset, None, 100, []),
            ),
        ):
            from axolotl.common.datasets import load_datasets

            result = load_datasets(cfg=cfg)

        assert cfg.lora_r == 16  # 5_000 samples → rank 16

    def test_auto_sets_lora_r_for_qlora(self):
        from unittest.mock import MagicMock, patch

        cfg = self._make_cfg("qlora")
        fake_dataset = MagicMock()
        fake_dataset.__len__ = lambda self: 150_000

        with (
            patch("axolotl.common.datasets.load_tokenizer"),
            patch("axolotl.common.datasets.load_processor"),
            patch(
                "axolotl.common.datasets.prepare_datasets",
                return_value=(fake_dataset, None, 100, []),
            ),
        ):
            from axolotl.common.datasets import load_datasets

            result = load_datasets(cfg=cfg)

        assert cfg.lora_r == 64  # 150_000 samples → rank 64

    def test_does_not_override_explicit_lora_r(self):
        from unittest.mock import MagicMock, patch

        cfg = self._make_cfg("lora", lora_r=32)
        fake_dataset = MagicMock()
        fake_dataset.__len__ = lambda self: 500

        with (
            patch("axolotl.common.datasets.load_tokenizer"),
            patch("axolotl.common.datasets.load_processor"),
            patch(
                "axolotl.common.datasets.prepare_datasets",
                return_value=(fake_dataset, None, 100, []),
            ),
        ):
            from axolotl.common.datasets import load_datasets

            load_datasets(cfg=cfg)

        assert cfg.lora_r == 32  # user-set value preserved

    def test_no_auto_set_without_lora_adapter(self):
        from unittest.mock import MagicMock, patch

        cfg = self._make_cfg(adapter=None)
        fake_dataset = MagicMock()
        fake_dataset.__len__ = lambda self: 50_000

        with (
            patch("axolotl.common.datasets.load_tokenizer"),
            patch("axolotl.common.datasets.load_processor"),
            patch(
                "axolotl.common.datasets.prepare_datasets",
                return_value=(fake_dataset, None, 100, []),
            ),
        ):
            from axolotl.common.datasets import load_datasets

            load_datasets(cfg=cfg)

        assert cfg.lora_r is None  # not a LoRA run, should stay None
