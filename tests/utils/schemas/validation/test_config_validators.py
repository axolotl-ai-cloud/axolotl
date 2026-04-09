"""
Tests for new config validators added to AxolotlInputConfig.

Covers:
  - save_strategy: 'best' requires metric_for_best_model
  - streaming=True with val_set_size > 0 is rejected
  - lora_target_modules with invalid regex patterns is rejected
"""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestSaveStrategyBestValidator:
    """save_strategy: 'best' must be accompanied by metric_for_best_model."""

    def test_save_strategy_best_without_metric_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="best")
        with pytest.raises(ValueError, match="metric_for_best_model"):
            validate_config(cfg)

    def test_save_strategy_best_with_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            save_strategy="best",
            metric_for_best_model="eval_loss",
        )
        validated = validate_config(cfg)
        assert validated.save_strategy == "best"
        assert validated.metric_for_best_model == "eval_loss"

    def test_save_strategy_epoch_without_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="epoch")
        validated = validate_config(cfg)
        assert validated.save_strategy == "epoch"

    def test_save_strategy_no_without_metric_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(save_strategy="no")
        validated = validate_config(cfg)
        assert validated.save_strategy == "no"

    def test_save_strategy_unset_without_metric_passes(self, min_base_cfg):
        """The default (None / not set) should not require metric_for_best_model."""
        validated = validate_config(min_base_cfg)
        assert validated.save_strategy is None


class TestStreamingWithValSetSizeValidator:
    """streaming=True is incompatible with val_set_size > 0."""

    def test_streaming_with_val_set_size_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            streaming=True, val_set_size=0.1, max_steps=100
        )
        with pytest.raises(ValueError, match="val_set_size"):
            validate_config(cfg)

    def test_streaming_with_val_set_size_zero_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            streaming=True, val_set_size=0.0, max_steps=100
        )
        validated = validate_config(cfg)
        assert validated.streaming is True

    def test_streaming_false_with_val_set_size_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(streaming=False, val_set_size=0.1)
        validated = validate_config(cfg)
        assert validated.val_set_size == pytest.approx(0.1)

    def test_streaming_unset_with_val_set_size_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(val_set_size=0.2)
        validated = validate_config(cfg)
        assert validated.val_set_size == pytest.approx(0.2)


class TestLoraTargetModulesRegexValidator:
    """lora_target_modules entries must be valid Python regex patterns."""

    def test_invalid_regex_raises(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "[invalid_regex"],
        )
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_config(cfg)

    def test_valid_regex_passes(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "v_proj", r".*_proj"],
        )
        validated = validate_config(cfg)
        assert "q_proj" in validated.lora_target_modules

    def test_plain_module_names_pass(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        validated = validate_config(cfg)
        assert len(validated.lora_target_modules) == 4

    def test_lora_target_linear_string_not_validated(self, min_base_cfg):
        """When lora_target_modules is a string (e.g. 'all-linear'), skip regex check."""
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules="all-linear",
        )
        # Should not raise
        validate_config(cfg)

    def test_multiple_invalid_patterns_reported(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            adapter="lora",
            lora_target_modules=["[bad1", "[bad2"],
        )
        with pytest.raises(ValueError, match="invalid regex pattern"):
            validate_config(cfg)
