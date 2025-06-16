"""Test token weighting functionality for dataset merging."""

import pytest
from datasets import Dataset
from axolotl.utils.data.shared import merge_datasets, _validate_weights, _has_token_weighting
from axolotl.utils.dict import DictDefault


def create_sample_datasets():
    """Create sample datasets with input_ids for testing."""
    ds1 = Dataset.from_list([
        {"input_ids": [1, 2, 3, 4, 5]},  # 5 tokens
        {"input_ids": [6, 7, 8]},        # 3 tokens
    ])  # Total: 8 tokens
    
    ds2 = Dataset.from_list([
        {"input_ids": [9, 10, 11, 12]},  # 4 tokens
        {"input_ids": [13, 14]},         # 2 tokens
    ])  # Total: 6 tokens
    
    return [ds1, ds2]


def create_cfg():
    """Basic configuration for testing."""
    return DictDefault({
        "seed": 42,
        "shuffle_merged_datasets": True
    })


class TestTokenWeighting:
    """Test token weighting functionality."""

    def test_backward_compatibility_no_weights(self):
        """Test that merge_datasets works without weights (backward compatibility)."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        result = merge_datasets(sample_datasets, cfg)
        assert len(result) == 4  # 2 + 2 samples
        assert "input_ids" in result.features

    def test_backward_compatibility_default_weights(self):
        """Test that default weights (1.0) don't trigger token weighting."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 1.0, "weight_strategy": "upsample"}),
            DictDefault({"weight": 1.0, "weight_strategy": "upsample"})
        ]
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) == 4  # Should be same as no weighting

    def test_token_weighting_validation_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.7, "path": "dataset1"}),
            DictDefault({"weight": 0.4, "path": "dataset2"})  # Sum = 1.1
        ]
        
        with pytest.raises(ValueError, match="Dataset weights must sum to 1.0"):
            merge_datasets(sample_datasets, cfg, datasets_configs)

    def test_token_weighting_validation_range(self):
        """Test that weights must be between 0.0 and 1.0."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 1.5, "path": "dataset1"}),  # Invalid
            DictDefault({"weight": 0.5, "path": "dataset2"})
        ]
        
        with pytest.raises(ValueError, match="Dataset weight must be between 0.0 and 1.0"):
            merge_datasets(sample_datasets, cfg, datasets_configs)

    def test_token_weighting_equal_weights(self):
        """Test token weighting with equal weights."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.5, "weight_strategy": "upsample"}),
            DictDefault({"weight": 0.5, "weight_strategy": "upsample"})
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_token_weighting_unequal_weights(self):
        """Test token weighting with unequal weights."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.8, "weight_strategy": "upsample"}),
            DictDefault({"weight": 0.2, "weight_strategy": "upsample"})
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_downsample_strategy(self):
        """Test downsample strategy."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.3, "weight_strategy": "downsample"}),
            DictDefault({"weight": 0.7, "weight_strategy": "upsample"})
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_single_dataset_with_weights(self):
        """Test that single dataset bypasses weighting logic."""
        cfg = create_cfg()
        single_dataset = [Dataset.from_list([{"input_ids": [1, 2, 3]}])]
        datasets_configs = [DictDefault({"weight": 0.5})]
        
        result = merge_datasets(single_dataset, cfg, datasets_configs)
        assert len(result) == 1

    def test_has_token_weighting_detection(self):
        """Test _has_token_weighting helper function."""
        configs1 = [DictDefault({"weight": 1.0, "weight_strategy": "upsample"})]
        assert not _has_token_weighting(configs1)
        
        configs2 = [DictDefault({"weight": 0.5, "weight_strategy": "upsample"})]
        assert _has_token_weighting(configs2)
        
        configs3 = [DictDefault({"weight": 1.0, "weight_strategy": "downsample"})]
        assert _has_token_weighting(configs3)

    def test_validate_weights_helper(self):
        """Test _validate_weights helper function."""
        configs1 = [
            DictDefault({"weight": 0.3}),
            DictDefault({"weight": 0.7})
        ]
        _validate_weights(configs1)  # Should not raise
        
        configs2 = [
            DictDefault({"weight": 0.3}),
            DictDefault({"weight": 0.8})
        ]
        with pytest.raises(ValueError):
            _validate_weights(configs2)

    def test_zero_weight_validation(self):
        """Test that zero weights are allowed."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.0, "weight_strategy": "upsample"}),
            DictDefault({"weight": 1.0, "weight_strategy": "upsample"})
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_unknown_weight_strategy(self):
        """Test handling of unknown weight strategy."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.5, "weight_strategy": "unknown_strategy", "path": "dataset1"}),
            DictDefault({"weight": 0.5, "weight_strategy": "upsample", "path": "dataset2"})
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_downsample_with_weight_greater_than_one(self):
        """Test downsample strategy with weight >= 1 (should be ignored)."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 1.2, "weight_strategy": "downsample", "path": "dataset1"}),
            DictDefault({"weight": -0.2, "weight_strategy": "upsample", "path": "dataset2"})  # This will cause validation error
        ]
        
        with pytest.raises(ValueError, match="Dataset weight must be between 0.0 and 1.0"):
            merge_datasets(sample_datasets, cfg, datasets_configs)

    def test_floating_point_precision_weights(self):
        """Test that small floating point errors in weight sum are tolerated."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.1, "weight_strategy": "upsample"}),
            DictDefault({"weight": 0.9000000000000001, "weight_strategy": "upsample"})  # Sum slightly > 1.0
        ]
        
        result = merge_datasets(sample_datasets, cfg, datasets_configs)
        assert len(result) > 0
        assert "input_ids" in result.features

    def test_large_floating_point_error_weights(self):
        """Test that large floating point errors in weight sum are caught."""
        sample_datasets = create_sample_datasets()
        cfg = create_cfg()
        datasets_configs = [
            DictDefault({"weight": 0.1, "weight_strategy": "upsample"}),
            DictDefault({"weight": 0.95, "weight_strategy": "upsample"})  # Sum = 1.05, too large
        ]
        
        with pytest.raises(ValueError, match="Dataset weights must sum to 1.0"):
            merge_datasets(sample_datasets, cfg, datasets_configs)
