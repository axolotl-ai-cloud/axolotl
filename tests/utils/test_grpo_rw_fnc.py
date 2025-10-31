import os

import pytest

from axolotl.core.trainers.grpo import GRPOStrategy


def test_get_rollout_func_loads_successfully():
    """Test that a valid rollout function can be loaded"""
    rollout_func = GRPOStrategy.get_rollout_func("os.path.join")
    assert callable(rollout_func)
    assert rollout_func == os.path.join


def test_get_rollout_func_invalid_module_raises_error():
    """Test that invalid module path raises clear ValueError"""
    with pytest.raises(ValueError, match="Rollout function .* not found"):
        GRPOStrategy.get_rollout_func("nonexistent_module.my_func")
