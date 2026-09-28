import os

import httpx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

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


def test_get_reward_func_unknown_hub_repo_raises_error(monkeypatch):
    """A reward function that is neither importable nor a hub repo gets a clear ValueError"""

    def repo_not_found(repo_id, repo_type=None):
        request = httpx.Request("GET", f"https://huggingface.co/api/models/{repo_id}")
        raise RepositoryNotFoundError(
            "404 Client Error", response=httpx.Response(404, request=request)
        )

    monkeypatch.setattr("axolotl.core.trainers.grpo.snapshot_download", repo_not_found)
    with pytest.raises(ValueError, match="Reward function .* not found"):
        GRPOStrategy.get_reward_func("nonexistent_module.my_reward")
