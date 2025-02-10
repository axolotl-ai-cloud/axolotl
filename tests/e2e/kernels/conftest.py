"""Shared pytest fixtures for axolotl.kernels tests."""

import pytest
import torch


@pytest.fixture
def device():
    """Returns the appropriate device for testing"""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up CUDA memory between tests."""
    yield
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
