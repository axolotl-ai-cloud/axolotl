"""Shared pytest fixtures for axolotl.kernels tests."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up CUDA memory between tests."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    yield
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
