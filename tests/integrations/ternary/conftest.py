"""Suite hygiene: tests that route configs through `normalize_config` flip global
torch precision state (tf32/matmul precision), which breaks later exact-equality
quantization tests. Every test runs inside a snapshot/restore of that state."""

import pytest
import torch


def _snapshot():
    state = {
        "matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "default_dtype": torch.get_default_dtype(),
    }
    if hasattr(torch.backends, "fp32_precision"):
        state["fp32_precision"] = torch.backends.fp32_precision
        state["fp32_precision_matmul"] = torch.backends.cuda.matmul.fp32_precision
        state["fp32_precision_cudnn"] = torch.backends.cudnn.conv.fp32_precision
    return state


def _restore(state):
    torch.set_float32_matmul_precision(state["matmul_precision"])
    torch.backends.cuda.matmul.allow_tf32 = state["allow_tf32_matmul"]
    torch.backends.cudnn.allow_tf32 = state["allow_tf32_cudnn"]
    torch.set_default_dtype(state["default_dtype"])
    if "fp32_precision" in state:
        torch.backends.fp32_precision = state["fp32_precision"]
        torch.backends.cuda.matmul.fp32_precision = state["fp32_precision_matmul"]
        torch.backends.cudnn.conv.fp32_precision = state["fp32_precision_cudnn"]


@pytest.fixture(autouse=True)
def _torch_precision_state():
    state = _snapshot()
    try:
        yield
    finally:
        _restore(state)
