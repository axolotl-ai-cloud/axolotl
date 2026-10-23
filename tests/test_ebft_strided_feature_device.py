"""Tests for the device the strided EBFT feature network is placed on.

The trainer builds a frozen feature network next to the actor, so the two have to
live on the same accelerator: a run on an NPU (or any other non-CUDA accelerator)
must not end up with a feature network on a cuda device, and a device count read
from cuda must not decide how the network is created on such a run.
"""

import pytest
import torch

from axolotl.core.trainers.ebft.strided import _feature_device_type
from axolotl.utils.distributed import get_device_type


def accelerator_device(device_type: str, index: int = 0) -> torch.device:
    """A device of an accelerator type this torch build may not register yet.

    torch_npu registers 'npu' as the privateuse1 backend when it is imported; on a
    CPU-only runner that has not happened, so rename it the same way torch_npu does.
    """
    try:
        return torch.device(device_type, index)
    except RuntimeError:
        try:
            torch.utils.rename_privateuse1_backend(device_type)
        except (RuntimeError, AttributeError):
            pytest.skip(f"this torch build cannot represent a {device_type} device")
        return torch.device(device_type, index)


@pytest.mark.parametrize("device_type", ["cuda", "xpu", "npu"])
def test_feature_device_keeps_the_actor_accelerator(device_type):
    device = accelerator_device(device_type, 1)
    assert _feature_device_type(device) == device_type


@pytest.mark.parametrize("device_type", ["cpu", "meta"])
def test_placeholder_device_falls_back_to_the_run_device(device_type):
    assert _feature_device_type(torch.device(device_type)) == str(get_device_type())
