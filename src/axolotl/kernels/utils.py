"""Utilities for kernel submodules."""

import torch
import triton
from packaging.version import Version

if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type="cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type="cuda")


# pylint: disable=invalid-name
def calculate_grid(n_elements: int):
    """Calculate grid size based on input size."""
    BLOCK_SIZE = min(max(triton.next_power_of_2(n_elements), 128), 4096)
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    return (triton.cdiv(n_elements, BLOCK_SIZE),), BLOCK_SIZE, num_warps
