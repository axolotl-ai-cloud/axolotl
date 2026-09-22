"""Utilities for `axolotl.kernels` submodules."""

import torch
from packaging.version import Version

# Detect the actual accelerator so the AMP decorators are device-agnostic.
_amp_device_type = str(torch.accelerator.current_accelerator()) if hasattr(torch, "accelerator") else "cuda"

if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type=_amp_device_type)
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type=_amp_device_type)
