"""Utilities for `axolotl.kernels` submodules."""

import torch

# Detect the actual accelerator so the AMP decorators are device-agnostic.
_accelerator = torch.accelerator.current_accelerator()
# torch.amp.custom_fwd/bwd expect a device type string ("cuda", "npu", ...);
# str(device) would be the invalid device type "None" on builds without one.
_amp_device_type = _accelerator.type if _accelerator is not None else "cuda"

torch_amp_custom_fwd = torch.amp.custom_fwd(device_type=_amp_device_type)
torch_amp_custom_bwd = torch.amp.custom_bwd(device_type=_amp_device_type)
