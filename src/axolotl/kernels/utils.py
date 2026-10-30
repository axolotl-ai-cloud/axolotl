"""Utilities for `axolotl.kernels` submodules."""

import torch
from transformers.utils.import_utils import is_torch_xla_available

# Detect the actual accelerator so the AMP decorators are device-agnostic.
# torch.accelerator does not recognise XLA, so check that first.
if is_torch_xla_available():
    _amp_device_type = "xla"
else:
    _accelerator = torch.accelerator.current_accelerator()
    # torch.amp.custom_fwd/bwd expect a device type string ("cuda", "npu", ...);
    # str(device) would be the invalid device type "None" on builds without one.
    _amp_device_type = _accelerator.type if _accelerator is not None else "cuda"

torch_amp_custom_fwd = torch.amp.custom_fwd(device_type=_amp_device_type)
torch_amp_custom_bwd = torch.amp.custom_bwd(device_type=_amp_device_type)
