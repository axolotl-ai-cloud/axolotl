"""
Disable CPU memory pinning to allow FSDP to use disk swap if RAM becomes exhausted.
"""

import logging
import os

LOG = logging.getLogger(__name__)


def patch_memory_pinning():
    """Apply patch to disable CPU memory pinning for FSDP v1."""
    import torch

    if hasattr(torch.Tensor, "_original_pin_memory"):
        return

    def noop_pin_memory(self, device=None):
        return self

    torch.Tensor._original_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = noop_pin_memory

    if os.getenv("LOCAL_RANK", "0") == "0":
        LOG.info("patching tensor pin_memory for FSDP v1 swap support")
