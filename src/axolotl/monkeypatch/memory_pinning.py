"""
Disable CPU memory pinning to allow FSDP to use disk swap if RAM becomes exhausted.
"""

import logging
import os

LOG = logging.getLogger(__name__)


def apply_memory_pinning_patch():
    """Apply patches to disable CPU memory pinning."""
    import torch

    # Check if already patched to avoid double-patching
    if hasattr(torch.Tensor, "_original_pin_memory"):
        return

    # 1) Patch FSDP CPUOffloadPolicy
    from torch.distributed.fsdp import CPUOffloadPolicy

    CPUOffloadPolicy.pin_memory = False

    # 2) Patch torch.Tensor.pin_memory to be a no-op
    def noop_pin_memory(self, device=None):
        return self

    torch.Tensor._original_pin_memory = torch.Tensor.pin_memory
    torch.Tensor.pin_memory = noop_pin_memory

    # Only log info once (main process or rank 0)
    if os.getenv("LOCAL_RANK", "0") == "0":
        LOG.info(
            "Memory pinning DISABLED - swap enabled for FSDP "
            "Training may be slower but can now use swap if RAM becomes exhausted"
        )
