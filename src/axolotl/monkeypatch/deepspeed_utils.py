import torch
from typing import Any
import importlib.util
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_deepspeed_zero3_missing_attributes():
    """
    Patch for DeepSpeed ZeRO Stage 3 missing ds_grads_remaining attribute

    This addresses the issue where Linear modules (and potentially other modules)
    don't have the ds_grads_remaining attribute that DeepSpeed expects during
    backward pass hooks.

    References:
    - https://github.com/deepspeedai/DeepSpeed/issues/7203
    """

    LOG.debug("Applying DeepSpeed ZeRO Stage 3 missing attributes patch")
    # original_getattribute = torch.nn.Module.__getattribute__

    def patched_getattribute(self, name):
        if name == "ds_grads_remaining":
            if not hasattr(self, "_ds_grads_remaining"):
                object.__setattr__(self, "_ds_grads_remaining", 0)
                LOG.debug(f"Initialized ds_grads_remaining for {type(self).__name__}")
            return object.__getattribute__(self, "_ds_grads_remaining")

    def patched_setattr(self, name: str, value: Any) -> None:
        """
        Patched __setattr__ to handle ds_grads_remaining assignment
        """
        if name == "ds_grads_remaining":
            object.__setattr__(self, "_ds_grads_remaining", value)
        else:
            object.__setattr__(self, name, value)

    torch.nn.Module.__getattribute__ = patched_getattribute
    torch.nn.Module.__setattr__ = patched_setattr
    LOG.debug("DeepSpeed ZeRO Stage 3 patch applied successfully")


def apply_deepspeed_patches():
    """
    Apply all DeepSpeed-related patches
    """
    if importlib.util.find_spec("deepspeed") is not None:
        patch_deepspeed_zero3_missing_attributes()
    else:
        LOG.debug("DeepSpeed not available, skipping patches")
