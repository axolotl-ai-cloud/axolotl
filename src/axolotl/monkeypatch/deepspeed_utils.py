import torch
from typing import Any
import importlib.util
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_deepspeed_zero3_missing_attributes():
    """
    Patch DeepSpeed's parameter_offload module to properly initialize ds_grads_remaining.

    This addresses the issue where DeepSpeed expects ds_grads_remaining attribute
    but doesn't initialize it.

    References:
    - https://github.com/deepspeedai/DeepSpeed/issues/7203
    """

    LOG.info("Applying DeepSpeed ZeRO Stage 3 ds_grads_remaining patch")
    
    try:
        import deepspeed.runtime.zero.parameter_offload as param_offload

        original_register_module = param_offload.DeepSpeedZeRoOffload._register_deepspeed_module
        
        def patched_register_deepspeed_module(self, module, count=[0]):
            """
            Patched version that initializes ds_grads_remaining before DeepSpeed
            tries to use it in its hooks.
            """
            if not hasattr(module, 'ds_grads_remaining'):
                module.ds_grads_remaining = 0
                LOG.debug(f"Initialized ds_grads_remaining for {type(module).__name__}")
            return original_register_module(self, module, count)
        

        param_offload.DeepSpeedZeRoOffload._register_deepspeed_module = patched_register_deepspeed_module
        LOG.info("DeepSpeed ZeRO Stage 3 patch applied successfully to _register_deepspeed_module")
        
    except ImportError as e:
        LOG.warning(f"Could not import DeepSpeed parameter_offload module: {e}")



def apply_deepspeed_patches():
    """
    Apply all DeepSpeed-related patches
    """
    if importlib.util.find_spec("deepspeed") is not None:
        patch_deepspeed_zero3_missing_attributes()
    else:
        LOG.debug("DeepSpeed not available, skipping patches")
