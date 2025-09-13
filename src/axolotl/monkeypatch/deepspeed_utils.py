import importlib
import importlib.util

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_checkpoint_wrapper_setattr():
    """
    Patch CheckpointWrapper to properly forward DeepSpeed attributes to wrapped modules.

    This fixes the issue where CheckpointWrapper doesn't forward ds_* attributes
    (like ds_grads_remaining) to the actual wrapped module, causing DeepSpeed
    ZeRO-3 to fail when gradient checkpointing is enabled.

    This issue occurs specifically with:
    - QLoRA + DeepSpeed ZeRO-3
    - gradient_checkpointing: true
    - activation_offloading: true

    References:
    - https://github.com/deepspeedai/DeepSpeed/issues/7203
    - https://github.com/deepspeedai/DeepSpeed/blob/38d1a9eb64c9e01e32eccc50b25ba18925287441/deepspeed/runtime/zero/parameter_offload.py#L424-L458
    - https://github.com/axolotl-ai-cloud/axolotl/pull/3102
    """

    try:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        # Check if already patched
        if hasattr(CheckpointWrapper, "_axolotl_setattr_patched"):
            LOG.debug("CheckpointWrapper already patched")
            return

        original_setattr = CheckpointWrapper.__setattr__

        def new_setattr(self, name: str, value) -> None:
            if name.startswith("ds_") and hasattr(self, "_checkpoint_wrapped_module"):
                setattr(self._checkpoint_wrapped_module, name, value)
                LOG.debug(
                    f"Forwarded {name} to wrapped module {type(self._checkpoint_wrapped_module).__name__}"
                )
            else:
                original_setattr(self, name, value)

        CheckpointWrapper.__setattr__ = new_setattr
        CheckpointWrapper._axolotl_setattr_patched = True

        LOG.info("CheckpointWrapper patched to forward DeepSpeed attributes")

    except ImportError as e:
        LOG.debug(f"CheckpointWrapper not available: {e}")
    except Exception as e:
        LOG.warning(f"Failed to patch CheckpointWrapper: {e}")


def apply_deepspeed_patches():
    """
    Apply DeepSpeed-related patches
    """
    if importlib.util.find_spec("deepspeed") is not None:
        patch_checkpoint_wrapper_setattr()
    else:
        LOG.debug("DeepSpeed not available, skipping patches")
