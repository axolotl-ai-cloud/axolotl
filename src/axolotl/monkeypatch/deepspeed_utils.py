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


def patch_zero_gradient_accumulation_dtype():
    """Preserve requested FP32 ZeRO reduction and accumulation buffers."""
    from functools import wraps

    import torch
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    original = DeepSpeedZeroOptimizer.get_all_grad_tensors
    if getattr(original, "_axolotl_accumulation_dtype", False):
        return

    @wraps(original)
    def get_all_grad_tensors(self, tensor_list, dtype):
        gradients = original(self, tensor_list, dtype)
        # DeepSpeed 0.18.9 only applies dtype to missing gradients in this method.
        if dtype == torch.float32:
            return [gradient.to(dtype=dtype) for gradient in gradients]
        return gradients

    get_all_grad_tensors._axolotl_accumulation_dtype = True
    DeepSpeedZeroOptimizer.get_all_grad_tensors = get_all_grad_tensors

    def fp32_reduction(self):
        return (
            self.gradient_accumulation_dtype == torch.float32
            and self.communication_data_type == torch.float32
        )

    def promote_gradient(gradient):
        if gradient is not None and gradient.dtype != torch.float32:
            gradient.data = gradient.data.float()
        return gradient

    original_gradient = DeepSpeedZeroOptimizer.get_gradient_for_reduction

    @wraps(original_gradient)
    def get_gradient_for_reduction(self, parameter):
        if not fp32_reduction(self):
            return original_gradient(self, parameter)
        gradient = (
            parameter.grad_accum if self.use_grad_accum_attribute else parameter.grad
        )
        return promote_gradient(gradient)

    DeepSpeedZeroOptimizer.get_gradient_for_reduction = get_gradient_for_reduction

    def fp32_allocation_scope(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            if not fp32_reduction(self):
                return method(self, *args, **kwargs)
            original_dtype = self.dtype
            try:
                self.dtype = torch.float32
                return method(self, *args, **kwargs)
            finally:
                self.dtype = original_dtype

        return wrapped

    # ZeRO allocates reduction/partition buffers in the working parameter dtype.
    for method_name in ("setup_buckets", "copy_grads_in_partition"):
        setattr(
            DeepSpeedZeroOptimizer,
            method_name,
            fp32_allocation_scope(getattr(DeepSpeedZeroOptimizer, method_name)),
        )

    from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3

    def fp32_scatter_scope(method, contiguous):
        @wraps(method)
        def wrapped(self, values, communication_data_type):
            if not fp32_reduction(self) or communication_data_type != torch.float32:
                return method(self, values, communication_data_type)
            parameters = (
                self.ipg_buckets[communication_data_type].params
                if contiguous
                else values
            )
            for parameter in parameters:
                promote_gradient(parameter.grad)
            original_dtype = self.dtype
            try:
                self.dtype = torch.float32
                return method(self, values, communication_data_type)
            finally:
                self.dtype = original_dtype

        return wrapped

    for suffix, contiguous in (
        ("__avg_scatter_grads", False),
        ("__avg_scatter_contiguous_grads", True),
    ):
        method_name = "_DeepSpeedZeroOptimizer_Stage3" + suffix
        setattr(
            DeepSpeedZeroOptimizer_Stage3,
            method_name,
            fp32_scatter_scope(
                getattr(DeepSpeedZeroOptimizer_Stage3, method_name), contiguous
            ),
        )
