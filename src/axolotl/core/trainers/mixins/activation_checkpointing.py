"""
Trainer mixin for activation checkpointing w offloading
"""

import contextlib

from peft import PeftModel
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import Trainer
from trl.models.activation_offloading import (
    NoOpManager,
    OffloadActivations,
    get_act_offloading_ctx_manager,
)

from axolotl.monkeypatch.trainer.trl_offload import patch_trl_offload_dense_layout
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _storage_ptr(tensor):
    local = (
        getattr(tensor, "_local_tensor", tensor)
        if isinstance(tensor, DTensor)
        else tensor
    )
    try:
        return local.untyped_storage().data_ptr()
    except (RuntimeError, AttributeError):
        return None


def _install_live_fsdp_parameter_storage_hooks(context, model) -> None:
    if not hasattr(context, "param_storages"):
        return
    if getattr(context, "_axolotl_live_parameter_storage_hooks", False):
        return
    if not any(isinstance(parameter, DTensor) for parameter in model.parameters()):
        return

    # TRL captures storage before Accelerate applies FSDP2. Those pointers can be
    # freed and reused, so only retain storage observed while a parameter owner runs.
    context.param_storages.clear()
    active_storages: dict[int, int] = {}
    forward_storages: dict[int, list[set[int]]] = {}
    owners = {
        module
        for module in model.modules()
        if any(True for _ in module.parameters(recurse=False))
    }
    wrapper_owners = {
        module
        for module in model.modules()
        if hasattr(module, "base_layer")
        and (hasattr(module, "lora_A") or hasattr(module, "lora_embedding_A"))
    }
    owners.update(wrapper_owners)

    def enter(module, _args):
        pointers = {
            pointer
            for parameter in module.parameters(recurse=module in wrapper_owners)
            if (pointer := _storage_ptr(parameter)) not in (None, 0)
        }
        forward_storages.setdefault(id(module), []).append(pointers)
        for pointer in pointers:
            active_storages[pointer] = active_storages.get(pointer, 0) + 1
            context.param_storages.add(pointer)

    def exit(module, _args, _output):
        stacks = forward_storages.get(id(module))
        if not stacks:
            return
        for pointer in stacks.pop():
            remaining = active_storages[pointer] - 1
            if remaining:
                active_storages[pointer] = remaining
            else:
                del active_storages[pointer]
                context.param_storages.discard(pointer)

    handles = []
    for module in owners:
        handles.append(module.register_forward_pre_hook(enter))
        handles.append(module.register_forward_hook(exit, always_call=True))
    context._axolotl_live_parameter_storage_hooks = True
    context._axolotl_live_parameter_storage_hook_handles = handles


class ActivationOffloadingMixin(Trainer):
    """
    Trainer mixin class for activation checkpointing w offloading
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.activation_offloading:
            patch_trl_offload_dense_layout()
            if self.args.activation_offloading == "hidden_states":
                from axolotl.monkeypatch.checkpoint_activation_offload import (
                    get_checkpoint_hidden_states_offloading_ctx_manager,
                )

                self.activation_offload_context = (
                    get_checkpoint_hidden_states_offloading_ctx_manager()
                )
            elif isinstance(self.model, PeftModel):
                self.activation_offload_context = get_lora_act_offloading_ctx_manager(
                    self.model
                )
            else:
                self.activation_offload_context = get_act_offloading_ctx_manager(
                    self.model
                )
            if hasattr(self, "maybe_activation_offload_context"):
                # a TRL trainer's own offloader would nest inside this one
                self.maybe_activation_offload_context = contextlib.nullcontext()
        else:
            self.activation_offload_context = contextlib.nullcontext()

    def training_step(self, *args, **kwargs):
        _install_live_fsdp_parameter_storage_hooks(
            self.activation_offload_context, self.model
        )
        with self.activation_offload_context:
            return super().training_step(*args, **kwargs)


def get_lora_act_offloading_ctx_manager(
    model: nn.Module,
    use_pin_memory: bool = True,
    use_streams: bool = True,
    min_offload_size: int = 1024,
    max_fwd_stash_size: int = 5,
    warn_if_no_head: bool = True,
) -> OffloadActivations:
    """
    Returns the activation offloading context manager for the model. All but the last output Linear in every step will
    be offloaded.

    If activation offloading is enabled, we return the OffloadActivations context manager. If activation offloading is
    disabled, we return a NoOpManager context manager.

    Args:
        model (`nn.Module`):
            Model to wrap with the activation offloading context manager.
        use_pin_memory (`bool`, *optional*, defaults to `True`):
            Whether to offloaded Tensor will be placed in pinned memory on the CPU. Pinned memory allows the Tensor to
            be moved back onto GPU more quickly but is a limited resource.
        use_streams (`bool`, *optional*, defaults to `True`):
            Whether to use streams for performance optimization where the communications get overlapped with the
            computation. Requires a torch build after torch-2.5.0.
        min_offload_size (`int`, *optional*, defaults to `1024`):
            Minimum number of bytes a Tensor must be in order to qualify for offloading. If the tensor is too small, we
            do not want to waste bandwidth and resources moving it to CPU and back.
        max_fwd_stash_size (`int`, *optional*, defaults to `5`):
            Maximum size of the forward stash, or the maximum number of consecutive activations to keep alive during
            the forward pass. This number must be at least 1. Keeping alive more activations will potentially allow
            more overlap between the communication and compute streams at the cost of increasing memory usage. Keeping
            alive fewer activations will conserve memory, but may cause poor overlap between the streams, increasing
            runtime.
        warn_if_no_head (`bool`, *optional*, defaults to `True`):
            Whether to warn if no output head is detected. If set to `False`, no warning will be raised if no output
            head is detected.

    Returns:
        `contextlib.ContextDecorator`:
            Activation offloading context manager for the model.
    """

    activations_handling_ctx = OffloadActivations(
        use_pin_memory=use_pin_memory,
        use_streams=use_streams,
        min_offload_size=min_offload_size,
        max_fwd_stash_size=max_fwd_stash_size,
    )
    # Frozen base weights are saved as non-Parameter views (e.g. weight.t()).
    activations_handling_ctx.update_model_params(model)

    # Below is our hack to disable offloading the last output Linear in every
    # step, as the cost for offloading the activation and then soon after bringing
    # it back is expensive.
    output_head_detected = False
    noop_ctx = NoOpManager()

    # Try to get the actual model if it's wrapped
    unwrapped_model = model
    if hasattr(unwrapped_model, "module"):
        unwrapped_model = unwrapped_model.module
    # check for PEFT models
    if hasattr(unwrapped_model, "base_model") and hasattr(
        unwrapped_model, "peft_config"
    ):
        unwrapped_model = unwrapped_model.base_model

    # Check for different types of output heads
    if hasattr(unwrapped_model, "output"):
        if isinstance(unwrapped_model.output, nn.Module):
            unwrapped_model.output.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            unwrapped_model.output.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True
        elif hasattr(unwrapped_model.output, "linear") and isinstance(
            unwrapped_model.output.linear, nn.Module
        ):
            unwrapped_model.output.linear.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            unwrapped_model.output.linear.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True

    # Check for HuggingFace model output heads
    elif hasattr(unwrapped_model, "lm_head"):
        unwrapped_model.lm_head.register_forward_pre_hook(
            lambda *args: noop_ctx.__enter__()
        )
        unwrapped_model.lm_head.register_forward_hook(
            lambda *args: noop_ctx.__exit__(), always_call=True
        )
        output_head_detected = True

    # Check for decoder-based models
    elif hasattr(unwrapped_model, "decoder"):
        decoder = unwrapped_model.decoder
        if hasattr(decoder, "output"):
            decoder.output.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            decoder.output.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True
        # Some models have lm_head in the decoder
        elif hasattr(decoder, "lm_head"):
            decoder.lm_head.register_forward_pre_hook(
                lambda *args: noop_ctx.__enter__()
            )
            decoder.lm_head.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
            output_head_detected = True

    # Check for transformer models with final layer norm
    elif hasattr(unwrapped_model, "final_layer_norm") or hasattr(
        unwrapped_model, "ln_f"
    ):
        final_norm = (
            getattr(unwrapped_model, "final_layer_norm", None) or unwrapped_model.ln_f
        )
        final_norm.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
        final_norm.register_forward_hook(
            lambda *args: noop_ctx.__exit__(), always_call=True
        )
        output_head_detected = True

    # Check for models with head module
    elif hasattr(unwrapped_model, "head") and isinstance(
        unwrapped_model.head, nn.Module
    ):
        unwrapped_model.head.register_forward_pre_hook(
            lambda *args: noop_ctx.__enter__()
        )
        unwrapped_model.head.register_forward_hook(
            lambda *args: noop_ctx.__exit__(), always_call=True
        )
        output_head_detected = True

    if not output_head_detected and warn_if_no_head:
        LOG.warning(
            "During activation offloading, no output head was detected. If your model has an output head, it will be "
            "offloaded. This usually greatly slows training, given the large vocabulary size. To change this "
            "behavior, set your output head as model.output and make it an nn.Module. You can disable this warning by "
            "passing `warn_if_no_head=False`."
        )

    for name, module in unwrapped_model.named_modules():
        # Disable offloading for any Liger modules
        if "liger" in name.lower():
            module.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
            module.register_forward_hook(
                lambda *args: noop_ctx.__exit__(), always_call=True
            )
        # disable offloading for any submodules to fix LoRA training
        if name.endswith("._checkpoint_wrapped_module"):
            for _, sub_module in module.named_modules():
                sub_module.register_forward_pre_hook(lambda *args: noop_ctx.__enter__())
                sub_module.register_forward_hook(
                    lambda *args: noop_ctx.__exit__(), always_call=True
                )

    return activations_handling_ctx
