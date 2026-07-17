"""
Trainer mixin for activation checkpointing w offloading
"""

import contextlib

import torch
from peft import PeftModel
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers import GradientCheckpointingLayer, Trainer
from trl.models.activation_offloading import (
    NoOpManager,
    OffloadActivations,
    get_act_offloading_ctx_manager,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _patch_trl_offload_current_stream() -> None:
    """Make TRL ``OffloadActivations`` sync its CPU<->GPU copies against the LIVE compute stream
    instead of the once-captured ``torch.cuda.default_stream()``.

    ``OffloadActivations`` caches ``self.s0 = torch.cuda.default_stream()`` in ``__init__`` and uses
    it for every ``s0.wait_stream(s1)`` / ``s0.wait_event`` / ``record_stream(s0)``. When the actual
    forward/backward compute runs on a NON-default stream — FSDP2, the checkpoint recompute, or any
    custom autograd Function whose backward launches kernels (e.g. the scattermoe NVFP4 MoE) — that
    sync targets the wrong stream, so a streamed-in activation is read before its H2D copy finishes:
    garbage saved tensors -> NaN/illegal-memory in the backward (silent, finite forward). Querying
    ``current_stream()`` at each use site fixes the ordering (xpu/npu already use ``current_stream``).
    """
    from trl.models.activation_offloading import OffloadActivations

    if getattr(OffloadActivations, "_axolotl_live_stream", False):
        return

    def _live_compute_stream(self):
        t = getattr(self, "accelerator_type", "cuda")
        if t == "xpu":
            return torch.xpu.current_stream()
        if t == "npu":
            import torch_npu  # noqa: F401

            return torch.npu.current_stream()
        return torch.cuda.current_stream()

    # property read returns the live stream; the no-op setter swallows __init__'s `self.s0 = ...`.
    OffloadActivations.s0 = property(_live_compute_stream, lambda self, _v: None)
    OffloadActivations._axolotl_live_stream = True
    LOG.info(
        "Patched TRL OffloadActivations to sync against the live compute stream "
        "(fixes streamed activation offloading under FSDP2 / custom-Function backward)"
    )


def _patch_trl_offload_compute_stream_clone() -> None:
    """Clone offset/non-contiguous saved tensors on the COMPUTE stream before TRL's pack hook.

    ``pack_tensor`` clones such tensors inside its ``torch.cuda.stream(s1)`` context, so the
    stash copy is transfer-stream-pool memory written on s1 but consumed by compute kernels with
    no ordering edge: the consumer can read the block before s1 executes the clone (stale bytes
    from a recycled block), and once the stash entry is dropped mid-backward the block returns to
    the s1 pool and the next H2D restore overwrites it while the compute-stream read is still
    pending. Cross-entropy's saved ``shift_labels`` (storage_offset=1 from the label shift) hits
    this every step: garbage targets -> device-side assert in nll_loss backward, corrupted CUDA
    context. Pre-cloning on the compute stream closes both edges — the write is stream-ordered
    with all consumers and the block never enters the transfer stream's allocator pool.
    """
    from trl.models.activation_offloading import OffloadActivations

    if getattr(OffloadActivations, "_axolotl_compute_stream_clone", False):
        return

    orig_init = OffloadActivations.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        pack = self.pack_hook

        def pack_contiguous(tensor):
            if (
                torch.is_tensor(tensor)
                and tensor.device.type in ("cuda", "xpu", "npu")
                and not isinstance(tensor, torch.nn.Parameter)
                and (not tensor.is_contiguous() or tensor.storage_offset() != 0)
            ):
                tensor = tensor.clone(memory_format=torch.contiguous_format)
            return pack(tensor)

        self.pack_hook = pack_contiguous

    OffloadActivations.__init__ = patched_init
    OffloadActivations._axolotl_compute_stream_clone = True
    LOG.info(
        "Patched TRL OffloadActivations to clone offset/non-contiguous saved tensors "
        "on the compute stream (fixes cross-stream use-after-free of the pack-time clone)"
    )


class ActivationOffloadingMixin(Trainer):
    """
    Trainer mixin class for activation checkpointing w offloading
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.activation_offloading:
            # "legacy" uses the previous synchronous implementation (no CUDA
            # streams), which keeps far fewer activations resident on-GPU than
            # the stream-overlapped path (True/"disk"); the streams path stashes
            # several activations to overlap copies, inflating peak reserved.
            use_streams = self.args.activation_offloading != "legacy"
            if use_streams:
                _patch_trl_offload_current_stream()
                _patch_trl_offload_compute_stream_clone()
            if self.args.activation_offloading == "hidden_states":
                from axolotl.monkeypatch.checkpoint_activation_offload import (
                    get_checkpoint_hidden_states_offloading_ctx_manager,
                )

                self.activation_offload_context = (
                    get_checkpoint_hidden_states_offloading_ctx_manager(
                        use_streams=use_streams
                    )
                )
            elif isinstance(self.model, PeftModel):
                self.activation_offload_context = get_lora_act_offloading_ctx_manager(
                    self.model, use_streams=use_streams
                )
            else:
                self.activation_offload_context = get_act_offloading_ctx_manager(
                    self.model, use_streams=use_streams
                )
        else:
            self.activation_offload_context = contextlib.nullcontext()

    def training_step(self, *args, **kwargs):
        with self.activation_offload_context:
            return super().training_step(*args, **kwargs)


def ac_wrap_hf_model(model: nn.Module, **kwargs):
    auto_wrap_policy = ModuleWrapPolicy(set((GradientCheckpointingLayer,)))
    apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy, **kwargs)


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
