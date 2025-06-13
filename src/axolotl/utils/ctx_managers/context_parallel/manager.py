"""Module for Axolotl trainer context parallelism manager and utilities."""

import functools
import inspect
from typing import Callable, Literal

import torch
import torch.distributed as dist
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from axolotl.monkeypatch.ring_attn import (
    get_ring_attn_group,
    patch_prepare_data_loader,
    patch_prepare_device_mesh,
    register_ring_attn,
)
from axolotl.utils.ctx_managers.context_parallel.utils import AllGatherWithGrad, apply_context_parallelism
from axolotl.utils.ctx_managers.utils import get_context_parallel_manager
from axolotl.utils.schemas.enums import RingAttnFunc


class ContextParallelContextManager:
    """Context manager for context parallelism operations.

    This class provides a context that will automatically apply context parallelism
    during model forward passes using a pre-forward hook, and gather outputs from
    across the context parallelism group using a post-forward hook.

    Args:
        models: List of models to apply context parallelism to pre- and post- forward
            hooks.
        backend: Which attention backend to use.
        context_parallel_degree: Number of processes to split sequences over.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        ring_attn_func: Which ring attention function to use. Currently unused.
        heads_k_stride: Context parallelism K head stride size. Passed through to
            `varlen_llama3` `ring_flash_attn` implementation.
    """

    def __init__(
        self,
        models: list[PreTrainedModel],
        backend: Literal["sdp_attention", "flash_attention"],
        context_parallel_degree: int,
        gradient_accumulation_steps: int,
        ring_attn_func: RingAttnFunc,
        heads_k_stride: int | None,
    ):
        self.models = models
        self.backend = backend
        self.context_parallel_degree = context_parallel_degree
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ring_attn_func = ring_attn_func
        self.heads_k_stride = heads_k_stride
        self._register_ring_attn()

        # Set distributed info for local rank
        self.process_group = get_ring_attn_group()
        self.local_rank = dist.get_rank(self.process_group)
        self.local_world_size = dist.get_world_size(self.process_group)

        # Will store hook handles for removal
        self.hook_handles: list[RemovableHandle] = []

        # Store original sequence length and padding information
        self.original_seq_len = 0
        self.pad_len = 0

        # Create a partially applied version of the apply_context_parallelism function
        self.apply_context_parallelism = functools.partial(
            apply_context_parallelism,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            ring_attn_func=self.ring_attn_func,
        )

        # SPDA CP initialization
        world_size = dist.get_world_size()
        mesh_shape = (
            world_size // self.context_parallel_degree,
            self.context_parallel_degree,
        )
        world_mesh = dist.DeviceMesh(
            "cuda",
            torch.tensor(list(range(world_size))).reshape(mesh_shape),
            mesh_dim_names=("dp", "cp"),
        )
        self.context_parallel_managers = []
        for model in models:
            ctx_manager = get_context_parallel_manager(
                enabled=self.context_parallel_degree > 1,
                world_mesh=world_mesh,
                model=model,
            )
            self.context_parallel_managers.append(ctx_manager)

    def __enter__(self):
        self._register_model_hooks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # TODO(djsaunde): Un-patch attention and accelerate functions (low priority)

    def _register_ring_attn(self):
        if self.backend == "flash_attention":
            # Initialize ring attn for context parallelism
            register_ring_attn(
                context_parallel_degree=self.context_parallel_degree,
                heads_k_stride=self.heads_k_stride,
                ring_attn_func=self.ring_attn_func,
            )

        # Patches for accelerate functionality
        patch_prepare_data_loader()
        patch_prepare_device_mesh(context_parallel_degree=self.context_parallel_degree)

    def _register_model_hooks(self):
        # Forward pre-hook to apply context parallelism
        def cp_flash_pre_hook(_, args, kwargs):
            # Get parameter names from the model's forward function
            forward_params = list(
                inspect.signature(self.models[0].forward).parameters.keys()
            )

            updated_kwargs = kwargs.copy()
            for i, arg in enumerate(args):
                if i < len(forward_params):
                    updated_kwargs[forward_params[i]] = arg

            # Any excess positional arguments are kept as-is
            remaining_args = args[len(forward_params) :]

            # Apply context parallelism to updated kwargs
            updated_kwargs, self.original_seq_len, self.pad_len = (
                self.apply_context_parallelism(updated_kwargs)
            )

            return remaining_args, updated_kwargs

        # Forward post-hook to gather outputs
        def cp_flash_post_hook(_, __, output: ModelOutput) -> ModelOutput:
            # Gather the sharded outputs
            output = self._gather_outputs(output)

            # Remove padding if it was added
            if self.pad_len > 0:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        if value.size(1) == self.original_seq_len + self.pad_len:
                            # Slice to remove padding
                            output[key] = value[:, : self.original_seq_len].contiguous()

            return output

        # Register both hooks
        for i, model in enumerate(self.models):
            if self.backend == "flash_attention":
                self.hook_handles.append(
                    model.register_forward_pre_hook(cp_flash_pre_hook, with_kwargs=True)
                )
                self.hook_handles.append(
                    model.register_forward_hook(cp_flash_post_hook)
                )
            else:

                def make_sdpa_pre_hook(model_idx: int) -> Callable:
                    def cp_sdpa_pre_hook(_, args, kwargs):
                        with self.context_parallel_managers[model_idx]:
                            return args, kwargs

                    return cp_sdpa_pre_hook

                self.hook_handles.append(
                    model.register_forward_pre_hook(
                        make_sdpa_pre_hook(i), with_kwargs=True
                    )
                )

    def _gather_outputs(self, output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        """Gather sharded outputs from all ranks and reconstruct the full tensor."""
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                output[key] = AllGatherWithGrad.apply(value, self.process_group)

        return output

