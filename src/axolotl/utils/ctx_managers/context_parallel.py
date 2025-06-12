"""Module for Axolotl trainer sequence parallelism manager and utilities"""

import functools
import inspect
from typing import Literal

import torch
import torch.distributed as dist
from torch.distributed.tensor.experimental import context_parallel
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
from axolotl.utils.schemas.enums import RingAttnFunc
from axolotl.utils.ctx_managers.utils import get_context_parallel_manager


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

        # Create a partially applied version of the apply_sequence_parallelism function
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
                sequence_parallel_degree=self.context_parallel_degree,
                heads_k_stride=self.heads_k_stride,
                ring_attn_func=self.ring_attn_func,
            )
        else:
            stack.enter_context(context_parallel(mesh=mesh))

        # Patches for accelerate functionality
        patch_prepare_data_loader()
        patch_prepare_device_mesh(
            sequence_parallel_degree=self.context_parallel_degree
        )

    def _register_model_hooks(self):
        # Forward pre-hook to apply sequence parallelism
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

            # Apply sequence parallelism to updated kwargs
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

        def cp_sdpa_pre_hook(_, args, kwargs):
            with self.context_parallel_managers[?](list(inputs.values())):
                

        # Register both hooks
        for model in self.models:
            self.hook_handles.append(
                model.register_forward_pre_hook(
                    cp_flash_pre_hook, with_kwargs=True
                )
            )
            self.hook_handles.append(
                model.register_forward_hook(cp_flash_post_hook)
            )

    def _gather_outputs(self, output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        """Gather sharded outputs from all ranks and reconstruct the full tensor."""
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                output[key] = AllGatherWithGrad.apply(value, self.process_group)

        return output


class AllGatherWithGrad(torch.autograd.Function):
    """Custom autograd function for all-gather to preserve gradients."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Forward pass of all-gather of data with sequence dimension.

        Args:
            ctx: `torch.autograd` function context.
            input_tensor: Tensor from model output with sequence dimension.
            group: `torch.distributed` process group.

        Returns:
            Tensor from gathering the `input_tensor` from across the process group and
                concatenating along the sequence dimension.
        """
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)

        # Gather shape metadata
        local_shape = torch.tensor(list(input_tensor.shape), device=input_tensor.device)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape, group=group)

        # Store sequence lengths for backward pass
        seq_lens = [int(shape[1].item()) for shape in all_shapes]
        ctx.seq_lens = seq_lens

        # Perform all_gather operation
        gathered = [
            torch.zeros(
                tuple(shape.tolist()),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            for shape in all_shapes
        ]
        dist.all_gather(gathered, input_tensor, group=group)

        # Concatenate tensors along sequence dimension
        result = torch.cat(gathered, dim=1)

        return result

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        Backward pass for all-gather operation.

        Extracts the gradient slice corresponding to this rank's original input
        from the full gradient tensor.

        Args:
            ctx: `torch.autograd` function context.
            grad_output: Gradient from subsequent layers with respect to the
                concatenated output tensor.

        Returns:
            Tuple containing the gradient slice for this rank's input tensor and `None`
                for the process group parameter which doesn't require gradients.
        """
        rank = ctx.rank
        seq_lens = ctx.seq_lens

        # Extract gradient for this rank's chunk
        offset = sum(seq_lens[:rank])
        grad_slice = grad_output[:, offset : offset + seq_lens[rank]].contiguous()

        return grad_slice, None
