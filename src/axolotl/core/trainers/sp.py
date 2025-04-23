import contextlib
import functools
import logging
from typing import Dict, List, Optional, Set

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from axolotl.monkeypatch.attention.ring_attn.patch import (
    RingAttnFunc,
    get_ring_attn_group,
    update_ring_attn_params,
)

logger = logging.getLogger(__name__)


class SequenceParallelContext:
    """
    Context manager for sequence parallelism operations.

    This class provides a context that will automatically apply sequence parallelism
    during model forward passes.
    """

    # Keep track of active contexts to support nested contexts
    _active_contexts = []

    def __init__(
        self,
        sequence_parallel_degree: int,
        ring_attn_func: RingAttnFunc,
    ):
        self.sequence_parallel_degree = sequence_parallel_degree
        self.ring_attn_func = ring_attn_func
        self.process_group = get_ring_attn_group()

        # Initialize sequence parallel group details
        self.local_rank = 0
        self.local_world_size = 1
        self.active = False

        # Will store original methods for restoration
        self._original_module_forward = None
        self._hooks: List[RemovableHandle] = []

        if self.sequence_parallel_degree > 1:
            if self.process_group is None:
                self.process_group = dist.group.WORLD

            self.local_rank = dist.get_rank(self.process_group)
            self.local_world_size = dist.get_world_size(self.process_group)

    def __enter__(self):
        self.active = True
        SequenceParallelContext._active_contexts.append(self)

        # Store the original forward method
        if self._original_module_forward is None:
            self._original_module_forward = nn.Module.forward

        # Replace nn.Module.forward with our sequence parallel version
        nn.Module.forward = self._make_sequence_parallel_forward(nn.Module.forward)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False

        # Only restore original forward if this is the last active context
        if (
            SequenceParallelContext._active_contexts
            and SequenceParallelContext._active_contexts[-1] == self
        ):
            SequenceParallelContext._active_contexts.pop()

            # Restore original forward method
            if self._original_module_forward is not None:
                nn.Module.forward = self._original_module_forward
                self._original_module_forward = None

        # Remove any hooks we added
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _make_sequence_parallel_forward(self, original_forward):
        """Create a wrapped forward method that applies sequence parallelism."""

        @functools.wraps(original_forward)
        def sequence_parallel_forward(module_self, *args, **kwargs):
            # Skip sequence parallelism if not active or degree is 1
            if not self.active or self.sequence_parallel_degree <= 1:
                return original_forward(module_self, *args, **kwargs)

            # Convert args to kwargs if needed
            if args:
                # Try to map positional args to kwargs based on the forward method signature
                import inspect

                try:
                    signature = inspect.signature(original_forward)
                    param_names = list(signature.parameters.keys())[1:]  # Skip 'self'
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            kwargs[param_names[i]] = arg
                        else:
                            # If we can't map all args, fall back to original forward
                            return original_forward(module_self, *args, **kwargs)
                except (ValueError, TypeError):
                    # If we can't get the signature, just use the original forward
                    return original_forward(module_self, *args, **kwargs)

            # Apply sequence parallelism to the inputs
            kwargs = self.apply_sequence_parallelism(kwargs)

            # Call the original forward with modified kwargs
            return original_forward(module_self, **kwargs)

        return sequence_parallel_forward

    def apply_sequence_parallelism(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply sequence parallelism slicing to a batch.

        Args:
            batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)

        Returns:
            Sliced batch dictionary.
        """
        if self.sequence_parallel_degree <= 1 or not self.active:
            return batch

        # Make a copy of the batch to avoid modifying the original
        new_batch = dict(batch)

        # Get total sequence length from input_ids or inputs_embeds
        if "input_ids" in new_batch:
            total_seq_len = new_batch["input_ids"].size(1)
        elif "inputs_embeds" in new_batch:
            total_seq_len = new_batch["inputs_embeds"].size(1)
        else:
            # If we can't determine sequence length, return the batch as is
            return new_batch

        # Update ring attention params if needed
        if new_batch.get("position_ids") is not None:
            update_ring_attn_params(position_ids=new_batch["position_ids"])

        # Slice batch for sequence parallel processing
        for key in new_batch:
            if (
                key in new_batch
                and isinstance(new_batch[key], torch.Tensor)
                and new_batch[key].dim() > 1
                and new_batch[key].size(1) == total_seq_len
            ):

                if self.ring_attn_func in [
                    RingAttnFunc.VARLEN_LLAMA3,
                    RingAttnFunc.BATCH_RING,
                ]:
                    new_batch[key] = (
                        new_batch[key]
                        .chunk(self.local_world_size, dim=1)[self.local_rank]
                        .contiguous()
                    )
                elif self.ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
                    chunks = new_batch[key].chunk(2 * self.local_world_size, dim=1)

                    # Take rank's chunk and opposing chunk for zigzag pattern
                    selected_chunks = [
                        chunks[self.local_rank],
                        chunks[2 * self.local_world_size - self.local_rank - 1],
                    ]
                    new_batch[key] = torch.cat(selected_chunks, dim=1).contiguous()
                elif self.ring_attn_func is RingAttnFunc.BATCH_STRIPE:
                    # Split into striped data and stack
                    tensor = torch.stack(
                        new_batch[key].split(self.local_world_size, dim=1),
                        dim=1,
                    ).transpose(1, 2)
                    new_batch[key] = tensor[:, self.local_rank].contiguous()

        return new_batch


@contextlib.contextmanager
def sequence_parallel(
    sequence_parallel_degree: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
    ring_attn_func: Optional[RingAttnFunc] = None,
    buffers: Optional[List[torch.Tensor]] = None,
    buffer_seq_dims: Optional[List[int]] = None,
    no_restore_buffers: Optional[Set[torch.Tensor]] = None,
):
    """
    Context manager for sequence parallelism.

    This context manager will apply sequence parallelism to model inputs
    for all forward passes within its scope.

    Args:
        sequence_parallel_degree: The degree of sequence parallelism.
        process_group: The process group to use for communication. Default is the world group.
        ring_attn_func: The ring attention function to use.
        buffers: Optional list of buffers to shard (e.g., input tensors, position embeddings).
        buffer_seq_dims: Sequence dimensions for each buffer to shard.
        no_restore_buffers: Optional set of buffers that don't need to be restored.

    Yields:
        The sequence parallel context.
    """
    buffers = [] if buffers is None else buffers
    buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
    no_restore_buffers = set() if no_restore_buffers is None else no_restore_buffers

    if len(buffers) != len(buffer_seq_dims):
        raise ValueError(
            "`buffer_seq_dims` must have the same number of elements as `buffers`."
        )

    # Save original buffer states
    original_buffers = []
    for buffer in buffers:
        if buffer in no_restore_buffers:
            original_buffers.append(None)
        else:
            original_buffers.append(buffer.clone())

    # Create context
    context = SequenceParallelContext(
        sequence_parallel_degree=sequence_parallel_degree,
        process_group=process_group,
        ring_attn_func=ring_attn_func,
    )

    # Apply sequence parallelism to buffers if provided
    if buffers and buffer_seq_dims:
        for i, (buffer, dim) in enumerate(zip(buffers, buffer_seq_dims)):
            if context.sequence_parallel_degree > 1:
                # Get local shard
                sharded_tensor = context.apply_sequence_parallelism(
                    {"tensor": buffer.unsqueeze(0)}
                )["tensor"].squeeze(0)

                # Resize and copy in-place
                buffer.resize_(sharded_tensor.shape)
                buffer.copy_(sharded_tensor)

    try:
        # Enter the context
        with context:
            yield context
    finally:
        # Restore original buffer states
        for buffer, original in zip(buffers, original_buffers):
            if original is not None:
                buffer.resize_(original.shape)
                buffer.copy_(original)


def enable_sequence_parallel_for_module(
    module: nn.Module,
    sequence_parallel_degree: int = 1,
    process_group: Optional[dist.ProcessGroup] = None,
    ring_attn_func: Optional[RingAttnFunc] = None,
):
    """
    Enable sequence parallelism for a specific module.

    This function wraps the module's forward method to automatically apply
    sequence parallelism without using a context manager.

    Args:
        module: The module to enable sequence parallelism for.
        sequence_parallel_degree: The degree of sequence parallelism.
        process_group: The process group to use for communication.
        ring_attn_func: The ring attention function to use.

    Returns:
        The module with sequence parallelism enabled.
    """
    # Create a context for this module
    context = SequenceParallelContext(
        sequence_parallel_degree=sequence_parallel_degree,
        process_group=process_group,
        ring_attn_func=ring_attn_func,
    )

    # Save the original forward method
    original_forward = module.forward

    @functools.wraps(original_forward)
    def sequence_parallel_forward(*args, **kwargs):
        # Activate the context
        context.active = True

        # Convert args to kwargs if needed
        if args:
            import inspect

            try:
                signature = inspect.signature(original_forward)
                param_names = list(signature.parameters.keys())
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = arg
                    else:
                        return original_forward(*args, **kwargs)
            except (ValueError, TypeError):
                return original_forward(*args, **kwargs)

        # Apply sequence parallelism to inputs
        kwargs = context.apply_sequence_parallelism(kwargs)

        # Call original forward with modified inputs
        result = original_forward(**kwargs)

        # Deactivate the context
        context.active = False

        return result

    # Replace the forward method
    module.forward = sequence_parallel_forward

    return module
