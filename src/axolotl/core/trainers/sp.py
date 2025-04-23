"""Module for definition of sequence parallel context manager"""

import inspect
import logging

import torch
import torch.distributed as dist
from torch import nn
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
    during model forward passes using pre-forward hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        sequence_parallel_degree: int,
        ring_attn_func: RingAttnFunc,
    ):
        self.model = model
        self.sequence_parallel_degree = sequence_parallel_degree
        self.ring_attn_func = ring_attn_func
        self.process_group = get_ring_attn_group()

        # Initialize sequence parallel group details
        self.local_rank = dist.get_rank(self.process_group)
        self.local_world_size = dist.get_world_size(self.process_group)
        self.active = False

        # Will store hook handles for removal
        self.hook_handles: list[RemovableHandle] = []

    def __enter__(self):
        self.active = True

        # Define a pre-forward hook to apply sequence parallelism with kwargs support
        def sequence_parallel_pre_hook(module, args, kwargs):
            if not self.active or self.sequence_parallel_degree <= 1:
                return None

            # Apply sequence parallelism to kwargs
            if kwargs:
                transformed_kwargs = self.apply_sequence_parallelism(kwargs)
                return args, transformed_kwargs

            # If no kwargs but we have args, try to convert them to kwargs
            if args and not kwargs:
                try:
                    signature = inspect.signature(module.forward)
                    param_names = list(signature.parameters.keys())[1:]  # Skip 'self'

                    # Create kwargs from args
                    new_kwargs = {}
                    for i, arg in enumerate(args):
                        if i < len(param_names):
                            new_kwargs[param_names[i]] = arg
                        else:
                            # If we can't map all args, don't transform
                            return None

                    # Apply sequence parallelism to the new kwargs
                    transformed_kwargs = self.apply_sequence_parallelism(new_kwargs)

                    # Return empty args and the transformed kwargs
                    return (), transformed_kwargs
                except (ValueError, TypeError):
                    # If conversion fails, don't transform
                    return None

            # If no args and no kwargs, nothing to transform
            return None

        # Register the pre-forward hook on the model
        self.hook_handles.append(
            self.model.register_forward_pre_hook(
                sequence_parallel_pre_hook, with_kwargs=True
            )
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.active = False

        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def apply_sequence_parallelism(
        self, batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Apply sequence parallelism slicing to a batch.

        Args:
            batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)

        Returns:
            Sliced batch dictionary.
        """
        if self.sequence_parallel_degree <= 1 or not self.active:
            return batch

        # Update ring attention params if needed
        if batch.get("position_ids") is not None:
            update_ring_attn_params(position_ids=batch["position_ids"])

        # Slice batch for sequence parallel processing
        total_seq_len = batch["input_ids"].size(1)
        for key in batch:
            if (
                key in batch
                and isinstance(batch[key], torch.Tensor)
                and batch[key].dim() > 1
                and batch[key].size(1) == total_seq_len
            ):

                if self.ring_attn_func in [
                    RingAttnFunc.VARLEN_LLAMA3,
                    RingAttnFunc.BATCH_RING,
                ]:
                    # Split in sequential fashion and grab this rank's chunk
                    batch[key] = (
                        batch[key]
                        .chunk(self.local_world_size, dim=1)[self.local_rank]
                        .contiguous()
                    )
                elif self.ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
                    chunks = batch[key].chunk(2 * self.local_world_size, dim=1)

                    # Take rank's chunk and opposing chunk for zigzag pattern
                    selected_chunks = [
                        chunks[self.local_rank],
                        chunks[2 * self.local_world_size - self.local_rank - 1],
                    ]
                    batch[key] = torch.cat(selected_chunks, dim=1).contiguous()
                elif self.ring_attn_func is RingAttnFunc.BATCH_STRIPE:
                    # Split into striped data and stack
                    tensor = torch.stack(
                        batch[key].split(self.local_world_size, dim=1),
                        dim=1,
                    ).transpose(1, 2)
                    batch[key] = tensor[:, self.local_rank].contiguous()

        return batch
