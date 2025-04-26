"""
Module for Axolotl trainer sequence parallelism mixin and training context manager
"""

import functools

import torch
import torch.distributed as dist
from datasets import Dataset
from torch import nn
from torch.utils.data import DistributedSampler, Sampler
from torch.utils.hooks import RemovableHandle
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
    update_ring_attn_params,
)
from axolotl.utils.schemas.enums import RingAttnFunc


def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,
    local_world_size: int,
    ring_attn_func: RingAttnFunc,
) -> dict[str, torch.Tensor]:
    """
    Apply sequence parallelism slicing to a batch.

    Args:
        batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)
        local_rank: Local rank in the sequence parallel group
        local_world_size: World size of the sequence parallel group
        ring_attn_func: The ring attention function to use

    Returns:
        Sliced batch dictionary.
    """
    print("in apply_sequence_parallelism")

    # Update ring attention params if needed
    if batch.get("position_ids") is not None:
        update_ring_attn_params(position_ids=batch["position_ids"])

    # Slice batch for sequence parallel processing
    total_seq_len = batch["input_ids"].size(1)
    for key in batch:
        if (
            isinstance(batch[key], torch.Tensor)
            and batch[key].dim() > 1
            and batch[key].size(1) == total_seq_len
        ):
            if ring_attn_func in [
                RingAttnFunc.VARLEN_LLAMA3,
                RingAttnFunc.BATCH_RING,
            ]:
                if key == "input_ids":
                    print(f"{dist.get_rank()}: batch[key].shape before: {batch[key].shape}")
                # Split in sequential fashion and grab this rank's chunk
                batch[key] = (
                    batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
                )
                if key == "input_ids":
                    print(f"{dist.get_rank()}: batch[key].shape after: {batch[key].shape}")
            elif ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
                chunks = batch[key].chunk(2 * local_world_size, dim=1)

                # Take rank's chunk and opposing chunk for zigzag pattern
                selected_chunks = [
                    chunks[local_rank],
                    chunks[2 * local_world_size - local_rank - 1],
                ]
                batch[key] = torch.cat(selected_chunks, dim=1).contiguous()
            elif ring_attn_func is RingAttnFunc.BATCH_STRIPE:
                # Split into striped data and stack
                tensor = torch.stack(
                    batch[key].split(local_world_size, dim=1),
                    dim=1,
                ).transpose(1, 2)
                batch[key] = tensor[:, local_rank].contiguous()

    return batch


class SequenceParallelMixin:
    """
    Mixin class for sequence parallelism support in trainers.

    This mixin provides functionality for handling sequence parallelism,
    specifically for creating appropriate data samplers.
    """

    args = None  # type: "AxolotlTrainingArguments"  # type: ignore[name-defined]

    def _setup_sequence_parallel(self):
        """Set up sequence parallelism environment."""
        self.ring_attn_group = get_ring_attn_group()

    def _create_sequence_parallel_sampler(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        is_eval: bool = False,
    ) -> DistributedSampler:
        """
        Helper method to create sampler for sequence parallelism (SP).

        We create a distributed sampler with rank equal to the SP group ID, which
        means that all ranks in the SP group receive the same sample / set of samples
        per training step. We also set the number of replicas equal to the number of
        SP groups, which is a bit of a hack / unintended use, but works!

        Args:
            dataset: Dataset to sample from.
            shuffle: Whether to shuffle the dataset.
            is_eval: Whether we are creating a sampler for evaluation or training.

        Returns:
            Distributed sampler.
        """
        num_sp_groups = self.args.world_size // self.args.sequence_parallel_degree
        sp_group_id = dist.get_rank() // self.args.sequence_parallel_degree

        return DistributedSampler(
            dataset,
            num_replicas=num_sp_groups,
            rank=sp_group_id,
            seed=self.args.seed if shuffle else None,
            shuffle=shuffle,
            drop_last=not is_eval,
        )

    def _sp_get_train_sampler(self, dataset) -> Sampler | None:
        """
        Get a training sampler configured for sequence parallelism.

        Args:
            dataset: The training dataset

        Returns:
            Configured sequence parallel sampler.
        """
        return self._create_sequence_parallel_sampler(
            dataset,
            shuffle=not self.args.curriculum_sampling,
        )

    def _sp_get_eval_sampler(self, eval_dataset) -> Sampler | None:
        """
        Get an evaluation sampler configured for sequence parallelism.

        Args:
            eval_dataset: The evaluation dataset.

        Returns:
            Configured sequence parallel sampler.
        """
        return self._create_sequence_parallel_sampler(
            eval_dataset, shuffle=False, is_eval=True
        )


class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, gathered_tensors, rank, world_size, ring_attn_func):
        ctx.rank = rank
        ctx.world_size = world_size
        ctx.ring_attn_func = ring_attn_func
        ctx.input_shape = input_tensor.shape
        ctx.save_for_backward(input_tensor)
        
        # Process based on ring_attn_func for forward pass
        if ring_attn_func in [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]:
            return torch.cat(gathered_tensors, dim=1)

        if ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
            reconstituted_tensors = [None] * (world_size * 2)
            for r, g_tensor in enumerate(gathered_tensors):
                chunk_size = g_tensor.size(1) // 2
                chunk1, chunk2 = g_tensor.split(chunk_size, dim=1)
                reconstituted_tensors[r] = chunk1
                reconstituted_tensors[world_size * 2 - r - 1] = chunk2

            return torch.cat(reconstituted_tensors, dim=1)

        else:  # RingAttnFunc.BATCH_STRIPE
            batch_size = input_tensor.size(0)
            hidden_dim = input_tensor.size(-1)
            total_seq_len = sum(t.size(1) for t in gathered_tensors)
            result = torch.zeros(
                batch_size, total_seq_len, hidden_dim,
                dtype=input_tensor.dtype, device=input_tensor.device,
            )
            for r, g_tensor in enumerate(gathered_tensors):
                seq_len = g_tensor.size(1)
                for i in range(seq_len):
                    pos = i * world_size + r
                    if pos < total_seq_len:
                        result[:, pos] = g_tensor[:, i]

            return result
    
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        rank = ctx.rank
        world_size = ctx.world_size
        ring_attn_func = ctx.ring_attn_func
        input_shape = ctx.input_shape
        
        # Extract gradients based on ring attention function
        if ring_attn_func in [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]:
            # Simple sharding - each rank gets its corresponding section
            seq_len = input_shape[1]
            start_idx = rank * seq_len
            end_idx = start_idx + seq_len
            grad_input = grad_output[:, start_idx:end_idx].contiguous()

            # Verify shape
            if grad_input.shape != input_shape:
                raise ValueError(
                    f"Gradient shape mismatch in AllGatherWithGrad.backward: "
                    f"got {grad_input.shape} but expected {input_shape}"
                )
            
        elif ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
            # For zigzag pattern, input_tensor has 2 chunks, need to gather both parts
            # Important: input_shape already includes both chunks
            full_seq_len = input_shape[1] 
            chunk_size = full_seq_len // 2
            
            # Get first chunk from position 'rank'
            first_start = rank * chunk_size
            first_end = first_start + chunk_size
            
            # Get second chunk from position 'world_size*2-rank-1'
            second_start = (world_size * 2 - rank - 1) * chunk_size
            second_end = second_start + chunk_size
            
            # Concatenate both chunks to match input shape
            first_chunk = grad_output[:, first_start:first_end]
            second_chunk = grad_output[:, second_start:second_end]
            grad_input = torch.cat([first_chunk, second_chunk], dim=1).contiguous()
            
            # Verify shape
            if grad_input.shape != input_shape:
                raise ValueError(
                    f"Gradient shape mismatch in zigzag backward: "
                    f"got {grad_input.shape} but expected {input_shape}"
                )

        else:  # RingAttnFunc.BATCH_STRIPE
            # For stripe pattern, need to grab every world_size-th position
            grad_input = torch.zeros_like(input_tensor)
            seq_len = input_shape[1]
            total_seq_len = grad_output.size(1)
            
            idx = 0
            for i in range(rank, total_seq_len, world_size):
                if idx < seq_len:
                    grad_input[:, idx] = grad_output[:, i]
                    idx += 1
            
            # Verify shape
            if grad_input.shape != input_shape:
                raise ValueError(f"Gradient shape mismatch in stripe backward: "
                                f"got {grad_input.shape} but expected {input_shape}")
        
        # Return gradients for each input (plus None for non-tensor inputs)
        return grad_input, None, None, None, None


class SequenceParallelContextManager:
    """
    Context manager for sequence parallelism operations.

    This class provides a context that will automatically apply sequence parallelism
    during model forward passes using a pre-forward hook, and gather outputs from
    across the sequence parallelism group using a post-forward hook.
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

        # Will store hook handles for removal
        self.hook_handles: list[RemovableHandle] = []

        # Create a partially applied version of the apply_sequence_parallelism function
        # with pre-configured params
        self.apply_sequence_parallelism = functools.partial(
            apply_sequence_parallelism,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
            ring_attn_func=self.ring_attn_func,
        )

    def __enter__(self):
        # Forward pre-hook to apply sequence parallelism
        def sequence_parallel_pre_hook(_, args, kwargs):
            # Apply sequence parallelism to kwargs
            kwargs = self.apply_sequence_parallelism(batch=kwargs)
            return args, kwargs

        # Forward post-hook to gather outputs
        def sequence_parallel_post_hook(_, __, output):
            # Gather the sharded outputs
            output = self.gather_outputs(output)
            return output

        # Register both hooks
        self.hook_handles.append(
            self.model.register_forward_pre_hook(
                sequence_parallel_pre_hook, with_kwargs=True
            )
        )
        self.hook_handles.append(
            self.model.register_forward_hook(sequence_parallel_post_hook)
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def gather_outputs(self, output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        """Gather sharded outputs from all ranks and reconstruct the full tensor."""
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                # Gather logits or other sequence-sharded tensors
                gathered_value = self.gather_tensor(value)
                output[key] = gathered_value
            else:
                gathered_value = value.clone()
                dist.all_reduce(
                    gathered_value, op=dist.ReduceOp.SUM, group=self.process_group
                )
                output[key] = gathered_value

        return output

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a sharded tensor from all ranks while preserving gradients."""
        # Prepare tensors for all_gather
        world_size = self.local_world_size
        requires_grad = tensor.requires_grad
        
        # Create list to store tensors from all ranks
        with torch.no_grad():
            gathered_tensors = [torch.zeros_like(tensor.detach()) for _ in range(world_size)]
            
            # All-gather operation
            dist.all_gather(gathered_tensors, tensor.detach(), group=self.process_group)
        
        # If doesn't require gradient, just process normally
        if not requires_grad:
            if self.ring_attn_func in [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]:
                return torch.cat(gathered_tensors, dim=1)
            
            if self.ring_attn_func is RingAttnFunc.BATCH_ZIGZAG:
                reconstituted_tensors = [None] * (world_size * 2)
                for rank, gathered_tensor in enumerate(gathered_tensors):
                    chunk_size = gathered_tensor.size(1) // 2
                    chunk1, chunk2 = gathered_tensor.split(chunk_size, dim=1)
                    reconstituted_tensors[rank] = chunk1
                    reconstituted_tensors[world_size * 2 - rank - 1] = chunk2

                return torch.cat(reconstituted_tensors, dim=1)
            
            # RingAttnFunc.BATCH_STRIPE
            batch_size = tensor.size(0)
            hidden_dim = tensor.size(-1)
            total_seq_len = 0
            for t in gathered_tensors:
                total_seq_len += t.size(1)
            result = torch.zeros(
                batch_size, total_seq_len, hidden_dim,
                dtype=tensor.dtype, device=tensor.device,
            )
            for rank, gathered_tensor in enumerate(gathered_tensors):
                seq_len = gathered_tensor.size(1)
                for i in range(seq_len):
                    pos = i * world_size + rank
                    if pos < total_seq_len:
                        result[:, pos] = gathered_tensor[:, i]

            return result
        
        # For tensors requiring gradients, use a custom autograd function
        rank = dist.get_rank(group=self.process_group)
        result = AllGatherWithGrad.apply(
            tensor, gathered_tensors, rank, world_size, self.ring_attn_func
        )
        
        return result
