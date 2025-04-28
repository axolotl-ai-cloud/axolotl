"""Module for Axolotl trainer sequence parallelism manager using DTensor"""

import functools
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor, Replicate
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group
from axolotl.monkeypatch.attention.ring_attn.patch import update_ring_attn_params
from axolotl.utils.schemas.enums import RingAttnFunc


def create_device_mesh(process_group: Optional[dist.ProcessGroup] = None) -> DeviceMesh:
    """
    Create a 1D device mesh for sequence parallelism from process group.
    
    Args:
        process_group: Process group for sequence parallelism
        
    Returns:
        DeviceMesh for distributing tensors
    """
    if process_group is None:
        process_group = get_ring_attn_group()
        
    world_size = dist.get_world_size(process_group)
    ranks = list(range(world_size))
    
    # Create 1D mesh with shape [world_size]
    mesh = DeviceMesh("cuda", ranks, mesh_dim_names=["sp"])
    return mesh


def apply_sequence_parallelism_dtensor(
    batch: dict[str, Union[torch.Tensor, DTensor]],
    device_mesh: DeviceMesh,
    local_rank: int,
    ring_attn_func: RingAttnFunc,
) -> dict[str, Union[torch.Tensor, DTensor]]:
    """
    Apply sequence parallelism using DTensor API.
    
    Args:
        batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)
        device_mesh: Device mesh for distribution
        local_rank: Local rank in the sequence parallel group
        ring_attn_func: The ring attention function to use
        
    Returns:
        Batch with distributed tensors
    """
    # Update ring attention params if needed
    if batch.get("position_ids") is not None:
        update_ring_attn_params(position_ids=batch["position_ids"])

    # Get sequence length for sharding
    total_seq_len = batch["input_ids"].size(1)
    world_size = device_mesh.size(0)

    if dist.get_rank() == 0:
        import ipdb; ipdb.set_trace()
    dist.barrier()

    if ring_attn_func in [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]:
        for key in batch:
            if (
                isinstance(batch[key], torch.Tensor)
                and batch[key].dim() > 1
                and batch[key].size(1) == total_seq_len
            ):
                # First, manually shard the tensor to match existing implementation
                # This is needed because we need to start with local tensors matching
                # what the original implementation would have
                sharded_tensor = batch[key].chunk(world_size, dim=1)[local_rank].contiguous()
                
                # Then create DTensor from this local tensor
                batch[key] = DTensor.from_local(
                    sharded_tensor, 
                    device_mesh, 
                    [Shard(1)],  # Shard on sequence dimension
                )
    else:
        valid_ring_attn_funcs = [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]
        raise NotImplementedError(
            f"ring_attn_func {ring_attn_func} must be in {valid_ring_attn_funcs}"
        )
    
    return batch


def gather_output_dtensor(
    output: CausalLMOutputWithPast, 
    device_mesh: DeviceMesh,
    ring_attn_func: RingAttnFunc
) -> CausalLMOutputWithPast:
    """
    Gather distributed output tensors across sequence parallel group.
    
    Args:
        output: Model output
        device_mesh: Device mesh for distribution
        ring_attn_func: Ring attention function used
        
    Returns:
        Output with gathered tensors
    """
    # Process each value in the output dict
    for key, value in output.items():
        if isinstance(value, DTensor):
            # Convert DTensor to full tensor
            if ring_attn_func in [RingAttnFunc.VARLEN_LLAMA3, RingAttnFunc.BATCH_RING]:
                # Simple concatenation for basic patterns
                output[key] = value.full_tensor()
            else:
                # For complex patterns, we may need custom handling
                # This is a placeholder for custom tensor reconstruction
                output[key] = value.full_tensor()
                
        elif isinstance(value, torch.Tensor) and value.dim() <= 1:
            # For scalar tensors, ensure they're consistent across ranks
            gathered_value = value.clone()
            dist.all_reduce(
                gathered_value, 
                op=dist.ReduceOp.SUM, 
                group=device_mesh.get_process_group()
            )
            output[key] = gathered_value

    return output


class DTensorSequenceParallelContextManager:
    """
    Context manager for sequence parallelism operations using DTensor.

    This class provides a context that will automatically apply sequence parallelism
    during model forward passes using DTensor for sharding and gathering tensors
    across the sequence parallelism group.
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
        
        # Create device mesh for DTensor operations
        self.device_mesh = create_device_mesh(self.process_group)
        
        # Will store hook handles for removal
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        
    def __enter__(self):
        self._convert_model_params_to_dtensor()

        # Forward pre-hook to apply sequence parallelism
        def sequence_parallel_pre_hook(_, args, kwargs):
            # Apply sequence parallelism to kwargs using DTensor
            kwargs = apply_sequence_parallelism_dtensor(
                batch=kwargs,
                device_mesh=self.device_mesh,
                local_rank=self.local_rank,
                ring_attn_func=self.ring_attn_func,
            )
            return args, kwargs
        
        # Forward post-hook to gather outputs
        def sequence_parallel_post_hook(_, __, output):
            # Gather the sharded outputs using DTensor
            output = gather_output_dtensor(
                output, 
                self.device_mesh,
                self.ring_attn_func
            )
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
    
    def _convert_model_params_to_dtensor(self):
        """Convert model parameters to DTensors to avoid mixed tensor types."""
        for _, param in self.model.named_parameters():
            # For embeddings and parameters that are accessed during attention computation,
            # we need to convert to DTensor with Replicate placement
            with torch.no_grad():
                # Create a DTensor with Replicate placement (all devices have full copy)
                param_dtensor = DTensor.from_local(
                    param, 
                    self.device_mesh, 
                    [Replicate()],
                )
                
                # Replace the parameter with its DTensor version
                # This approach preserves the parameter in the model's parameter list
                param.data = param_dtensor.to_local()
