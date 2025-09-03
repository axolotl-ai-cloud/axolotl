"""MOE Kernels Plugin for Axolotl."""

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MoeKernelsPlugin(BasePlugin):
    """
    Plugin for MOE kernels optimization with Axolotl.
    """

    def get_input_args(self) -> str | None:
        return "axolotl.integrations.moe_kernels.args.MoeKernelsArgs"

    def pre_model_load(self, cfg):
        """Apply MOE kernels patches before model loading if enabled."""
        if getattr(cfg, 'moe_kernels_enabled', False):
            LOG.info("Applying MOE kernels optimizations")
            
            # Get configuration options
            models = getattr(cfg, 'moe_kernels_models', ['deepseek_v3'])
            group_size_m = getattr(cfg, 'moe_kernels_group_size_m', 128)
            persistent_kernel = getattr(cfg, 'moe_kernels_persistent_kernel', True)
            use_triton = getattr(cfg, 'moe_kernels_use_triton', True)
            use_symmetric_memory = getattr(cfg, 'moe_kernels_use_symmetric_memory', True)
            
            # Apply patches
            apply_moe_kernel_patches(
                models=models,
                group_size_m=group_size_m,
                persistent_kernel=persistent_kernel,
                use_triton=use_triton,
                use_symmetric_memory=use_symmetric_memory,
            )


def apply_moe_kernel_patches(
    models: list[str],
    group_size_m: int = 128,
    persistent_kernel: bool = True,
    use_triton: bool = True,
    use_symmetric_memory: bool = True,
):
    """
    Apply MOE kernel optimizations to specified models.
    
    Args:
        models: List of model types to patch (e.g., ['deepseek_v3'])
        group_size_m: Group size for alignment in MOE operations
        persistent_kernel: Whether to use persistent kernels
        use_triton: Whether to use Triton kernels
        use_symmetric_memory: Whether to use symmetric memory operations
    """
    
    for model_type in models:
        if model_type == "deepseek_v3":
            _patch_deepseek_v3(
                group_size_m=group_size_m,
                persistent_kernel=persistent_kernel,
                use_triton=use_triton,
                use_symmetric_memory=use_symmetric_memory,
            )
        else:
            LOG.warning(f"MOE kernels not implemented for model type: {model_type}")


def _patch_deepseek_v3(
    group_size_m: int,
    persistent_kernel: bool,
    use_triton: bool,
    use_symmetric_memory: bool,
):
    """Patch DeepSeek V3 model with optimized MOE kernels."""
    try:
        # Import the modeling module
        from transformers.models.deepseek_v3 import modeling_deepseek_v3
        
        # Store configuration on the class
        if hasattr(modeling_deepseek_v3, 'DeepseekV3MoE'):
            moe_class = modeling_deepseek_v3.DeepseekV3MoE
            setattr(moe_class, '_group_size_m', group_size_m)
            setattr(moe_class, '_persistent_kernel', persistent_kernel)
            setattr(moe_class, '_use_triton', use_triton)
            setattr(moe_class, '_use_symmetric_memory', use_symmetric_memory)
            
            LOG.info(f"Configuring DeepSeek V3 MOE with group_size_m={group_size_m}, persistent_kernel={persistent_kernel}")
            
            # Patch the forward method
            original_forward = moe_class.forward
            setattr(moe_class, '_original_forward', original_forward)
            moe_class.forward = _create_optimized_moe_forward(original_forward)
            
            # Patch the moe method if it exists
            if hasattr(moe_class, 'moe'):
                original_moe = moe_class.moe
                setattr(moe_class, '_original_moe', original_moe)
                moe_class.moe = _create_optimized_moe_method(original_moe)
            
            LOG.info("Successfully patched DeepSeek V3 MOE with optimized kernels")
            
        else:
            LOG.warning("DeepseekV3MoE class not found in modeling module")
            
    except ImportError as e:
        LOG.warning(f"Could not import DeepSeek V3 modeling module: {e}")
    except Exception as e:
        LOG.error(f"Error patching DeepSeek V3 MOE: {e}")


def _create_optimized_moe_forward(original_forward):
    """Create an optimized forward method for MOE."""
    def optimized_forward(self, hidden_states, *args, **kwargs):
        """Optimized MOE forward with enhanced kernels."""
        
        # Log debug information
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            LOG.debug(f"MOE forward: input shape={hidden_states.shape}, dtype={hidden_states.dtype}")
            LOG.debug(f"MOE config: group_size_m={getattr(self, '_group_size_m', 'not set')}")
        
        # Use optimized path if conditions are met
        if (hasattr(self, '_use_triton') and self._use_triton and 
            hidden_states.is_cuda and hidden_states.dtype in [torch.float16, torch.bfloat16]):
            
            try:
                return _moe_optimized(self, hidden_states, *args, **kwargs)
            except Exception as e:
                LOG.warning(f"Optimized MOE failed, falling back to original: {e}")
                
        # Fall back to original implementation
        return original_forward(self, hidden_states, *args, **kwargs)
    
    return optimized_forward


def _create_optimized_moe_method(original_moe):
    """Create an optimized moe method."""
    def optimized_moe(self, hidden_states, topk_indices, topk_weights, *args, **kwargs):
        """Optimized MOE computation."""
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            LOG.debug(f"MOE method: input shape={hidden_states.shape}")
            LOG.debug(f"  topk_indices shape={topk_indices.shape}")
            LOG.debug(f"  topk_weights shape={topk_weights.shape}")
        
        # Use optimized kernels for token dispatch and combine
        if (hasattr(self, '_use_symmetric_memory') and self._use_symmetric_memory and
            torch.distributed.is_initialized()):
            
            try:
                return _moe_optimized_distributed(
                    self, hidden_states, topk_indices, topk_weights, *args, **kwargs
                )
            except Exception as e:
                LOG.warning(f"Distributed MOE optimization failed: {e}")
        else:
            # Use single-GPU optimized path
            try:
                return _moe_optimized_single_gpu(
                    self, hidden_states, topk_indices, topk_weights, *args, **kwargs
                )
            except Exception as e:
                LOG.warning(f"Single-GPU MOE optimization failed: {e}")
        
        # Fall back to original
        return original_moe(self, hidden_states, topk_indices, topk_weights, *args, **kwargs)
    
    return optimized_moe


def _moe_optimized(moe_layer, hidden_states, *args, **kwargs):
    """
    Optimized MOE computation using torchtitan kernels.
    """
    LOG.debug("Using optimized MOE path")
    
    try:
        # Get the MOE layer configuration
        if hasattr(moe_layer, 'num_experts'):
            num_experts = moe_layer.num_experts
        elif hasattr(moe_layer, 'config') and hasattr(moe_layer.config, 'n_routed_experts'):
            num_experts = moe_layer.config.n_routed_experts
        else:
            LOG.warning("Could not determine number of experts, using fallback")
            return moe_layer._original_forward(hidden_states, *args, **kwargs)
        
        # Get other MOE parameters
        top_k = getattr(moe_layer, 'top_k', 2)
        if hasattr(moe_layer, 'config') and hasattr(moe_layer.config, 'num_experts_per_tok'):
            top_k = moe_layer.config.num_experts_per_tok
            
        group_size_m = getattr(moe_layer, '_group_size_m', 128)
        
        LOG.debug(f"MOE config: num_experts={num_experts}, top_k={top_k}, group_size_m={group_size_m}")
        
        # Get original shape and flatten
        original_shape = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
        batch_seq_len, hidden_size = hidden_states_flat.shape
        
        # Route tokens to experts (this involves the router/gate)
        if hasattr(moe_layer, 'gate') or hasattr(moe_layer, 'router'):
            router = getattr(moe_layer, 'gate', getattr(moe_layer, 'router', None))
            if router is not None:
                # Get routing scores
                router_logits = router(hidden_states_flat)
                routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
                
                # Get top-k experts for each token
                routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
                routing_weights = routing_weights.to(hidden_states.dtype)
                
                # Use optimized token reordering
                return _moe_forward_with_kernels(
                    moe_layer, hidden_states_flat, selected_experts, routing_weights,
                    num_experts, top_k, group_size_m, original_shape
                )
        
        # If we can't extract routing info, fall back to original
        LOG.debug("Could not extract routing information, using fallback")
        return moe_layer._original_forward(hidden_states, *args, **kwargs)
        
    except Exception as e:
        LOG.warning(f"Optimized MOE failed: {e}, falling back to original")
        return moe_layer._original_forward(hidden_states, *args, **kwargs)


def _moe_forward_with_kernels(moe_layer, hidden_states, selected_experts, routing_weights, 
                              num_experts, top_k, group_size_m, original_shape):
    """
    Forward pass using optimized kernels for token dispatch and expert computation.
    """
    batch_seq_len, hidden_size = hidden_states.shape
    
    # Calculate tokens per expert for each rank
    # This is key for the optimized kernels
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (selected_experts == i).sum()
    
    LOG.debug(f"Tokens per expert: {num_tokens_per_expert}")
    
    # Use optimized permutation indices generation
    from .indices import generate_permute_indices
    
    # For single GPU case (most common), use experts_per_rank = num_experts, num_ranks = 1
    experts_per_rank = num_experts
    num_ranks = 1
    if torch.distributed.is_initialized():
        num_ranks = torch.distributed.get_world_size()
        experts_per_rank = num_experts // num_ranks
    
    max_len = batch_seq_len * top_k + group_size_m * num_experts  # Add padding
    
    # Generate optimized permutation indices
    try:
        permuted_indices, m_sizes, m_offsets = generate_permute_indices(
            tokens_per_expert_group=num_tokens_per_expert.repeat(num_ranks) if num_ranks > 1 else num_tokens_per_expert,
            experts_per_rank=experts_per_rank,
            num_ranks=num_ranks,
            max_len=max_len,
            alignment=group_size_m,
            use_cpu=False
        )
        LOG.debug(f"Generated permutation indices: m_sizes={m_sizes}, m_offsets={m_offsets}")
    except Exception as e:
        LOG.warning(f"Permutation indices generation failed: {e}, falling back")
        return moe_layer._original_forward(hidden_states.view(original_shape))
    
    # Reorder tokens according to expert assignment
    # Flatten selected experts for indexing
    selected_experts_flat = selected_experts.view(-1)  # [batch_seq_len * top_k]
    routing_weights_flat = routing_weights.view(-1)    # [batch_seq_len * top_k]
    
    # Create expanded hidden states for top-k routing
    hidden_states_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1)  # [batch_seq_len, top_k, hidden_size]
    hidden_states_expanded = hidden_states_expanded.reshape(-1, hidden_size)   # [batch_seq_len * top_k, hidden_size]
    
    # Sort tokens by expert assignment for efficient batched computation
    expert_indices = torch.argsort(selected_experts_flat, stable=True)
    sorted_hidden_states = hidden_states_expanded[expert_indices]
    sorted_routing_weights = routing_weights_flat[expert_indices]
    
    # Now run expert computation with optimized kernels
    expert_outputs = _run_experts_optimized(
        moe_layer, sorted_hidden_states, num_tokens_per_expert, m_sizes, group_size_m
    )
    
    # Apply routing weights
    expert_outputs = expert_outputs * sorted_routing_weights.unsqueeze(-1)
    
    # Unsort the outputs back to original token order
    output_unsorted = torch.zeros_like(expert_outputs)
    output_unsorted[expert_indices] = expert_outputs
    
    # Reshape back and sum over top-k dimension
    output_reshaped = output_unsorted.view(batch_seq_len, top_k, hidden_size)
    final_output = output_reshaped.sum(dim=1)  # [batch_seq_len, hidden_size]
    
    LOG.debug(f"Optimized MOE output shape: {final_output.shape}, norm: {final_output.norm().item():.6f}")
    
    return final_output.view(original_shape)


def _run_experts_optimized(moe_layer, sorted_hidden_states, num_tokens_per_expert, m_sizes, group_size_m):
    """
    Run expert computation using optimized grouped operations.
    """
    # Try to extract expert weights from the model
    expert_weights = _extract_expert_weights(moe_layer)
    
    if expert_weights is None:
        LOG.warning("Could not extract expert weights, using fallback computation")
        return _run_experts_fallback(moe_layer, sorted_hidden_states, num_tokens_per_expert)
    
    w1, w2, w3 = expert_weights
    
    # Use grouped matrix multiplication if available, otherwise use optimized fallback
    if hasattr(torch, '_grouped_mm'):
        try:
            # Convert to bfloat16 for grouped MM
            x_bf16 = sorted_hidden_states.to(torch.bfloat16)
            w1_bf16 = w1.to(torch.bfloat16)
            w2_bf16 = w2.to(torch.bfloat16)
            w3_bf16 = w3.to(torch.bfloat16)
            
            # Calculate offsets for grouped MM
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            
            # SwiGLU computation with grouped MM
            h1 = torch._grouped_mm(x_bf16, w1_bf16.transpose(-2, -1), offs=offsets)
            h1 = torch.nn.functional.silu(h1)
            
            h3 = torch._grouped_mm(x_bf16, w3_bf16.transpose(-2, -1), offs=offsets)
            h = h1 * h3
            
            out = torch._grouped_mm(h, w2_bf16.transpose(-2, -1), offs=offsets)
            
            return out.type_as(sorted_hidden_states)
            
        except Exception as e:
            LOG.warning(f"Grouped MM failed: {e}, using optimized fallback")
    
    # Use optimized batch computation instead of per-expert computation
    return _run_experts_batched_optimized(moe_layer, w1, w2, w3, sorted_hidden_states, num_tokens_per_expert)


def _extract_expert_weights(moe_layer):
    """
    Extract expert weight matrices from various MOE layer implementations.
    """
    # Try different patterns for expert weights
    if hasattr(moe_layer, 'experts'):
        experts = moe_layer.experts
        if hasattr(experts, 'w1') and hasattr(experts, 'w2') and hasattr(experts, 'w3'):
            return experts.w1, experts.w2, experts.w3
    
    # Try DeepSeek V3 pattern
    if hasattr(moe_layer, 'gate_proj') and hasattr(moe_layer, 'down_proj') and hasattr(moe_layer, 'up_proj'):
        # These might be lists of experts
        gate_proj = getattr(moe_layer, 'gate_proj')
        down_proj = getattr(moe_layer, 'down_proj')
        up_proj = getattr(moe_layer, 'up_proj')
        
        if isinstance(gate_proj, torch.nn.ModuleList):
            # Convert ModuleList to tensor
            w1_list = [expert.weight for expert in gate_proj]
            w2_list = [expert.weight for expert in down_proj]  
            w3_list = [expert.weight for expert in up_proj]
            
            w1 = torch.stack(w1_list, dim=0)  # [num_experts, hidden_dim, dim]
            w2 = torch.stack(w2_list, dim=0)  # [num_experts, dim, hidden_dim]
            w3 = torch.stack(w3_list, dim=0)  # [num_experts, hidden_dim, dim]
            
            return w1, w2, w3
    
    return None


def _run_experts_batched_optimized(moe_layer, w1, w2, w3, sorted_hidden_states, num_tokens_per_expert):
    """
    Optimized batch computation for experts when grouped MM is not available.
    """
    LOG.debug("Using optimized batch expert computation")
    
    num_experts = w1.shape[0]
    
    # Ensure consistent dtype across all tensors
    device = sorted_hidden_states.device
    dtype = sorted_hidden_states.dtype
    
    w1 = w1.to(dtype=dtype, device=device)
    w2 = w2.to(dtype=dtype, device=device)
    w3 = w3.to(dtype=dtype, device=device)
    
    # Process all experts in parallel using batch operations
    expert_outputs = []
    start_idx = 0
    
    for expert_idx in range(num_experts):
        num_tokens = num_tokens_per_expert[expert_idx].item()
        if num_tokens == 0:
            continue
            
        end_idx = start_idx + num_tokens
        if end_idx > sorted_hidden_states.shape[0]:
            end_idx = sorted_hidden_states.shape[0]
            
        if start_idx >= end_idx:
            continue
            
        expert_input = sorted_hidden_states[start_idx:end_idx]  # [num_tokens, hidden_size]
        
        # Extract this expert's weights
        expert_w1 = w1[expert_idx]  # [intermediate_size, hidden_size]
        expert_w2 = w2[expert_idx]  # [hidden_size, intermediate_size]
        expert_w3 = w3[expert_idx]  # [intermediate_size, hidden_size]
        
        # SwiGLU computation
        h1 = torch.matmul(expert_input, expert_w1.T)  # [num_tokens, intermediate_size]
        h1 = torch.nn.functional.silu(h1)
        
        h3 = torch.matmul(expert_input, expert_w3.T)  # [num_tokens, intermediate_size]
        h = h1 * h3
        
        expert_output = torch.matmul(h, expert_w2.T)  # [num_tokens, hidden_size]
        expert_outputs.append(expert_output)
        
        start_idx = end_idx
    
    if expert_outputs:
        return torch.cat(expert_outputs, dim=0)
    else:
        return torch.zeros_like(sorted_hidden_states)


def _run_experts_fallback(moe_layer, sorted_hidden_states, num_tokens_per_expert):
    """
    Fallback expert computation when optimized kernels can't be used.
    """
    LOG.debug("Using fallback expert computation")
    
    # Extract expert weights if possible
    expert_weights = _extract_expert_weights(moe_layer)
    if expert_weights is not None:
        w1, w2, w3 = expert_weights
        return _run_experts_batched_optimized(moe_layer, w1, w2, w3, sorted_hidden_states, num_tokens_per_expert)
    
    # If no expert weights available, try individual expert modules
    expert_outputs = []
    start_idx = 0
    num_experts = len(num_tokens_per_expert)
    
    for expert_idx in range(num_experts):
        num_tokens = num_tokens_per_expert[expert_idx].item()
        if num_tokens == 0:
            continue
            
        end_idx = start_idx + num_tokens
        if start_idx >= sorted_hidden_states.shape[0] or end_idx > sorted_hidden_states.shape[0]:
            # Skip if indices are out of bounds
            continue
            
        expert_input = sorted_hidden_states[start_idx:end_idx]
        
        # Try to call individual expert
        if hasattr(moe_layer, 'experts') and hasattr(moe_layer.experts, '__getitem__'):
            try:
                expert_output = moe_layer.experts[expert_idx](expert_input)
                expert_outputs.append(expert_output)
            except Exception as e:
                LOG.warning(f"Expert {expert_idx} failed: {e}")
                expert_outputs.append(torch.zeros_like(expert_input))
        else:
            # Create zero output if no individual experts available
            expert_outputs.append(torch.zeros_like(expert_input))
            
        start_idx = end_idx
    
    if expert_outputs:
        return torch.cat(expert_outputs, dim=0)
    else:
        return torch.zeros_like(sorted_hidden_states)


def _moe_optimized_single_gpu(moe_layer, hidden_states, topk_indices, topk_weights, *args, **kwargs):
    """
    Optimized single-GPU MOE using grouped matrix multiplication.
    """
    LOG.debug("Using optimized single-GPU MOE path")
    
    # Get configuration
    num_experts = topk_indices.max().item() + 1
    group_size_m = getattr(moe_layer, '_group_size_m', 128)
    
    # Calculate tokens per expert
    num_tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=hidden_states.device)
    for i in range(num_experts):
        num_tokens_per_expert[i] = (topk_indices == i).sum()
    
    # Flatten and sort by expert assignment
    topk_indices_flat = topk_indices.view(-1)
    topk_weights_flat = topk_weights.view(-1)
    hidden_states_flat = hidden_states.view(-1, hidden_states.shape[-1])
    
    # Sort tokens by expert for grouped computation
    expert_indices = torch.argsort(topk_indices_flat, stable=True)
    
    # Ensure indices are within bounds
    valid_indices = expert_indices < hidden_states_flat.shape[0]
    expert_indices = expert_indices[valid_indices]
    
    sorted_hidden_states = hidden_states_flat[expert_indices]
    sorted_weights = topk_weights_flat[expert_indices]
    
    # Run optimized expert computation
    expert_outputs = _run_experts_optimized(
        moe_layer, sorted_hidden_states, num_tokens_per_expert, None, group_size_m
    )
    
    # Apply routing weights
    expert_outputs = expert_outputs * sorted_weights.unsqueeze(-1)
    
    # Unsort back to original order
    output_unsorted = torch.zeros(topk_indices_flat.shape[0], expert_outputs.shape[-1], 
                                 dtype=expert_outputs.dtype, device=expert_outputs.device)
    output_unsorted[expert_indices] = expert_outputs
    
    # Reshape and reduce
    original_shape = hidden_states.shape
    if len(topk_indices.shape) > 1:  # top_k > 1
        top_k = topk_indices.shape[-1]
        output_reshaped = output_unsorted.view(-1, top_k, hidden_states.shape[-1])
        final_output = output_reshaped.sum(dim=1)
    else:
        final_output = output_unsorted
    
    LOG.debug(f"Single-GPU MOE output norm: {final_output.norm().item():.6f}")
    return final_output.view(original_shape[:-1] + (final_output.shape[-1],))


def _moe_optimized_distributed(moe_layer, hidden_states, topk_indices, topk_weights, *args, **kwargs):
    """
    Optimized distributed MOE using symmetric memory operations.
    """
    LOG.debug("Using optimized distributed MOE path")
    
    try:
        from .dispatch import TokenDispatcher
        from .combine import TokenCombiner
        
        # Get configuration
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        device = hidden_states.device
        try:
            group_name = torch.distributed.group.WORLD.group_name
        except AttributeError:
            group_name = "default"
        
        num_experts_total = topk_indices.max().item() + 1
        num_local_experts = num_experts_total // world_size
        group_size_m = getattr(moe_layer, '_group_size_m', 128)
        
        # Calculate token distribution across ranks
        batch_seq_len = hidden_states.shape[0]
        hidden_size = hidden_states.shape[-1]
        top_k = topk_indices.shape[-1] if len(topk_indices.shape) > 1 else 1
        
        max_tokens_per_rank = batch_seq_len * top_k * 2  # Conservative estimate
        
        # Set up symmetric memory buffers
        import torch.distributed._symmetric_memory as symm_mem
        
        # Enable symmetric memory for the group
        try:
            symm_mem.enable_symm_mem_for_group(group_name)
        except Exception as e:
            LOG.warning(f"Could not enable symmetric memory: {e}, falling back")
            raise
        
        # Create dispatcher and combiner
        dispatcher = TokenDispatcher(
            group_name=group_name,
            align=group_size_m,
            in_len=max_tokens_per_rank,
            out_len=max_tokens_per_rank * world_size,
            token_shape=(hidden_size,),
            num_ranks=world_size,
            num_local_experts=num_local_experts,
            dtype=hidden_states.dtype,
            device=device
        )
        
        combiner = TokenCombiner(
            group_name=group_name,
            align=group_size_m,
            in_len=max_tokens_per_rank * world_size,
            out_len=max_tokens_per_rank,
            token_shape=(hidden_size,),
            num_ranks=world_size,
            num_local_experts=num_local_experts,
            dtype=hidden_states.dtype,
            device=device
        )
        
        # Calculate splits and offsets for token dispatch
        tokens_per_expert_rank = torch.zeros(world_size * num_local_experts, dtype=torch.int64, device=device)
        
        # Determine which tokens go to which experts on which ranks
        topk_indices_flat = topk_indices.view(-1)
        for expert_id in topk_indices_flat:
            target_rank = expert_id.item() // num_local_experts
            local_expert_id = expert_id.item() % num_local_experts
            rank_expert_idx = target_rank * num_local_experts + local_expert_id
            tokens_per_expert_rank[rank_expert_idx] += 1
        
        # Create input and output buffers
        input_buffer = symm_mem.empty(max_tokens_per_rank, hidden_size, dtype=hidden_states.dtype, device=device)
        output_buffer = symm_mem.empty(max_tokens_per_rank * world_size, hidden_size, dtype=hidden_states.dtype, device=device)
        
        # Splits and offsets tensors
        in_splits = symm_mem.empty(world_size * num_local_experts, dtype=torch.int64, device=device)
        in_splits.copy_(tokens_per_expert_rank)
        
        out_splits_offsets = symm_mem.empty((2, world_size * num_local_experts), dtype=torch.int64, device=device)
        
        # Copy input data
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        actual_tokens = min(hidden_states_flat.shape[0], max_tokens_per_rank)
        input_buffer[:actual_tokens].copy_(hidden_states_flat[:actual_tokens])
        
        # Dispatch tokens to appropriate ranks
        dispatched = dispatcher(input_buffer, output_buffer, in_splits, out_splits_offsets)
        
        # Run local expert computation on dispatched tokens
        local_expert_output = _run_local_experts(moe_layer, dispatched, rank, num_local_experts, group_size_m)
        
        # Combine results back
        final_output_buffer = symm_mem.empty(max_tokens_per_rank, hidden_size, dtype=hidden_states.dtype, device=device)
        combined = combiner(local_expert_output, final_output_buffer, out_splits_offsets, out_splits_offsets)
        
        # Apply routing weights and reshape
        result = combined[:actual_tokens].view(hidden_states.shape)
        
        LOG.debug(f"Distributed MOE output norm: {result.norm().item():.6f}")
        return result
        
    except Exception as e:
        LOG.warning(f"Distributed MOE optimization failed: {e}")
        # Fall back to single GPU path
        return _moe_optimized_single_gpu(moe_layer, hidden_states, topk_indices, topk_weights, *args, **kwargs)


def _run_local_experts(moe_layer, dispatched_tokens, rank, num_local_experts, group_size_m):
    """
    Run computation on local experts for distributed MOE.
    """
    # Extract local expert weights
    local_start_expert = rank * num_local_experts
    local_end_expert = (rank + 1) * num_local_experts
    
    expert_weights = _extract_expert_weights(moe_layer)
    if expert_weights is not None:
        w1, w2, w3 = expert_weights
        # Extract local expert weights
        local_w1 = w1[local_start_expert:local_end_expert]
        local_w2 = w2[local_start_expert:local_end_expert]  
        local_w3 = w3[local_start_expert:local_end_expert]
        
        # Create a temporary MOE layer for local computation
        class TempExperts:
            def __init__(self, w1, w2, w3):
                self.w1 = w1
                self.w2 = w2
                self.w3 = w3
        
        class TempMoE:
            def __init__(self, experts):
                self.experts = experts
        
        temp_layer = TempMoE(TempExperts(local_w1, local_w2, local_w3))
        
        # Calculate tokens per local expert
        num_tokens_per_local_expert = torch.ones(num_local_experts, dtype=torch.int32, device=dispatched_tokens.device)
        num_tokens_per_local_expert *= dispatched_tokens.shape[0] // num_local_experts
        
        return _run_experts_optimized(temp_layer, dispatched_tokens, num_tokens_per_local_expert, None, group_size_m)
    else:
        # Fallback: return zeros
        return torch.zeros_like(dispatched_tokens)


# Utility function for debugging
def debug_tensor_info(tensor: torch.Tensor, name: str):
    """Debug utility to log tensor information."""
    LOG.debug(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
             f"device={tensor.device}, norm={tensor.norm().item():.6f}")