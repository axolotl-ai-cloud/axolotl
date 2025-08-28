"""Plugin to apply optimized MoE kernels to supported models"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from axolotl.integrations.base import BasePlugin
from axolotl.kernels.moe import cg_grouped_gemm_forward
from axolotl.utils.dict import DictDefault

logger = logging.getLogger(__name__)


def sort_tokens_by_expert(hidden_states, expert_indices, top_k):
    """
    Sort tokens by their assigned experts for contiguous memory access.

    Args:
        hidden_states: Input hidden states [batch_size * seq_len, hidden_dim]
        expert_indices: Expert assignment indices [batch_size * seq_len, top_k]
        top_k: Number of experts per token

    Returns:
        Sorted hidden states and permutation indices
    """
    # Flatten expert indices if needed
    if expert_indices.dim() > 1:
        flat_expert_indices = expert_indices.view(-1)
        hidden_states = hidden_states.repeat_interleave(top_k, dim=0)
    else:
        flat_expert_indices = expert_indices

    # Sort by expert index
    sorted_indices = torch.argsort(flat_expert_indices, stable=True)
    sorted_hidden_states = hidden_states[sorted_indices]

    # Create inverse permutation for restoring order
    inverse_indices = torch.empty_like(sorted_indices)
    inverse_indices[sorted_indices] = torch.arange(
        len(sorted_indices), device=sorted_indices.device
    )

    return (
        sorted_hidden_states,
        sorted_indices,
        inverse_indices,
        flat_expert_indices[sorted_indices],
    )


def patch_mixtral_moe_forward_optimized(
    group_size_m: int = 128, persistent_kernel: bool = True
) -> None:
    """
    Patch Mixtral MoE forward pass to use optimized kernels.
    """

    def moe_forward_optimized(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router logits and expert selection
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(
            routing_weights, self.top_k, dim=-1, sorted=False
        )
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(hidden_states.dtype)

        # Sort tokens by expert for contiguous memory access
        sorted_states, _, inverse_indices, expert_indices = sort_tokens_by_expert(
            hidden_states, topk_idx, self.top_k
        )

        # Prepare expert weights as a single tensor
        # Kernel expects [num_experts, N, K] where N=out_features, K=in_features
        # PyTorch weights are already [out_features, in_features], so just stack
        w1_weights = torch.stack([expert.w1.weight for expert in self.experts])
        w3_weights = torch.stack([expert.w3.weight for expert in self.experts])
        w2_weights = torch.stack([expert.w2.weight for expert in self.experts])

        # Sanity check dimensions
        expected_hidden_dim = sorted_states.shape[1]
        if w1_weights.shape[2] != expected_hidden_dim:
            raise ValueError(
                f"Weight dimension mismatch: w1 input dim {w1_weights.shape[2]} != hidden dim {expected_hidden_dim}"
            )
        if w2_weights.shape[1] != expected_hidden_dim:
            raise ValueError(
                f"Weight dimension mismatch: w2 output dim {w2_weights.shape[1]} != hidden dim {expected_hidden_dim}"
            )

        # First linear: w1 and w3 in parallel
        h1 = cg_grouped_gemm_forward(
            sorted_states, w1_weights, expert_indices, group_size_m, persistent_kernel
        )
        h3 = cg_grouped_gemm_forward(
            sorted_states, w3_weights, expert_indices, group_size_m, persistent_kernel
        )

        # Activation and element-wise multiplication
        current_states = self.act_fn(h1) * h3

        # Second linear: w2
        output_states = cg_grouped_gemm_forward(
            current_states, w2_weights, expert_indices, group_size_m, persistent_kernel
        )

        # Restore original token order
        output_states = output_states[inverse_indices]

        # Apply routing weights
        output_states = output_states.view(*topk_weight.shape, -1)
        output_states = (output_states * topk_weight.unsqueeze(-1)).sum(
            dim=1, dtype=output_states.dtype
        )

        final_hidden_states = output_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    try:
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        MixtralSparseMoeBlock.forward = moe_forward_optimized
        logger.info("Successfully patched Mixtral MoE with optimized kernels")
    except ImportError:
        logger.warning("Mixtral model not found, skipping optimized MoE patch")


def patch_qwen3_moe_forward_optimized(
    group_size_m: int = 128, persistent_kernel: bool = True
) -> None:
    """
    Patch Qwen3 MoE forward pass to use optimized kernels.
    """

    def moe_forward_optimized(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Router logits and expert selection
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1, sorted=False
        )
        topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(hidden_states.dtype)

        # Sort tokens by expert
        sorted_states, _, inverse_indices, expert_indices = sort_tokens_by_expert(
            hidden_states, topk_idx, self.num_experts_per_tok
        )

        # Stack expert weights
        gate_up_weights = torch.stack(
            [expert.gate_up_proj.weight for expert in self.experts]
        )
        down_weights = torch.stack([expert.down_proj.weight for expert in self.experts])

        # First projection (gate and up combined)
        gate_up_states = cg_grouped_gemm_forward(
            sorted_states,
            gate_up_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Split and apply activation
        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        gate_states = self.act_fn(gate_states)
        current_states = gate_states * up_states

        # Second projection
        output_states = cg_grouped_gemm_forward(
            current_states,
            down_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Restore order and apply weights
        output_states = output_states[inverse_indices]
        output_states = output_states.view(*topk_weight.shape, -1)
        output_states = (output_states * topk_weight.unsqueeze(-1)).sum(
            dim=1, dtype=output_states.dtype
        )

        final_hidden_states = output_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeSparseMoeBlock,
        )

        Qwen3MoeSparseMoeBlock.forward = moe_forward_optimized
        logger.info("Successfully patched Qwen3 MoE with optimized kernels")
    except ImportError:
        logger.warning("Qwen3 MoE model not found, skipping optimized MoE patch")


def patch_deepseek_v3_moe_forward_optimized(
    group_size_m: int = 128, persistent_kernel: bool = True
) -> None:
    """
    Patch DeepSeek v3 MoE forward pass to use optimized kernels.

    DeepSeek v3 has a unique architecture with:
    - 256 routed experts with 8 activated per token
    - Shared experts that are always activated
    - No auxiliary loss (uses bias term for load balancing)
    """

    def moe_forward_optimized(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass for DeepSeekV3MoE."""
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Get routing weights from gate
        topk_indices, topk_weights = self.gate(hidden_states)

        # Reshape for processing
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        topk_indices_flat = topk_indices.view(-1, topk_indices.shape[-1])
        topk_weights_flat = topk_weights.view(-1, topk_weights.shape[-1])

        # Sort tokens by expert for contiguous memory access
        topk = topk_indices.shape[-1]  # Get topk from the indices shape
        sorted_states, _, inverse_indices, expert_indices = sort_tokens_by_expert(
            hidden_states_flat, topk_indices_flat, topk
        )

        # Stack expert weights - DeepSeek v3 uses gate_proj, up_proj, down_proj
        gate_weights = torch.stack([expert.gate_proj.weight for expert in self.experts])
        up_weights = torch.stack([expert.up_proj.weight for expert in self.experts])
        down_weights = torch.stack([expert.down_proj.weight for expert in self.experts])

        # First projection: gate and up in parallel
        gate_states = cg_grouped_gemm_forward(
            sorted_states, gate_weights, expert_indices, group_size_m, persistent_kernel
        )
        up_states = cg_grouped_gemm_forward(
            sorted_states, up_weights, expert_indices, group_size_m, persistent_kernel
        )

        # Apply activation and element-wise multiplication
        current_states = self.experts[0].act_fn(gate_states) * up_states

        # Down projection
        output_states = cg_grouped_gemm_forward(
            current_states,
            down_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Restore original token order
        output_states = output_states[inverse_indices]

        # Apply routing weights
        output_states = output_states.view(
            batch_size * seq_len, topk_indices.shape[-1], -1
        )
        topk_weights_expanded = topk_weights_flat.unsqueeze(-1)
        output_states = (output_states * topk_weights_expanded).sum(
            dim=1, dtype=output_states.dtype
        )

        # Reshape back to original shape
        output_states = output_states.view(*orig_shape)

        # Add shared expert output (always activated)
        shared_expert_output = self.shared_experts(residuals)
        output_states = output_states + shared_expert_output

        return output_states

    def moe_optimized(
        self,
        hidden_states: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized moe method using contiguous grouped GEMM."""
        batch_size = hidden_states.shape[0]

        # Sort tokens by expert
        sorted_states, _, inverse_indices, expert_indices = sort_tokens_by_expert(
            hidden_states, topk_indices, topk_indices.shape[-1]
        )

        # Stack expert weights
        gate_weights = torch.stack([expert.gate_proj.weight for expert in self.experts])
        up_weights = torch.stack([expert.up_proj.weight for expert in self.experts])
        down_weights = torch.stack([expert.down_proj.weight for expert in self.experts])

        # Apply experts using optimized kernels
        gate_states = cg_grouped_gemm_forward(
            sorted_states, gate_weights, expert_indices, group_size_m, persistent_kernel
        )
        up_states = cg_grouped_gemm_forward(
            sorted_states, up_weights, expert_indices, group_size_m, persistent_kernel
        )

        # Activation
        current_states = self.experts[0].act_fn(gate_states) * up_states

        # Down projection
        output_states = cg_grouped_gemm_forward(
            current_states,
            down_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Restore order and apply weights
        output_states = output_states[inverse_indices]
        output_states = output_states.view(batch_size, topk_indices.shape[-1], -1)
        output_states = (output_states * topk_weights.unsqueeze(-1)).sum(
            dim=1, dtype=output_states.dtype
        )

        return output_states.type(hidden_states.dtype)

    try:
        # Try to patch the standard transformers module
        from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MoE

        DeepseekV3MoE.forward = moe_forward_optimized
        DeepseekV3MoE.moe = moe_optimized
        # Set the optimization parameters as class attributes
        DeepseekV3MoE._group_size_m = group_size_m
        DeepseekV3MoE._persistent_kernel = persistent_kernel
        logger.info(
            "Successfully patched standard DeepSeek v3 MoE with optimized kernels"
        )
    except ImportError:
        logger.warning(
            "Standard DeepSeek v3 model not found, skipping optimized MoE patch"
        )


def patch_qwen2_moe_forward_optimized(
    group_size_m: int = 128, persistent_kernel: bool = True
) -> None:
    """
    Patch Qwen2 MoE forward pass to use optimized kernels.
    """

    def moe_forward_optimized(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        # Shared expert processing (if exists)
        shared_expert_output = None
        if hasattr(self, "shared_expert") and self.shared_expert is not None:
            shared_expert_output = self.shared_expert(hidden_states)
            if (
                hasattr(self, "shared_expert_gate")
                and self.shared_expert_gate is not None
            ):
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states))
                    * shared_expert_output
                )

        # Router and expert selection
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        topk_weight, topk_idx = torch.topk(
            routing_weights, self.num_experts_per_tok, dim=-1, sorted=False
        )

        # Normalize if needed
        if hasattr(self, "norm_topk_prob") and self.norm_topk_prob:
            topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
        topk_weight = topk_weight.to(hidden_states.dtype)

        # Sort tokens by expert
        sorted_states, _, inverse_indices, expert_indices = sort_tokens_by_expert(
            hidden_states, topk_idx, self.num_experts_per_tok
        )

        # Stack expert weights
        gate_up_weights = torch.stack(
            [expert.gate_up_proj.weight for expert in self.experts]
        )
        down_weights = torch.stack([expert.down_proj.weight for expert in self.experts])

        # Apply experts
        gate_up_states = cg_grouped_gemm_forward(
            sorted_states,
            gate_up_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Activation
        gate_states, up_states = gate_up_states.chunk(2, dim=-1)
        gate_states = self.act_fn(gate_states)
        current_states = gate_states * up_states

        # Down projection
        output_states = cg_grouped_gemm_forward(
            current_states,
            down_weights,
            expert_indices,
            group_size_m,
            persistent_kernel,
        )

        # Restore order
        output_states = output_states[inverse_indices]
        output_states = output_states.view(*topk_weight.shape, -1)
        output_states = (output_states * topk_weight.unsqueeze(-1)).sum(
            dim=1, dtype=output_states.dtype
        )

        # Add shared expert output if exists
        if shared_expert_output is not None:
            output_states = output_states + shared_expert_output

        final_hidden_states = output_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits

    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeSparseMoeBlock,
        )

        Qwen2MoeSparseMoeBlock.forward = moe_forward_optimized
        logger.info("Successfully patched Qwen2 MoE with optimized kernels")
    except ImportError:
        logger.warning("Qwen2 MoE model not found, skipping optimized MoE patch")


def apply_moe_kernel_patches(
    models: Optional[list] = None,
    group_size_m: int = 128,
    persistent_kernel: bool = True,
    **kwargs,
) -> None:
    """
    Apply optimized MoE kernel patches to specified models.

    Args:
        models: List of model names to patch. If None, patches all supported models.
        group_size_m: Group size for contiguous grouped GEMM operations
        persistent_kernel: Whether to use persistent kernel with L2 cache optimization
    """

    if models is None:
        models = ["mixtral", "qwen3_moe", "qwen2_moe", "deepseek_v3"]

    patch_functions = {
        "mixtral": patch_mixtral_moe_forward_optimized,
        "qwen3_moe": patch_qwen3_moe_forward_optimized,
        "qwen2_moe": patch_qwen2_moe_forward_optimized,
        "deepseek_v3": patch_deepseek_v3_moe_forward_optimized,
    }

    for model_name in models:
        if model_name in patch_functions:
            try:
                patch_functions[model_name](group_size_m, persistent_kernel)
                logger.info(f"Applied optimized MoE patches to {model_name}")
            except Exception as e:
                logger.error(f"Failed to apply optimized patches to {model_name}: {e}")
        else:
            logger.warning(
                f"Model {model_name} not supported for optimized MoE patches"
            )


class MoeOptimizedPlugin(BasePlugin):
    """
    Plugin for optimized MoE kernel integration with Axolotl.
    """

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.group_size_m = 128
        self.persistent_kernel = True
        self.models = None

    def get_input_args(self) -> str:
        """Returns the pydantic model for the plugin's input arguments."""
        return "axolotl.integrations.moe_kernels.args.MoeOptimizedArgs"

    def register(self, cfg: dict):
        """Register the plugin with the given configuration."""
        self.enabled = cfg.get("moe_kernels", False)
        self.group_size_m = cfg.get("moe_group_size", 128)
        self.persistent_kernel = cfg.get("moe_persistent_kernel", True)
        self.models = cfg.get("moe_kernel_models", None)

        if self.enabled:
            logger.info(
                f"Optimized MoE plugin enabled with group_size_m={self.group_size_m}, persistent_kernel={self.persistent_kernel}"
            )
            if self.models:
                logger.info(f"Will apply to models: {self.models}")

    def pre_model_load(self, cfg: DictDefault):
        """
        Apply optimized MoE patches before model loading.

        This ensures the patches are in place when the model is instantiated.
        """
        if not self.enabled:
            return

        # Determine which models to patch
        models_to_patch = self.models
        if models_to_patch is None:
            # Auto-detect based on model type
            model_type = cfg.get("model_config_type", cfg.get("model_type", ""))
            base_model = cfg.get("base_model", "")

            if "mixtral" in model_type.lower():
                models_to_patch = ["mixtral"]
            elif "qwen3" in model_type.lower() and "moe" in model_type.lower():
                models_to_patch = ["qwen3_moe"]
            elif "qwen2" in model_type.lower() and "moe" in model_type.lower():
                models_to_patch = ["qwen2_moe"]
            elif "deepseek" in model_type.lower() and "v3" in model_type.lower():
                models_to_patch = ["deepseek_v3"]
            elif "deepseek" in base_model.lower() and "v3" in base_model.lower():
                models_to_patch = ["deepseek_v3"]
            elif "DeepSeek-V3" in base_model:
                models_to_patch = ["deepseek_v3"]
            else:
                logger.warning(
                    f"MoE optimization enabled but no compatible model detected for type: {model_type}, base_model: {base_model}"
                )
                return

        # Apply patches
        apply_moe_kernel_patches(
            models=models_to_patch,
            group_size_m=self.group_size_m,
            persistent_kernel=self.persistent_kernel,
        )

        logger.info(f"Optimized MoE patches applied to {models_to_patch}")
