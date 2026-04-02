"""
SonicMoE-accelerated experts forward for Gemma4.

Gemma4 has no separate SparseMoeBlock — MoE is embedded in the decoder layer.
This module provides a drop-in replacement for ``Gemma4TextExperts.forward``
that uses SonicMoE kernels while preserving the original call signature.
"""

import torch

from .lora import has_lora, materialize_expert_lora, unwrap_experts_lora


def _get_expert_weights_gemma4(experts_module):
    """Extract expert weights from Gemma4TextExperts, applying LoRA if active.

    Returns:
        (gate_up_weight, down_weight) in SonicMoE layout [dim, dim, E].
    """
    if has_lora(experts_module):
        base_experts, lora_dict = unwrap_experts_lora(experts_module)
        gate_up = materialize_expert_lora(
            base_experts.gate_up_proj, lora_dict.get("gate_up_proj")
        )
        down = materialize_expert_lora(
            base_experts.down_proj, lora_dict.get("down_proj")
        )
    else:
        gate_up = experts_module.gate_up_proj
        down = experts_module.down_proj

    # Permute to SonicMoE layout:
    #   gate_up: [E, 2*I, H] -> [2*I, H, E]
    #   down:    [E, H, I]   -> [H, I, E]
    return gate_up.permute(1, 2, 0), down.permute(1, 2, 0)


def gemma4_sonicmoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """SonicMoE-accelerated replacement for Gemma4TextExperts.forward.

    Same signature as the original: (hidden_states [T, H], top_k_index [T, K],
    top_k_weights [T, K]) -> output [T, H].
    """
    from sonicmoe import moe_general_routing_inputs
    from sonicmoe.enums import ActivationType

    T, _ = hidden_states.shape
    K = top_k_index.shape[1]
    E = self.num_experts

    # Convert routing outputs to SonicMoE's flat format
    # Token indices sorted ascending (required by SonicMoE)
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )
    flat_scores = top_k_weights.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = top_k_index.to(torch.int32).reshape(-1)  # [T*K]

    # Get weights (with LoRA materialization if needed)
    gate_up_weight, down_weight = _get_expert_weights_gemma4(self)
    gate_up_weight = gate_up_weight.to(hidden_states.dtype)
    down_weight = down_weight.to(hidden_states.dtype)

    if not torch.cuda.is_available():
        raise RuntimeError("SonicMoE requires CUDA. No CUDA device available.")
    cuda_stream = torch.cuda.current_stream().cuda_stream

    output, _ = moe_general_routing_inputs(
        hidden_states,
        flat_scores,
        flat_token_idx,
        flat_expert_idx,
        gate_up_weight,
        None,  # b1 (no gate/up bias)
        down_weight,
        None,  # b2 (no down bias)
        E,
        cuda_stream,
        ActivationType.GEGLU,
        False,  # is_inference_mode
    )

    return output


def patch_gemma4_sonicmoe():
    """Monkeypatch Gemma4TextExperts.forward with SonicMoE kernel."""
    from axolotl.integrations.kernels.constants import resolve_experts_class

    experts_cls = resolve_experts_class("gemma4_text")
    if experts_cls is None:
        raise ValueError("Could not resolve Gemma4TextExperts class")

    if hasattr(experts_cls, "_original_forward"):
        return  # already patched

    experts_cls._original_forward = experts_cls.forward
    experts_cls.forward = gemma4_sonicmoe_experts_forward
