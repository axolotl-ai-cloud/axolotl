"""
Routing functions for SonicMoE integration.

Different MoE architectures use different routing strategies:
- qwen3_moe / qwen2_moe: softmax -> topk (with optional renormalization)
- gpt_oss: topk -> softmax (uses fused moe_TC_softmax_topk_layer, routing_fn=None)
- glm_moe_dsa: sigmoid -> topk (with group-based expert selection)

Each model type maps to a (routing_fn, activation_type, router_attr) triple.
When routing_fn is None, the fused moe_TC_softmax_topk_layer path is used.
"""

import torch
import torch.nn.functional as F


def get_model_moe_config(model_type: str):
    """Returns (routing_fn, activation, router_attr) for a given model type.

    Args:
        model_type: HuggingFace model type string.

    Returns:
        routing_fn: Callable or None. None signals the fused
            moe_TC_softmax_topk_layer path (topk -> softmax models).
        activation: SonicMoE ActivationType enum value.
        router_attr: Name of the router module attribute on the MoE block
            (e.g. "gate" or "router").

    The activation type cannot be derived from config.hidden_act because
    e.g. qwen3_moe reports "silu" but architecturally uses SwiGLU
    (act_fn(gate) * up pattern). So we specify it per model type.
    """
    from sonicmoe.enums import ActivationType

    if model_type in (
        "qwen2_moe",
        "qwen3_moe",
        "qwen3_next",
        "olmoe",
        "mixtral",
        "minimax",
    ):
        return softmax_topk_routing, ActivationType.SWIGLU, "gate"
    elif model_type in (
        "glm_moe_dsa",
        "deepseek_v3",
        "glm4_moe",
        "glm4_moe_lite",
        "glm4v_moe",
        "minimax_m2",
    ):
        return sigmoid_topk_routing, ActivationType.SWIGLU, "gate"
    # elif model_type in ("ernie4_5_moe",):
    #     # Softmax→topk with e_score_correction_bias applied between softmax and topk.
    #     return ..., ActivationType.SWIGLU, "gate"
    # elif model_type in ("deepseek_v2",):
    #     # Softmax→topk with group_limited_greedy. Different attr names: num_group
    #     # (not n_group), gate is nn.Linear (not a router class).
    #     return ..., ActivationType.SWIGLU, "gate"
    # elif model_type in ("hunyuan_v1_moe",):
    #     # Softmax→topk but gate structure differs: gate.wg (not gate.weight),
    #     # top_k on block not gate, creates scatter routing matrix.
    #     return ..., ActivationType.SWIGLU, "gate"
    # Fused topk -> softmax path (routing_fn=None):
    # elif model_type in ("gpt_oss",):
    #     # NOTE: gpt_oss has a router bias which moe_TC_softmax_topk_layer
    #     # ignores (it only takes router_w, not bias). Also has transposed
    #     # weight layout [E, H, 2*I] and custom GLU activation.
    #     return None, ActivationType.SWIGLU, "router"
    else:
        raise ValueError(f"SonicMoE: unsupported model type '{model_type}'")


def softmax_topk_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen3/Qwen2-style routing: softmax -> topk -> optional renorm.

    Args:
        hidden_states: [T, H] flattened token representations
        moe_block: MoE block module (accesses moe_block.gate.*)

    Returns:
        router_scores: [T*K] flattened scores (float32)
        token_indices: [T*K] which token each entry belongs to (int32), sorted ascending
        expert_indices: [T*K] which expert (int32)
        router_logits: [T, E] original logits for aux loss
    """
    gate = moe_block.gate
    T, H = hidden_states.shape
    K = gate.top_k

    # Compute router logits and softmax over all experts
    router_logits = F.linear(hidden_states, gate.weight)  # [T, E]
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

    # Select top-k experts per token
    top_values, top_indices = torch.topk(router_probs, K, dim=-1)  # [T, K] each

    # Renormalize if configured (default True for models without the attribute,
    # e.g. Mixtral/MiniMax which always normalize)
    if getattr(gate, "norm_topk_prob", True):
        top_values = top_values / top_values.sum(dim=-1, keepdim=True)

    # no-op: matches transformers which casts to softmax output dtype (float32).
    # top_values = top_values.to(router_probs.dtype)

    # Flatten for moe_general_routing_inputs.
    # Token indices are naturally sorted ascending from the [T, K] layout:
    # [0, 0, ..., 1, 1, ..., T-1, T-1, ...] — this is required by SonicMoE.
    # Expert sorting is handled internally by general_routing_router_metadata.
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = top_values.reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = top_indices.to(torch.int32).reshape(-1)  # [T*K]

    return flat_scores, flat_token_idx, flat_expert_idx, router_logits


def sigmoid_topk_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sigmoid-based routing: sigmoid -> optional group selection -> topk.

    Supports two variants:
    - **Group selection** (glm_moe_dsa, deepseek_v3, etc.): n_group > 1,
      bias on gate, group-based masking before topk.
    - **No group selection** (minimax_m2): n_group == 1 (or absent),
      bias on moe_block, straight topk from all experts.

    Final routing weights come from the original sigmoid scores (not
    bias-corrected), with optional renormalization and scaling.

    Args:
        hidden_states: [T, H] flattened token representations
        moe_block: MoE block module (accesses moe_block.gate.* and
            optional moe_block.n_group, .topk_group, .top_k, .norm_topk_prob,
            .routed_scaling_factor, .n_routed_experts)

    Returns:
        router_scores: [T*K] flattened scores (float32)
        token_indices: [T*K] which token each entry belongs to (int32), sorted ascending
        expert_indices: [T*K] which expert (int32)
        router_logits: [T, E] original logits for aux loss
    """
    gate = moe_block.gate
    T, H = hidden_states.shape
    K = moe_block.top_k
    E = getattr(moe_block, "n_routed_experts", gate.weight.shape[0])
    n_group = getattr(moe_block, "n_group", 1)

    # Compute router logits and sigmoid probabilities
    router_logits = F.linear(hidden_states.float(), gate.weight.float())  # [T, E]
    router_probs = router_logits.sigmoid()  # [T, E]

    # Bias-corrected scores for expert selection (not used for final weights).
    # glm_moe_dsa/deepseek_v3 store the bias on gate; minimax_m2 stores it on the block.
    e_score_correction_bias = getattr(gate, "e_score_correction_bias", None)
    if e_score_correction_bias is None:
        e_score_correction_bias = getattr(moe_block, "e_score_correction_bias", None)
    if e_score_correction_bias is None:
        raise AttributeError(
            f"sigmoid_topk_routing requires e_score_correction_bias on "
            f"gate ({type(gate)}) or moe_block ({type(moe_block)}), but neither has it"
        )
    scores_for_choice = router_probs + e_score_correction_bias

    # Group-based selection: pick top groups, mask the rest (skip when n_group == 1)
    if n_group > 1:
        group_scores = (
            scores_for_choice.view(-1, n_group, E // n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [T, n_group]
        group_idx = torch.topk(
            group_scores, k=moe_block.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(-1, n_group, E // n_group).reshape(-1, E)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    # Final topk from (possibly masked) scores
    topk_indices = torch.topk(scores_for_choice, k=K, dim=-1, sorted=False)[1]

    # Gather weights from original sigmoid scores (not bias-corrected)
    topk_weights = router_probs.gather(1, topk_indices)

    # Optional renormalization + scaling
    norm_topk_prob = getattr(moe_block, "norm_topk_prob", True)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routed_scaling_factor = getattr(moe_block, "routed_scaling_factor", 1.0)
    topk_weights = topk_weights * routed_scaling_factor

    # Flatten for moe_general_routing_inputs.
    # Token indices are naturally sorted ascending from the [T, K] layout.
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = topk_weights.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = topk_indices.to(torch.int32).reshape(-1)  # [T*K]

    return flat_scores, flat_token_idx, flat_expert_idx, router_logits
