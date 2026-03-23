"""
Routing functions for SonicMoE integration.

Different MoE architectures use different routing strategies:
- qwen3_moe / qwen2_moe / qwen3_5_moe / qwen3_vl_moe / qwen3_omni_moe: softmax -> topk (with optional renormalization)
- mistral4: softmax -> group selection -> topk (with renormalization and scaling)
- glm_moe_dsa / deepseek_v3 / minimax_m2: sigmoid -> topk (with group-based expert selection)
- ernie4_5_moe: softmax -> bias correction -> topk -> gather (softmax_bias_topk_routing)
- hunyuan_v1_moe: softmax -> topk via gate.wg (softmax_topk_wg_routing)
- gpt_oss: topk -> softmax (uses fused moe_TC_softmax_topk_layer, routing_fn=None) [NOT YET SUPPORTED]

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
        "qwen3_5_moe",
        "qwen3_next",
        "qwen3_vl_moe",
        "qwen3_omni_moe",
        "olmoe",
        "mixtral",
        "minimax",
    ):
        return softmax_topk_routing, ActivationType.SWIGLU, "gate"
    elif model_type in ("mistral4",):
        return softmax_group_topk_routing, ActivationType.SWIGLU, "gate"
    elif model_type in (
        "glm_moe_dsa",
        "deepseek_v3",
        "glm4_moe",
        "glm4_moe_lite",
        "glm4v_moe",
        "minimax_m2",
    ):
        return sigmoid_topk_routing, ActivationType.SWIGLU, "gate"
    elif model_type in ("ernie4_5_moe",):
        return softmax_bias_topk_routing, ActivationType.SWIGLU, "gate"
    elif model_type in ("hunyuan_v1_moe",):
        return softmax_topk_wg_routing, ActivationType.SWIGLU, "gate"
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
    T, _ = hidden_states.shape
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


def softmax_group_topk_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mistral4-style routing: softmax -> group selection -> topk -> renorm -> scale."""
    gate = moe_block.gate
    T, _ = hidden_states.shape
    K = moe_block.top_k
    E = getattr(moe_block, "n_routed_experts", gate.weight.shape[0])
    n_group = getattr(moe_block, "n_group", 1)

    router_logits = F.linear(hidden_states, gate.weight)  # [T, E]
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

    scores_for_choice = router_probs

    # Group selection: pick top groups, mask the rest
    if n_group > 1:
        group_scores = (
            scores_for_choice.view(-1, n_group, E // n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=moe_block.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1).expand(-1, n_group, E // n_group).reshape(-1, E)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    topk_indices = torch.topk(scores_for_choice, k=K, dim=-1, sorted=False)[1]
    topk_weights = router_probs.gather(1, topk_indices)

    # Renormalization + scaling
    norm_topk_prob = getattr(moe_block, "norm_topk_prob", True)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routed_scaling_factor = getattr(moe_block, "routed_scaling_factor", 1.0)
    topk_weights = topk_weights * routed_scaling_factor

    # Flatten for moe_general_routing_inputs
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = topk_weights.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = topk_indices.to(torch.int32).reshape(-1)  # [T*K]

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
    T, _ = hidden_states.shape
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


def softmax_bias_topk_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ernie 4.5 MoE routing: softmax → bias correction → topk → gather → renorm.

    Differs from standard softmax_topk_routing in three ways:
    1. A learned e_score_correction_bias is added to softmax probs *before* topk
       (selection uses biased scores, but final weights use original probs).
    2. The bias is applied via gate.moe_statics module (not a raw tensor).
    3. Renormalization uses clamp(min=norm_min) instead of sum+epsilon.

    Reference: Ernie4_5_MoeTopKRouter.forward in transformers.

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
    T, _ = hidden_states.shape
    K = gate.top_k

    # Compute router logits and softmax (force float32 for numerical stability)
    router_logits = F.linear(hidden_states.float(), gate.weight.float())  # [T, E]
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

    # Bias-corrected scores for expert selection (via moe_statics module)
    scores_for_choice = gate.moe_statics(router_probs)  # [T, E]

    # Select top-k experts using biased scores
    _, selected_experts = torch.topk(scores_for_choice, K, dim=-1)  # [T, K]

    # Gather weights from *original* (unbiased) softmax probs
    top_values = torch.gather(router_probs, dim=-1, index=selected_experts)  # [T, K]

    # Renormalize with clamp(min=norm_min) instead of sum+epsilon
    norm_min = getattr(gate, "norm_min", 1e-20)
    top_values = top_values / torch.clamp(
        top_values.sum(dim=-1, keepdim=True), min=norm_min
    )

    # Flatten for moe_general_routing_inputs
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = top_values.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = selected_experts.to(torch.int32).reshape(-1)  # [T*K]

    return flat_scores, flat_token_idx, flat_expert_idx, router_logits


def softmax_group_limited_topk_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek V2 routing: softmax → group_limited_greedy/greedy → topk → scale.

    Differs from softmax_group_topk_routing (Mistral4) in several ways:
    1. Uses ``num_group`` attribute (not ``n_group``).
    2. Group score = max per group (not sum of top-2).
    3. Supports ``greedy`` method (plain topk without groups).
    4. No renormalization — just ``topk_weight * routed_scaling_factor``.
    5. Gate is ``nn.Linear`` (access weight via ``gate.weight``).

    Reference: DeepseekV2Moe.route_tokens_to_experts in transformers.

    Args:
        hidden_states: [T, H] flattened token representations
        moe_block: MoE block module (accesses moe_block.gate, .num_group,
            .topk_group, .top_k, .topk_method, .routed_scaling_factor)

    Returns:
        router_scores: [T*K] flattened scores (float32)
        token_indices: [T*K] which token each entry belongs to (int32), sorted ascending
        expert_indices: [T*K] which expert (int32)
        router_logits: [T, E] original logits for aux loss
    """
    gate = moe_block.gate
    T, _ = hidden_states.shape
    K = moe_block.top_k
    num_group = getattr(moe_block, "num_group", 1)
    num_experts = gate.weight.shape[0]
    topk_method = getattr(moe_block, "topk_method", "greedy")

    # Compute logits in float32 and softmax
    router_logits = F.linear(hidden_states.float(), gate.weight.float())  # [T, E]
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

    if topk_method == "greedy" or num_group == 1:
        topk_weights, topk_indices = torch.topk(router_probs, k=K, dim=-1, sorted=False)
    elif topk_method == "group_limited_greedy":
        # Guard: selected groups must contain enough experts for topk
        group_size = num_experts // num_group
        if moe_block.topk_group * group_size < K:
            raise ValueError(
                f"DeepSeek V2: topk_group ({moe_block.topk_group}) * group_size "
                f"({group_size}) = {moe_block.topk_group * group_size} < top_k ({K}). "
                f"Not enough experts in selected groups for topk selection."
            )
        # Group selection: pick top groups by max score per group
        group_scores = (
            router_probs.view(T, num_group, num_experts // num_group).max(dim=-1).values
        )  # [T, num_group]
        group_idx = torch.topk(
            group_scores, k=moe_block.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(T, num_group, num_experts // num_group)
            .reshape(T, -1)
        )
        tmp_scores = router_probs.masked_fill(~score_mask.bool(), 0.0)
        topk_weights, topk_indices = torch.topk(tmp_scores, k=K, dim=-1, sorted=False)
    else:
        raise ValueError(
            f"DeepSeek V2: unsupported topk_method '{topk_method}'. "
            f"Expected 'greedy' or 'group_limited_greedy'."
        )

    # Scale only — no renormalization (weights won't sum to 1.0 per token).
    # This matches the reference DeepseekV2Moe.route_tokens_to_experts behavior.
    routed_scaling_factor = getattr(moe_block, "routed_scaling_factor", 1.0)
    topk_weights = topk_weights * routed_scaling_factor

    # Flatten for moe_general_routing_inputs
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = topk_weights.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = topk_indices.to(torch.int32).reshape(-1)  # [T*K]

    return flat_scores, flat_token_idx, flat_expert_idx, router_logits


def softmax_topk_wg_routing(
    hidden_states: torch.Tensor, moe_block
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """HunYuan V1 MoE routing: softmax → topk → renorm (gate weight via gate.wg).

    Differs from standard softmax_topk_routing in:
    1. Gate weight lives at ``gate.wg.weight`` (not ``gate.weight``).
    2. ``top_k`` is on ``moe_block`` (not ``gate``).
    3. Always renormalizes (no ``norm_topk_prob`` flag).

    Reference: HunYuanMoEV1Moe.route_tokens_to_experts and
    HunYuanMoEV1Gate.forward in transformers.

    Args:
        hidden_states: [T, H] flattened token representations
        moe_block: MoE block module (accesses moe_block.gate.wg, moe_block.top_k)

    Returns:
        router_scores: [T*K] flattened scores (float32)
        token_indices: [T*K] which token each entry belongs to (int32), sorted ascending
        expert_indices: [T*K] which expert (int32)
        router_logits: [T, E] original logits for aux loss
    """
    gate = moe_block.gate
    T, _ = hidden_states.shape
    K = moe_block.top_k

    # Gate computes logits via gate.wg (nn.Linear, float32)
    wg = gate.wg
    router_logits = F.linear(hidden_states.float(), wg.weight.float())  # [T, E]
    router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)  # [T, E]

    # Select top-k experts
    top_values, top_indices = torch.topk(router_probs, K, dim=-1)  # [T, K] each

    # Always renormalize (HunYuan V1 has no norm_topk_prob flag)
    top_values = top_values / (top_values.sum(dim=-1, keepdim=True) + 1e-20)

    # Flatten for moe_general_routing_inputs
    token_indices = (
        torch.arange(T, device=hidden_states.device, dtype=torch.int32)
        .unsqueeze(1)
        .expand(T, K)
    )

    flat_scores = top_values.to(torch.float32).reshape(-1)  # [T*K]
    flat_token_idx = token_indices.reshape(-1)  # [T*K]
    flat_expert_idx = top_indices.to(torch.int32).reshape(-1)  # [T*K]

    return flat_scores, flat_token_idx, flat_expert_idx, router_logits
