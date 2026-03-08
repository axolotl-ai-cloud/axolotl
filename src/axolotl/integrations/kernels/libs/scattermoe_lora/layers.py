# SPDX-License-Identifier: Apache-2.0
#
# Original work Copyright (c) Shawn Tan and ScatterMoE Contributors
# Adapted from https://github.com/shawntan/scattermoe
# See https://github.com/shawntan/scattermoe/blob/main/LICENSE
#
# Modifications and LoRA adaptation Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
ScatterMoE layer replacements for HuggingFace MoE architectures.

Provides drop-in forward replacements that use ScatterMoE kernels for
acceleration. When used via the HF ``kernels`` library
(``replace_kernel_forward_from_hub``), these classes replace the forward
method of the original MoE block.

LoRA support
------------
When peft wraps parameters via ``target_parameters``, the ``self.experts``
submodule becomes a chain of ``ParamWrapper`` objects and the ``self.gate``
router may also become a ``ParamWrapper``.  The ``HFScatterMoEGatedMLP``
forward detects this and automatically:

1. Unwraps ``self.gate`` to the base router, applying gate LoRA delta
2. Unwraps ``self.experts`` to the base ``OlmoeExperts`` module
3. Extracts LoRA A/B weights and scaling from each wrapper
4. Converts B layout from peft rank-major to scattermoe expert-major
5. Routes to ``parallel_linear_lora`` for fused LoRA computation
6. Passes through ``self.shared_expert`` / ``self.shared_expert_gate``
   (peft wraps their linear layers with standard LoRA, no special handling)
"""

import torch
from torch import nn
from torch.nn import functional as F

from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora

# =============================================================================
# LoRA layout conversion utilities (peft <-> scattermoe)
# =============================================================================


def peft_lora_B_to_scattermoe(peft_B, num_experts, rank):
    """Convert peft rank-major lora_B ``[out, E*r]`` to scattermoe
    expert-major ``[N, r*E]``.

    peft reshapes B to ``[out, r, E]`` (rank-major).
    scattermoe slices B as ``[:, e*r:(e+1)*r]`` (expert-major).
    """
    N = peft_B.shape[0]
    return (
        peft_B.reshape(N, rank, num_experts)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(N, num_experts * rank)
    )


def peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
    """Convert peft LoRA weights to scattermoe layout (with A<->B swap).

    peft operates on the parameter in its native storage layout ``[E, dim1, dim2]``
    where ``in_features=dim1, out_features=dim2``.  ScatterMoE transposes the
    parameter (``W = param.transpose(2, 1)``) giving ``[E, dim2, dim1]`` with
    ``K=dim2, N=dim1``.  Because of this transposition, peft's A and B roles
    are swapped relative to scattermoe's convention.

    peft gives:
        lora_A ``[r*E, dim1]``, lora_B ``[dim2, r*E]``

    scattermoe needs:
        lora_A ``[r*E, K=dim2]``, lora_B ``[N=dim1, r*E]``

    This function swaps A<->B and converts B from rank-major to expert-major.
    Uses vectorized tensor operations (no Python loop over experts).

    Works for **both** gate_up_proj and down_proj since the transposition
    issue is the same for any parameter.
    """
    peft_B_em = peft_lora_B_to_scattermoe(peft_B, num_experts, rank)

    dim1 = peft_A.shape[1]  # peft in_features -> scattermoe N
    dim2 = peft_B_em.shape[0]  # peft out_features -> scattermoe K

    # smoe_A: per expert, transpose B_e [dim2, r] -> [r, dim2]
    # [dim2, E*r] -> [dim2, E, r] -> [E, r, dim2] -> [E*r, dim2]
    smoe_A = (
        peft_B_em.reshape(dim2, num_experts, rank)
        .permute(1, 2, 0)
        .contiguous()
        .reshape(rank * num_experts, dim2)
    )

    # smoe_B: per expert, transpose A_e [r, dim1] -> [dim1, r]
    # [E*r, dim1] -> [E, r, dim1] -> [dim1, E, r] -> [dim1, E*r]
    smoe_B = (
        peft_A.reshape(num_experts, rank, dim1)
        .permute(2, 0, 1)
        .contiguous()
        .reshape(dim1, num_experts * rank)
    )

    return smoe_A, smoe_B


def peft_down_proj_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
    """Deprecated alias for :func:`peft_lora_to_scattermoe`."""
    return peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank)


# =============================================================================
# ParamWrapper unwrapping
# =============================================================================


def _unwrap_gate_lora(gate_module):
    """Unwrap peft ``ParamWrapper`` on the router gate.

    When peft targets ``gate.weight``, ``self.gate`` becomes::

        ParamWrapper(weight)
          -> base_layer: OlmoeTopKRouter (the real module)

    This function detects the wrapping and returns the base router, its
    weight tensor, and an optional LoRA delta tensor.

    Returns:
        (base_gate, gate_weight, gate_lora_delta_or_None)

        ``base_gate`` is the original router module (with ``.top_k``,
        ``.num_experts``, ``.norm_topk_prob``).
        ``gate_weight`` is the base router weight (may be a DTensor under FSDP).
        ``gate_lora_delta_or_None`` is the LoRA delta tensor if LoRA is active,
        else ``None``.  Kept separate to avoid mixing DTensor + Tensor in an add.
    """
    if hasattr(gate_module, "base_layer") and hasattr(gate_module, "lora_A"):
        base_gate = gate_module.base_layer
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gate_module)
        if lora_A is not None:
            # gate weight: [num_experts, hidden_size]
            # lora_A: [r, hidden_size], lora_B: [num_experts, r]
            # delta = scaling * B @ A = [num_experts, hidden_size]
            delta = scaling * (lora_B @ lora_A)
            return base_gate, base_gate.weight, delta
        else:
            return base_gate, base_gate.weight, None
    else:
        # No wrapping — gate is the original module
        return gate_module, gate_module.weight, None


def _convert_smoe_lora(lora_A, lora_B, num_experts, rank, scaling):
    """Convert peft LoRA weights to scattermoe layout."""
    smoe_A, smoe_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
    return (smoe_A, smoe_B, scaling)


def _unwrap_experts_lora(experts_module):
    """Walk a peft ``ParamWrapper`` chain on ``self.experts``.

    When peft targets ``experts.gate_up_proj`` and ``experts.down_proj`` via
    ``target_parameters``, ``self.experts`` becomes a nested chain::

        ParamWrapper(down_proj)
          -> base_layer: ParamWrapper(gate_up_proj)
              -> base_layer: OlmoeExperts (the real module)

    This function walks the chain, collects LoRA params keyed by
    ``parameter_name``, and returns the base experts module.

    Returns:
        (base_experts, gup_lora, down_lora)

        Each ``*_lora`` is either ``(smoe_A, smoe_B, scaling)`` or ``None``.
        A/B are already in scattermoe layout.
    """
    # Collect ParamWrapper layers by their parameter_name
    wrappers = {}
    module = experts_module
    while hasattr(module, "base_layer") and hasattr(module, "lora_A"):
        param_name = getattr(module, "parameter_name", None)
        if param_name is not None:
            wrappers[param_name] = module
        module = module.base_layer

    base_experts = module

    if not wrappers:
        return base_experts, None, None

    # Determine num_experts from base module
    num_experts = getattr(base_experts, "num_experts", None)
    if num_experts is None:
        # Fallback: infer from parameter shape
        gup = getattr(base_experts, "gate_up_proj", None)
        if gup is not None:
            num_experts = gup.shape[0]

    # Extract gate_up_proj LoRA (needs A<->B swap due to transposition)
    gup_lora = None
    gup_wrapper = wrappers.get("gate_up_proj")
    if gup_wrapper is not None:
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gup_wrapper)
        if lora_A is not None:
            rank = lora_A.shape[0] // num_experts
            gup_lora = _convert_smoe_lora(lora_A, lora_B, num_experts, rank, scaling)

    # Extract down_proj LoRA (needs A<->B swap due to transposition)
    down_lora = None
    down_wrapper = wrappers.get("down_proj")
    if down_wrapper is not None:
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(down_wrapper)
        if lora_A is not None:
            rank = lora_A.shape[0] // num_experts
            down_lora = _convert_smoe_lora(lora_A, lora_B, num_experts, rank, scaling)

    return base_experts, gup_lora, down_lora


# =============================================================================
# Routing helpers
# =============================================================================


def _softmax_topk_route(
    moe_block, base_gate, hidden_states, gate_weight, gate_lora_delta
):
    """Softmax→topk routing (Qwen, OLMoE, Mixtral, MiniMax).

    Returns:
        (routing_weights [T, K], selected_experts [T, K], top_k, num_experts)
    """
    router_logits = F.linear(hidden_states, gate_weight)
    if gate_lora_delta is not None:
        router_logits = router_logits + F.linear(hidden_states, gate_lora_delta)
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

    top_k = base_gate.top_k
    num_experts = base_gate.num_experts
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    if getattr(base_gate, "norm_topk_prob", True):
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

    return routing_weights, selected_experts, top_k, num_experts


def _sigmoid_topk_route(
    moe_block, base_gate, hidden_states, gate_weight, gate_lora_delta
):
    """Sigmoid→topk routing (GLM, DeepSeek V3, MiniMax M2).

    Supports:
    - ``e_score_correction_bias`` on gate or moe_block
    - Group-based expert selection when ``n_group > 1``
    - ``routed_scaling_factor`` applied to final weights
    - Final weights gathered from original sigmoid probs (not bias-corrected)

    Returns:
        (routing_weights [T, K], selected_experts [T, K], top_k, num_experts)
    """
    router_logits = F.linear(hidden_states.float(), gate_weight.float())
    if gate_lora_delta is not None:
        router_logits = router_logits + F.linear(
            hidden_states.float(), gate_lora_delta.float()
        )
    router_probs = router_logits.sigmoid()  # [T, E]

    top_k = getattr(moe_block, "top_k", getattr(base_gate, "top_k", None))
    num_experts = getattr(moe_block, "n_routed_experts", gate_weight.shape[0])

    # Bias-corrected scores for expert selection (not used for final weights).
    # glm_moe_dsa/deepseek_v3 store the bias on gate; minimax_m2 on the block.
    e_score_correction_bias = getattr(base_gate, "e_score_correction_bias", None)
    if e_score_correction_bias is None:
        e_score_correction_bias = getattr(moe_block, "e_score_correction_bias", None)
    if e_score_correction_bias is not None:
        scores_for_choice = router_probs + e_score_correction_bias
    else:
        scores_for_choice = router_probs

    # Group-based selection: pick top groups, mask the rest
    n_group = getattr(moe_block, "n_group", 1)
    if n_group > 1:
        group_scores = (
            scores_for_choice.view(-1, n_group, num_experts // n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )  # [T, n_group]
        topk_group = getattr(moe_block, "topk_group", n_group)
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, num_experts // n_group)
            .reshape(-1, num_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)

    # Final topk from (possibly masked) scores
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]

    # Gather weights from original sigmoid scores (not bias-corrected)
    topk_weights = router_probs.gather(1, topk_indices)

    # Optional renormalization + scaling
    if getattr(moe_block, "norm_topk_prob", True):
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    routed_scaling_factor = getattr(moe_block, "routed_scaling_factor", 1.0)
    topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_indices, top_k, num_experts


def _route(moe_block, base_gate, hidden_states, gate_weight, gate_lora_delta):
    """Dispatch to the correct routing strategy based on block attributes.

    Detects sigmoid routing by the presence of ``e_score_correction_bias``
    on either the gate or the moe_block.
    """
    has_sigmoid = (
        getattr(base_gate, "e_score_correction_bias", None) is not None
        or getattr(moe_block, "e_score_correction_bias", None) is not None
    )
    if has_sigmoid:
        return _sigmoid_topk_route(
            moe_block, base_gate, hidden_states, gate_weight, gate_lora_delta
        )
    return _softmax_topk_route(
        moe_block, base_gate, hidden_states, gate_weight, gate_lora_delta
    )


# =============================================================================
# Shared expert helpers
# =============================================================================


def _compute_shared_expert(moe_block, hidden_states_flat):
    """Compute shared expert output if the block has one.

    Handles singular (qwen2_moe: ``shared_expert``), plural
    (glm_moe_dsa/deepseek_v3: ``shared_experts``), and MLP
    (hunyuan_v1_moe: ``shared_mlp``) attribute names.

    peft wraps individual linear layers inside the shared expert with
    standard LoRA — calling forward() handles this transparently.
    """
    shared_expert = (
        getattr(moe_block, "shared_expert", None)
        or getattr(moe_block, "shared_experts", None)
        or getattr(moe_block, "shared_mlp", None)
    )
    if shared_expert is None:
        return None

    shared_expert_output = shared_expert(hidden_states_flat)

    # Optional sigmoid gate (Qwen2MoE pattern).
    # shared_expert_gate may also be peft-wrapped (standard LoRA
    # on nn.Linear), its forward() applies LoRA automatically.
    shared_expert_gate = getattr(moe_block, "shared_expert_gate", None)
    if shared_expert_gate is not None:
        shared_expert_output = (
            F.sigmoid(shared_expert_gate(hidden_states_flat)) * shared_expert_output
        )

    return shared_expert_output


# =============================================================================
# Layer classes
# =============================================================================


class ScatterMoEGatedMLP(nn.Module):
    def forward(self, layer_input):
        """
        Forward pass of the mixture of experts layer.

        Args:
            layer_input (Tensor):
                Input tensor.

        Returns:
            Tensor:
                Output tensor.
        """
        bsz, length, emb_size = layer_input.size()
        layer_input = layer_input.reshape(-1, emb_size)
        # compute the top_k routing decision
        router_logits = self.router.layer(layer_input)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.router.top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(layer_input.dtype)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            selected_experts, num_experts=self.router.num_experts
        )

        # compute experts
        gates, h = parallel_linear(
            layer_input,
            self.input_linear.weight.transpose(2, 1),
            self.router.top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        ).chunk(2, dim=-1)
        h = self.activation(gates) * h
        layer_output = parallel_linear(
            h,
            self.output_linear.weight.transpose(2, 1),
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )
        layer_output = layer_output.view(bsz, length, emb_size)
        return layer_output


class HFScatterMoEGatedMLP(nn.Module):
    """
    ScatterMoE-accelerated forward pass for HF MoEs.

    Used as a kernel layer via the HF ``kernels`` library.  The ``forward``
    method replaces the original SparseMoeBlock.forward.

    Supports:

    * **Softmax→topk routing**: OLMoE, Qwen2/3MoE, Mixtral, MiniMax
    * **Sigmoid→topk routing**: GLM, DeepSeek V3, MiniMax M2
    * **Full-parameter training**: uses ``parallel_linear`` (base ScatterMoE)
    * **LoRA fine-tuning**: detects peft ``ParamWrapper`` on ``self.experts``,
      extracts adapter weights, and uses ``parallel_linear_lora`` (fused kernel)
    """

    @staticmethod
    def forward(self: nn.Module, layer_input: torch.Tensor):
        """
        Forward pass using ScatterMoE kernels.

        Args:
            self: The MoeSparseMoeBlock module containing:
                - self.gate: Router (or peft ParamWrapper wrapping it)
                - self.experts: Experts module (or peft ParamWrapper chain)
                - self.shared_expert(s): Optional shared expert
                - self.shared_expert_gate: Optional shared expert gate
            layer_input: Input tensor [batch_size, seq_len, hidden_size]

        Returns:
            Tensor: [batch_size, seq_len, hidden_size]
        """
        batch_size, sequence_length, hidden_dim = layer_input.shape
        hidden_states_flat = layer_input.view(-1, hidden_dim)

        # ====================================================================
        # Shared Expert (if present, e.g. Qwen2MoE, DeepSeek V3)
        # ====================================================================
        shared_expert_output = _compute_shared_expert(self, hidden_states_flat)

        # ====================================================================
        # Router Computation (with optional gate LoRA)
        # ====================================================================
        base_gate, gate_weight, gate_lora_delta = _unwrap_gate_lora(self.gate)
        routing_weights, selected_experts, top_k, num_experts = _route(
            self, base_gate, hidden_states_flat, gate_weight, gate_lora_delta
        )
        routing_weights = routing_weights.to(hidden_states_flat.dtype)

        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
            selected_experts, num_experts=num_experts
        )

        # ====================================================================
        # Detect LoRA (peft ParamWrapper) and extract adapter weights
        # ====================================================================
        experts, gup_lora, down_lora = _unwrap_experts_lora(self.experts)

        # ====================================================================
        # Gate + Up projection
        # ====================================================================
        gate_up_W = experts.gate_up_proj.transpose(2, 1)  # [E, hidden, 2*inter]

        if gup_lora is not None:
            gup_A, gup_B, gup_scaling = gup_lora
            gup = parallel_linear_lora(
                hidden_states_flat,
                gate_up_W,
                top_k,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                lora_A=gup_A,
                lora_B=gup_B,
                scaling=gup_scaling,
                grouped_in=False,
                grouped_out=True,
                use_fused_dX=True,
                use_fused_gather=True,
            )
        else:
            gup = parallel_linear(
                hidden_states_flat,
                gate_up_W,
                top_k,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                grouped_in=False,
                grouped_out=True,
            )

        gates, h = gup.chunk(2, dim=-1)
        h = experts.act_fn(gates) * h

        # ====================================================================
        # Down projection
        # ====================================================================
        down_W = experts.down_proj.transpose(2, 1)  # [E, inter, hidden]

        if down_lora is not None:
            down_A, down_B, down_scaling = down_lora
            expert_output = parallel_linear_lora(
                h,
                down_W,
                1,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                lora_A=down_A,
                lora_B=down_B,
                scaling=down_scaling,
                gates=routing_weights,
                grouped_in=True,
                grouped_out=False,
                use_fused_dX=True,
                use_fused_gather=True,
            )
        else:
            expert_output = parallel_linear(
                h,
                down_W,
                1,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                grouped_in=True,
                grouped_out=False,
                gates=routing_weights,
            )

        # ====================================================================
        # Combine with shared expert and reshape
        # ====================================================================
        if shared_expert_output is not None:
            expert_output = expert_output + shared_expert_output

        expert_output = expert_output.view(batch_size, sequence_length, hidden_dim)
        return expert_output
