"""
ScatterMoE-accelerated experts forward for Gemma4.

Gemma4 has no separate SparseMoeBlock — MoE is embedded in the decoder layer.
The decoder layer handles routing (Gemma4TextRouter) and calls
``experts(hidden_states, top_k_index, top_k_weights)`` directly.

This module registers a ``"scattermoe"`` implementation in the transformers
``ExpertsInterface``, which the ``@use_experts_implementation`` decorator
dispatches to when ``config._experts_implementation == "scattermoe"``.

This is the clean way to hook into transformers' MoE dispatch — no
monkeypatching required.  Works for Gemma4 and any future model that uses
``@use_experts_implementation`` with the standard forward signature
``(hidden_states, top_k_index, top_k_weights) -> Tensor``.
"""

import torch

from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora


def _has_peft_wrapper(module):
    """Check if a module's parameter has been wrapped by PEFT ParamWrapper."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        for attr in ("gate_up_proj", "down_proj"):
            param = getattr(module, attr, None)
            if isinstance(param, ParamWrapper):
                return True
    except ImportError:
        pass
    return False


def _unwrap_experts_lora(experts):
    """Extract base weights and LoRA params from a PEFT-wrapped Experts module.

    Returns:
        (base_experts, gup_lora, down_lora) where each lora is
        (lora_A, lora_B, scaling) or None.
    """
    try:
        from peft.tuners.param_wrapper import ParamWrapper
    except ImportError:
        return experts, None, None

    if not isinstance(getattr(experts, "gate_up_proj", None), ParamWrapper):
        return experts, None, None

    base_experts = experts
    gup_lora = None
    down_lora = None

    gup_param = experts.gate_up_proj
    if isinstance(gup_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(gup_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            gup_lora = (sm_A, sm_B, scaling)

    down_param = experts.down_proj
    if isinstance(down_param, ParamWrapper):
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(down_param)
        if lora_A is not None:
            num_experts = experts.num_experts
            rank = lora_A.shape[0] // num_experts
            from .layers import peft_lora_to_scattermoe

            sm_A, sm_B = peft_lora_to_scattermoe(lora_A, lora_B, num_experts, rank)
            down_lora = (sm_A, sm_B, scaling)

    return base_experts, gup_lora, down_lora


def _get_base_param(param):
    """Get the base tensor from a PEFT ParamWrapper or regular Parameter."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper

        while isinstance(param, ParamWrapper):
            param = param.original_parameter
    except ImportError:
        pass
    return param


def scattermoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE-accelerated experts forward.

    Drop-in replacement for the standard Experts forward signature used by
    ``@use_experts_implementation``-decorated classes (Gemma4, Mixtral, etc.):
    ``(hidden_states [T, H], top_k_index [T, K], top_k_weights [T, K]) -> [T, H]``
    """
    K = top_k_index.shape[1]

    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    # Get base weights (unwrap PEFT if needed)
    gate_up_weight = _get_base_param(self.gate_up_proj).transpose(2, 1)
    down_weight = _get_base_param(self.down_proj).transpose(2, 1)

    # Check for LoRA
    if _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

        if gup_lora is not None:
            lora_A, lora_B, scaling = gup_lora
            gates_h = parallel_linear_lora(
                hidden_states,
                gate_up_weight,
                K,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                lora_A,
                lora_B,
                scaling,
                grouped_in=False,
                grouped_out=True,
            )
        else:
            gates_h = parallel_linear(
                hidden_states,
                gate_up_weight,
                K,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                grouped_in=False,
                grouped_out=True,
            )

        gates, h = gates_h.chunk(2, dim=-1)
        h = self.act_fn(gates) * h

        if down_lora is not None:
            lora_A, lora_B, scaling = down_lora
            output = parallel_linear_lora(
                h,
                down_weight,
                1,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                lora_A,
                lora_B,
                scaling,
                grouped_in=True,
                grouped_out=False,
                gates=routing_weights,
            )
        else:
            output = parallel_linear(
                h,
                down_weight,
                1,
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                grouped_in=True,
                grouped_out=False,
                gates=routing_weights,
            )
    else:
        # No LoRA — standard ScatterMoE path
        gates_h = parallel_linear(
            hidden_states,
            gate_up_weight,
            K,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=False,
            grouped_out=True,
        )
        gates, h = gates_h.chunk(2, dim=-1)
        h = self.act_fn(gates) * h

        output = parallel_linear(
            h,
            down_weight,
            1,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            grouped_in=True,
            grouped_out=False,
            gates=routing_weights,
        )

    return output


def register_scattermoe_experts():
    """Register ``"scattermoe"`` in the transformers ExpertsInterface.

    After calling this, any model with ``@use_experts_implementation`` will
    dispatch to ScatterMoE when ``config._experts_implementation == "scattermoe"``.

    Also patches ``get_correct_experts_implementation`` to accept ``"scattermoe"``
    as a valid value (transformers hardcodes an allowlist).
    """
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    # 1. Register the forward function in the global interface
    ALL_EXPERTS_FUNCTIONS.register("scattermoe", scattermoe_experts_forward)

    # 2. Patch the validation to accept "scattermoe"
    _original_get_correct = PreTrainedModel.get_correct_experts_implementation

    def _patched_get_correct(self_model, requested_experts: str | None) -> str:
        if requested_experts == "scattermoe":
            return "scattermoe"
        return _original_get_correct(self_model, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = _patched_get_correct


# Legacy monkeypatch approach (kept for backward compat with existing tests)
def patch_gemma4_scattermoe():
    """Monkeypatch Gemma4TextExperts.forward with ScatterMoE kernel."""
    from axolotl.integrations.kernels.constants import resolve_experts_class

    experts_cls = resolve_experts_class("gemma4_text")
    if experts_cls is None:
        raise ValueError("Could not resolve Gemma4TextExperts class")

    if hasattr(experts_cls, "_original_forward"):
        return  # already patched

    experts_cls._original_forward = experts_cls.forward
    experts_cls.forward = scattermoe_experts_forward
