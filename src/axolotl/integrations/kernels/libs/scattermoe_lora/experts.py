"""ScatterMoE experts forward for the transformers ExpertsInterface.

PEFT LoRA on ``gate_up_proj`` / ``down_proj`` is fused into the
ScatterMoE Triton call via ``parallel_linear_lora``.
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


def _parallel_linear_maybe_lora(
    x,
    weight,
    top_k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    lora_tuple,
    grouped_in,
    grouped_out,
    gates=None,
):
    """Call parallel_linear or parallel_linear_lora depending on whether LoRA is active."""
    if lora_tuple is not None:
        lora_A, lora_B, scaling = lora_tuple
        return parallel_linear_lora(
            x,
            weight,
            top_k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            expert_offsets,
            lora_A,
            lora_B,
            scaling,
            grouped_in=grouped_in,
            grouped_out=grouped_out,
            gates=gates,
        )
    return parallel_linear(
        x,
        weight,
        top_k,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        grouped_in=grouped_in,
        grouped_out=grouped_out,
        gates=gates,
    )


def scattermoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE experts forward with fused-LoRA support."""
    # Assumes the standard expert layout: gate_up concatenated as [E, 2I, H],
    # gated SwiGLU, no expert bias. gpt_oss-style experts (interleaved gate/up,
    # transposed [E, H, 2I], expert bias) would be silently miscomputed by the
    # fixed transpose/chunk below, so reject rather than corrupt training.
    if (
        getattr(self, "is_transposed", False)
        or not getattr(self, "is_concatenated", True)
        or getattr(self, "has_bias", False)
        or not getattr(self, "has_gate", True)
    ):
        raise NotImplementedError(
            "scattermoe supports only concatenated, non-transposed, gated, biasless "
            "experts (qwen/mixtral/deepseek/glm/...). This model's experts use an "
            "unsupported layout; use use_sonicmoe or a built-in experts_implementation."
        )

    K = top_k_index.shape[1]

    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    # Get base weights (unwrap PEFT if needed)
    gate_up_weight = _get_base_param(self.gate_up_proj).transpose(2, 1)
    down_weight = _get_base_param(self.down_proj).transpose(2, 1)

    # Extract LoRA params if PEFT is active
    gup_lora, down_lora = None, None
    if _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    # Gate-up projection (with optional LoRA)
    gates_h = _parallel_linear_maybe_lora(
        hidden_states,
        gate_up_weight,
        K,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gup_lora,
        grouped_in=False,
        grouped_out=True,
    )
    gates, h = gates_h.chunk(2, dim=-1)
    h = self.act_fn(gates) * h

    # Down projection (with optional LoRA + routing weights)
    output = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        down_lora,
        grouped_in=True,
        grouped_out=False,
        gates=routing_weights,
    )

    return output


_SCATTERMOE_PATCHED = False


def register_scattermoe_experts():
    """Register ``"scattermoe"`` in the ExpertsInterface and the validator allowlist.

    Idempotent.
    """
    global _SCATTERMOE_PATCHED

    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    from transformers.modeling_utils import PreTrainedModel

    ALL_EXPERTS_FUNCTIONS.register("scattermoe", scattermoe_experts_forward)

    if _SCATTERMOE_PATCHED:
        return

    _original_get_correct = PreTrainedModel.get_correct_experts_implementation

    def _patched_get_correct(self_model, requested_experts: str | None) -> str:
        if requested_experts == "scattermoe":
            return "scattermoe"
        return _original_get_correct(self_model, requested_experts)

    PreTrainedModel.get_correct_experts_implementation = _patched_get_correct
    _SCATTERMOE_PATCHED = True
