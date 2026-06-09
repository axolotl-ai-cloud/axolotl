"""ScatterMoE experts forward for the transformers ExpertsInterface.

PEFT LoRA on ``gate_up_proj`` / ``down_proj`` is fused into the
ScatterMoE Triton call via ``parallel_linear_lora``.
"""

import torch

from .mx_weights import selective_mx_weights_fwd, selective_nvfp4_weights_fwd
from .parallel_experts import flatten_sort_count, parallel_linear
from .parallel_linear_lora import get_lora_params_from_wrapper, parallel_linear_lora
from .selective_dequant import (
    get_active_experts,
    is_mxfp4_param,
    is_nvfp4_param,
    remap_expert_indices,
    selective_lora_weights,
)


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
    expert_biases=None,
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
            expert_biases=expert_biases,
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
        expert_biases=expert_biases,
        grouped_in=grouped_in,
        grouped_out=grouped_out,
        gates=gates,
    )


def _prepare_weights_and_lora(
    gu_param,
    dn_param,
    sorted_expert_idxs,
    expert_offsets,
    num_experts,
    gup_lora,
    down_lora,
    dtype,
):
    """Resolve the gate_up / down weights (+ routing + LoRA) for the grouped GEMM.

    For an MXFP4 or NVFP4 base with LoRA on both projections, keep the weights packed
    (4-bit) and route through the fused kernel: select the active experts and remap the
    routing / LoRA to that compact set (dequant happens inside the kernel K-loop; NVFP4
    reuses the MXWeights container with linear E4M3 block scales). Otherwise return bf16
    ``[E, in, out]`` weights (dequantizing FP4 explicitly when LoRA is absent, since the
    fused kernel is LoRA-only and the FP4 tensors have no transpose).

    Returns ``(gate_up_weight, down_weight, sorted_expert_idxs, expert_offsets,
    gup_lora, down_lora)``.
    """
    is_mx, is_nv = is_mxfp4_param(gu_param), is_nvfp4_param(gu_param)
    if (is_mx or is_nv) and gup_lora is not None and down_lora is not None:
        select = selective_mx_weights_fwd if is_mx else selective_nvfp4_weights_fwd
        active = get_active_experts(sorted_expert_idxs, num_experts)
        sorted_expert_idxs, expert_offsets = remap_expert_indices(
            sorted_expert_idxs, expert_offsets, active, num_experts
        )
        gate_up_weight = select(gu_param, active)
        down_weight = select(dn_param, active)
        gup_lora = (
            *selective_lora_weights(gup_lora[0], gup_lora[1], active, num_experts),
            gup_lora[2],
        )
        down_lora = (
            *selective_lora_weights(down_lora[0], down_lora[1], active, num_experts),
            down_lora[2],
        )
    elif is_mx or is_nv:
        # quantized base without LoRA: the fused kernel is LoRA-only and the FP4
        # tensors have no transpose, so dequantize to bf16 here.
        gate_up_weight = gu_param.dequantize(dtype).transpose(2, 1)
        down_weight = dn_param.dequantize(dtype).transpose(2, 1)
    else:
        gate_up_weight = gu_param.transpose(2, 1)  # bf16 [E, out, in] -> [E, in, out]
        down_weight = dn_param.transpose(2, 1)
    return (
        gate_up_weight,
        down_weight,
        sorted_expert_idxs,
        expert_offsets,
        gup_lora,
        down_lora,
    )


def scattermoe_supports_layout(self) -> bool:
    """True iff this experts module uses the standard layout scattermoe handles:
    gate_up concatenated as [E, 2I, H], gated SwiGLU, no expert bias. gpt_oss-style
    experts (interleaved gate/up, transposed [E, H, 2I], expert bias) return False."""
    return not (
        getattr(self, "is_transposed", False)
        or not getattr(self, "is_concatenated", True)
        or getattr(self, "has_bias", False)
        or not getattr(self, "has_gate", True)
    )


def _check_supported_layout(self):
    """Reject expert layouts the fixed transpose/chunk below would miscompute."""
    # gpt_oss-style experts (interleaved gate/up, transposed [E, H, 2I], expert
    # bias) would be silently miscomputed by the fixed transpose/chunk below, so
    # reject rather than corrupt training.
    if not scattermoe_supports_layout(self):
        raise NotImplementedError(
            "scattermoe supports only concatenated, non-transposed, gated, biasless "
            "experts (qwen/mixtral/deepseek/glm/...). This model's experts use an "
            "unsupported layout; use use_sonicmoe or a built-in experts_implementation."
        )


def _is_gptoss_layout(self) -> bool:
    """gpt_oss expert layout: transposed [E, H, 2I] weights, interleaved gate/up,
    per-expert bias, clamped sigmoid-GLU. Handled by the Triton path below."""
    return (
        getattr(self, "is_transposed", False)
        and not getattr(self, "is_concatenated", True)
        and getattr(self, "has_bias", False)
        and getattr(self, "has_gate", True)
        and hasattr(self, "gate_up_proj_bias")
        and hasattr(self, "down_proj_bias")
    )


def _gptoss_glu(gate_up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """gpt_oss clamped sigmoid-GLU over interleaved gate/up columns (matches
    ``GptOssExperts._apply_gate``)."""
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    return (up + 1) * glu


def _scattermoe_gptoss_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE forward for gpt_oss-style experts (transposed/interleaved/biased).

    gpt_oss stores weights already in ``[E, in, out]`` form, so no transpose; gate/up
    are interleaved (``[..., ::2]`` / ``[..., 1::2]``); per-expert bias is folded into
    the grouped GEMM; the activation is the clamped sigmoid-GLU. LoRA fuses exactly as
    in the standard path (in/out dims match), and the down bias is added per row before
    the routing-weight combine.
    """
    K = top_k_index.shape[1]
    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    gate_up_weight = _get_base_param(
        self.gate_up_proj
    )  # [E, H, 2I], already [E,in,out]
    down_weight = _get_base_param(self.down_proj)  # [E, I, H]
    gate_up_bias = _get_base_param(self.gate_up_proj_bias)  # [E, 2I]
    down_bias = _get_base_param(self.down_proj_bias)  # [E, H]

    gup_lora, down_lora = None, None
    if _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    gate_up = _parallel_linear_maybe_lora(
        hidden_states,
        gate_up_weight,
        K,
        sorted_expert_idxs,
        sorted_scattered_idxs,
        expert_offsets,
        gup_lora,
        grouped_in=False,
        grouped_out=True,
        expert_biases=gate_up_bias,
    )
    h = _gptoss_glu(gate_up, getattr(self, "alpha", 1.702), getattr(self, "limit", 7.0))

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
        expert_biases=down_bias,
    )
    return output


def scattermoe_experts_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE experts forward with fused-LoRA support."""
    if not scattermoe_supports_layout(self):
        if _is_gptoss_layout(self):
            return _scattermoe_gptoss_forward(
                self, hidden_states, top_k_index, top_k_weights
            )
        _check_supported_layout(self)  # raises for any other unsupported layout

    K = top_k_index.shape[1]

    routing_weights = top_k_weights.to(hidden_states.dtype)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count(
        top_k_index, num_experts=self.num_experts
    )

    # Extract LoRA params if PEFT is active
    gup_lora, down_lora = None, None
    if _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    (
        gate_up_weight,
        down_weight,
        sorted_expert_idxs,
        expert_offsets,
        gup_lora,
        down_lora,
    ) = _prepare_weights_and_lora(
        _get_base_param(self.gate_up_proj),
        _get_base_param(self.down_proj),
        sorted_expert_idxs,
        expert_offsets,
        self.num_experts,
        gup_lora,
        down_lora,
        hidden_states.dtype,
    )

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


def scattermoe_experts_forward_ep(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """ScatterMoE experts forward for the DeepEP local path, skipping EP sentinels.

    After DeepEP dispatch ``top_k_index`` holds local expert ids in ``[0, E_local)``
    for slots this rank owns and ``-1`` for slots routed to remote ranks. Rather than
    map sentinels to expert 0 / weight 0 and run the full grouped GEMM over all
    ``N*K`` rows (compute-and-mask), drop the sentinel rows so only the valid routed
    rows hit the GEMM + per-row LoRA. Output matches the masked path since sentinel
    slots carry weight 0.

    Runs both projections fully grouped (the sentinel-compacted routing breaks the
    ``L_scattered == X.rows * k`` fan-out contract of the scattered path), with the
    weighted token-combine done via ``index_add_``.
    """
    _check_supported_layout(self)

    N = hidden_states.size(0)
    K = top_k_index.shape[1]
    E_local = self.num_experts

    idx_flat = top_k_index.reshape(-1)
    valid = idx_flat >= 0
    e_v = idx_flat[valid].to(torch.long)
    if e_v.numel() == 0:
        return torch.zeros_like(hidden_states)

    tok = torch.arange(N, device=hidden_states.device).repeat_interleave(K)
    tok_v = tok[valid]
    w_v = top_k_weights.reshape(-1)[valid].to(hidden_states.dtype)

    # Sort valid rows by expert so each expert's rows are contiguous (grouped layout).
    se, order = torch.sort(e_v)
    tok_sorted = tok_v[order].to(torch.int32)
    w_sorted = w_v[order]
    expert_offsets = torch.bincount(e_v, minlength=E_local).cumsum(-1)
    M = se.size(0)
    ss = torch.arange(M, device=hidden_states.device, dtype=torch.int32)

    gup_lora, down_lora = None, None
    if _has_peft_wrapper(self):
        _, gup_lora, down_lora = _unwrap_experts_lora(self)

    gate_up_weight, down_weight, se, expert_offsets, gup_lora, down_lora = (
        _prepare_weights_and_lora(
            _get_base_param(self.gate_up_proj),
            _get_base_param(self.down_proj),
            se,
            expert_offsets,
            E_local,
            gup_lora,
            down_lora,
            hidden_states.dtype,
        )
    )

    # Pre-gather token rows into expert-grouped order; both projections run grouped.
    grouped_x = hidden_states.index_select(0, tok_sorted.to(torch.long))

    gates_h = _parallel_linear_maybe_lora(
        grouped_x,
        gate_up_weight,
        1,
        se,
        ss,
        expert_offsets,
        gup_lora,
        grouped_in=True,
        grouped_out=True,
    )
    gates, h = gates_h.chunk(2, dim=-1)
    h = self.act_fn(gates) * h

    down_out = _parallel_linear_maybe_lora(
        h,
        down_weight,
        1,
        se,
        ss,
        expert_offsets,
        down_lora,
        grouped_in=True,
        grouped_out=True,
    )
    down_out = down_out * w_sorted.unsqueeze(-1)

    output = hidden_states.new_zeros((N, down_out.size(-1)))
    output.index_add_(0, tok_sorted.to(torch.long), down_out)
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
