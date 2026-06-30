"""LoRA-aware sonicmoe experts forward for the transformers ExpertsInterface.

Wraps upstream ``_sonicmoe_wrapper`` and materializes expert LoRA via
``MoELoRAMaterialize`` before the CUTLASS call.
"""

from __future__ import annotations

import functools

import torch

from .lora import (
    MoELoRAMaterialize,
    get_lora_params_from_wrapper,
    has_lora,
    materialize_expert_lora,
    unwrap_experts_lora,
)


def _maybe_unwrap_param_wrapper(param):
    """Return ``(base_tensor, lora_params_or_None)`` for a PEFT-wrapped Parameter."""
    try:
        from peft.tuners.param_wrapper import ParamWrapper
    except ImportError:
        return param, None

    if not isinstance(param, ParamWrapper):
        return param, None

    base = param.original_parameter
    lora_A, lora_B, scaling = get_lora_params_from_wrapper(param)
    if lora_A is None:
        return base, None
    return base, (lora_A, lora_B, scaling)


def _resolve_weights_and_lora(experts_module):
    """Resolve raw expert weights/biases + optional LoRA tuples.

    Handles both PEFT layouts: module-level wrap (walked via ``unwrap_experts_lora``)
    and per-parameter ``ParamWrapper``. No layout permute applied.
    """
    if has_lora(experts_module):
        base_experts, lora_dict = unwrap_experts_lora(experts_module)
        w1 = base_experts.gate_up_proj
        w2 = base_experts.down_proj
        b1 = getattr(base_experts, "gate_up_proj_bias", None)
        b2 = getattr(base_experts, "down_proj_bias", None)
        return w1, b1, w2, b2, lora_dict.get("gate_up_proj"), lora_dict.get("down_proj")

    w1, lora_w1 = _maybe_unwrap_param_wrapper(experts_module.gate_up_proj)
    w2, lora_w2 = _maybe_unwrap_param_wrapper(experts_module.down_proj)
    b1 = getattr(experts_module, "gate_up_proj_bias", None)
    b2 = getattr(experts_module, "down_proj_bias", None)
    return w1, b1, w2, b2, lora_w1, lora_w2


@functools.lru_cache(maxsize=1)
def _sonicmoe_kernel_supported() -> bool:
    """The kernels-community/sonic-moe CUTLASS GEMM fails to compile on sm_120
    (Blackwell): its bundled quack ``GemmSm120`` predates the ``concat_layout`` arg
    the dispatcher passes. Gate it off there so standard-layout models fall back to
    the vendored scattermoe Triton path. Drop this once a fixed sonic-moe ships."""
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_capability()[0] < 12


def sonicmoe_experts_forward_with_lora(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Sonicmoe experts forward with PEFT LoRA materialization.

    On devices where the sonic-moe kernel can't run (sm_120), standard-layout
    experts transparently use the vendored scattermoe Triton path instead.
    """
    from ..scattermoe_lora.experts import (
        scattermoe_experts_forward,
        scattermoe_supports_layout,
    )

    # Validate the expert layout / device BEFORE the sm120 fallback dispatch: a non-CUDA input or
    # ungated experts are invalid for both paths, and the fallback (entered on sm120 where the
    # sonic kernel can't run) would otherwise hit a CPU tensor and raise an opaque AttributeError.
    if not getattr(self, "has_gate", True):
        raise ValueError("sonicmoe requires gated experts (has_gate=True)")
    if hidden_states.device.type != "cuda":
        raise ValueError("sonicmoe requires CUDA device")

    if not _sonicmoe_kernel_supported() and scattermoe_supports_layout(self):
        return scattermoe_experts_forward(
            self, hidden_states, top_k_index, top_k_weights
        )

    from transformers.integrations.sonicmoe import _sonicmoe_wrapper

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)

    # sonic-moe requires int32 token indices sorted ascending.
    token_idx = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_top_k)
        .reshape(-1)
        .int()
    )
    router_scores = top_k_weights.reshape(-1).to(hidden_states.dtype)
    expert_ids = top_k_index.reshape(-1).int()

    w1, b1, w2, b2, lora_w1, lora_w2 = _resolve_weights_and_lora(self)
    if not getattr(self, "has_bias", False):
        b1 = b2 = None

    # sonic-moe takes raw CUTLASS pointers, so unwrap FSDP2/EP DTensors to local shards first.
    # to_local() is autograd-aware: backward rewraps the gradient as a DTensor.
    if isinstance(w1, torch.distributed.tensor.DTensor):
        w1 = w1.to_local()
        w2 = w2.to_local()
        b1 = b1.to_local() if b1 is not None else None
        b2 = b2.to_local() if b2 is not None else None

    # Materialize W_eff = W + scaling * (B @ A) per expert. No-op when no LoRA.
    if lora_w1 is not None:
        w1 = MoELoRAMaterialize.apply(w1, *lora_w1)
    if lora_w2 is not None:
        w2 = MoELoRAMaterialize.apply(w2, *lora_w2)

    # Match upstream layout expectations:
    #   is_transposed=False: gate_up [E, 2*I, H] / down [E, H, I] -> permute(1, 2, 0)
    #   is_transposed=True:  gate_up [E, H, 2*I] / down [E, I, H] -> permute(2, 1, 0)
    perm = (2, 1, 0) if getattr(self, "is_transposed", False) else (1, 2, 0)
    w1 = w1.permute(*perm)
    w2 = w2.permute(*perm)

    act_name = getattr(self.config, "hidden_act", "silu").lower()

    return _sonicmoe_wrapper(
        hidden_states=hidden_states,
        router_scores=router_scores,
        expert_ids=expert_ids,
        token_idx=token_idx,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        act_name=act_name,
        num_experts=self.num_experts,
        concat_layout=getattr(self, "is_concatenated", True),
        is_inference_mode_enabled=not torch.is_grad_enabled(),
    )


def register_sonicmoe_experts() -> None:
    """Register the LoRA-aware ``"sonicmoe"`` forward, overriding upstream. Idempotent."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register("sonicmoe", sonicmoe_experts_forward_with_lora)


# Re-export utilities for tests / external callers.
__all__ = [
    "sonicmoe_experts_forward_with_lora",
    "register_sonicmoe_experts",
    "materialize_expert_lora",
]
