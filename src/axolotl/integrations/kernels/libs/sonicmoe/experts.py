"""LoRA-aware sonicmoe experts forward for the transformers ExpertsInterface.

Dense experts materialize expert LoRA via ``MoELoRAMaterialize`` and hand the result
to upstream's ``sonicmoe_experts_forward`` through a facade. NVFP4 experts (which the
CUTLASS kernel cannot read) take the grouped dequant path in ``nvfp4_lora`` instead.
"""

from __future__ import annotations

import os

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
    # The ParamWrapper fastpath (experts_lora_fastpath) resolves the LoRA tuples itself and
    # hands them over before calling the raw base module.
    fastpath_lora = getattr(experts_module, "_sonicmoe_lora", None)
    if fastpath_lora is not None:
        w1 = experts_module.gate_up_proj
        w2 = experts_module.down_proj
        b1 = getattr(experts_module, "gate_up_proj_bias", None)
        b2 = getattr(experts_module, "down_proj_bias", None)
        return (
            w1,
            b1,
            w2,
            b2,
            fastpath_lora.get("gate_up_proj"),
            fastpath_lora.get("down_proj"),
        )

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


# sonic-moe's GEGLU epilogue is itself tanh-approximate (quack ``activation.geglu``), so these
# map onto it exactly. Without this Gemma resolves to no ``hidden_act`` and silently runs SwiGLU.
_ACT_ALIASES = {
    "gelu_pytorch_tanh": "gelu",
    "gelu_tanh": "gelu",
    "gelu_new": "gelu",
}


class _FacadeConfig:
    __slots__ = ("hidden_act",)

    def __init__(self, hidden_act: str):
        self.hidden_act = hidden_act


class _LoRAExpertsFacade:
    """Stand-in module exposing ``W_eff`` where upstream reads the raw expert weights.

    Upstream reads all of these as plain attributes, so reusing its forward beats
    duplicating the unwrap, permute and kernel call. Activation resolves via
    ``resolve_gated_activation`` because Gemma stores its own under ``hidden_activation``,
    where upstream's ``hidden_act`` lookup silently yields SwiGLU.

    Must stay a plain class: ``SimpleNamespace`` is constructed inside the compiled region
    and Dynamo cannot trace its ``__new__``, which breaks ``fullgraph=True``.
    """

    __slots__ = (
        "config",
        "down_proj",
        "down_proj_bias",
        "gate_up_proj",
        "gate_up_proj_bias",
        "has_bias",
        "has_gate",
        "is_concatenated",
        "is_transposed",
        "num_experts",
    )

    def __init__(self, experts_module, w1, b1, w2, b2):
        from .nvfp4 import resolve_gated_activation

        act = resolve_gated_activation(experts_module.config)

        self.has_gate = True
        self.gate_up_proj = w1
        self.down_proj = w2
        self.has_bias = b1 is not None or b2 is not None
        self.gate_up_proj_bias = b1
        self.down_proj_bias = b2
        self.is_transposed = getattr(experts_module, "is_transposed", False)
        self.is_concatenated = getattr(experts_module, "is_concatenated", True)
        self.num_experts = experts_module.num_experts
        self.config = _FacadeConfig(_ACT_ALIASES.get(act, act))


def sonicmoe_experts_forward_with_lora(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Sonicmoe experts forward with PEFT LoRA support.

    Dense bf16 experts use the fast sonic-moe CUTLASS kernel (LoRA materialized
    into W_eff first). NVFP4 experts, which the opaque CUTLASS kernel cannot
    read, take the grouped reference path (dequant base + fused low-rank LoRA).
    """
    from transformers.integrations.sonicmoe import sonicmoe_experts_forward

    from .nvfp4 import is_nvfp4_param

    if not getattr(self, "has_gate", True):
        raise ValueError("sonicmoe requires gated experts (has_gate=True)")
    if hidden_states.device.type != "cuda":
        raise ValueError("sonicmoe requires CUDA device")

    w1, b1, w2, b2, lora_w1, lora_w2 = _resolve_weights_and_lora(self)
    if not getattr(self, "has_bias", False):
        b1 = b2 = None

    # Unwrap FSDP2/EP DTensors to local shards. to_local() is autograd-aware:
    # backward rewraps the gradient as a DTensor.
    if isinstance(w1, torch.distributed.tensor.DTensor):
        w1 = w1.to_local()
        w2 = w2.to_local()
        b1 = b1.to_local() if b1 is not None else None
        b2 = b2.to_local() if b2 is not None else None

    # The opaque CUTLASS kernel cannot read packed FP4, so NVFP4 experts take the
    # grouped dequant path instead.
    if is_nvfp4_param(w1) or is_nvfp4_param(w2):
        return _sonicmoe_nvfp4_forward(
            self,
            hidden_states,
            top_k_index,
            top_k_weights,
            w1,
            b1,
            w2,
            b2,
            lora_w1,
            lora_w2,
        )

    # Materialize W_eff = W + scaling * (B @ A) per expert. No-op when no LoRA.
    if lora_w1 is not None:
        w1 = MoELoRAMaterialize.apply(w1, *lora_w1)
    if lora_w2 is not None:
        w2 = MoELoRAMaterialize.apply(w2, *lora_w2)

    return sonicmoe_experts_forward(
        _LoRAExpertsFacade(self, w1, b1, w2, b2),
        hidden_states,
        top_k_index,
        top_k_weights,
    )


def _select_nvfp4_backend(w1, w2) -> str:
    """``fp4_cute`` (SM100 in-kernel W4A4) when available and dims align, else
    ``dequant``. ``AXOLOTL_SONICMOE_NVFP4_BACKEND`` forces a backend."""
    override = os.environ.get("AXOLOTL_SONICMOE_NVFP4_BACKEND")
    if override:
        return override

    from .fp4_cute import fp4_cute_available
    from .fp4_cute_ops import fp4_cute_dims_ok
    from .nvfp4 import is_nvfp4_param

    if (
        is_nvfp4_param(w1)
        and is_nvfp4_param(w2)
        and fp4_cute_available()
        and fp4_cute_dims_ok(w1, w2)
    ):
        return "fp4_cute"
    return "dequant"


def _sonicmoe_nvfp4_forward(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    w1: torch.Tensor,
    b1,
    w2: torch.Tensor,
    b2,
    lora_w1,
    lora_w2,
) -> torch.Tensor:
    """NVFP4 experts forward via the grouped path (base frozen).

    LoRA is applied as a fused low-rank delta at the grouped-token level (no
    base-weight gradient). On SM100/SM110 the base stays packed and runs the
    in-kernel W4A4 grouped GEMM (``fp4_cute``); elsewhere it is dequantized to
    dense per matmul (``dequant``).
    """
    from .nvfp4 import resolve_gated_activation
    from .nvfp4_lora import grouped_moe_reference_forward

    if getattr(self, "is_transposed", False):
        raise NotImplementedError(
            "sonicmoe NVFP4 path supports the [E, 2*I, H] / [E, H, I] layout only "
            "(is_transposed=False)"
        )

    # Expert parallelism is not finished for this path: the base is EP-sharded to
    # local experts while routing/LoRA still use global expert ids, and the local
    # token dispatch is not implemented. Fail loudly instead of computing garbage.
    if getattr(self, "num_experts_global", self.num_experts) != self.num_experts:
        raise NotImplementedError(
            "sonicmoe NVFP4 path does not support expert parallelism yet "
            "(EP-sharded base with global-id routing/LoRA is unfinished and untested)"
        )

    lora1 = (lora_w1[0], lora_w1[1]) if lora_w1 is not None else None
    lora2 = (lora_w2[0], lora_w2[1]) if lora_w2 is not None else None
    scaling1 = lora_w1[2] if lora_w1 is not None else 1.0
    scaling2 = lora_w2[2] if lora_w2 is not None else 1.0

    return grouped_moe_reference_forward(
        hidden_states,
        top_k_index,
        top_k_weights,
        w1,
        b1,
        w2,
        b2,
        lora1,
        lora2,
        self.num_experts,
        act=resolve_gated_activation(self.config),
        backend=_select_nvfp4_backend(w1, w2),
        limit=getattr(self, "limit", None),
        concat=getattr(self, "is_concatenated", True),
        scaling1=scaling1,
        scaling2=scaling2,
    )


def register_sonicmoe_experts() -> None:
    """Register the LoRA-aware ``"sonicmoe"`` forward, overriding upstream. Idempotent."""
    from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

    ALL_EXPERTS_FUNCTIONS.register("sonicmoe", sonicmoe_experts_forward_with_lora)

    # Route PEFT target_parameters expert LoRA past the parametrization merge (which cannot
    # run on quantized bases) to the fused low-rank path.
    try:
        from .experts_lora_fastpath import patch_paramwrapper_sonicmoe_fastpath

        patch_paramwrapper_sonicmoe_fastpath()
    except (ImportError, AttributeError):
        pass


# Re-export utilities for tests / external callers.
__all__ = [
    "sonicmoe_experts_forward_with_lora",
    "register_sonicmoe_experts",
    "materialize_expert_lora",
]
