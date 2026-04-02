"""
SonicMoE patching for SparseMoeBlock forward pass.

Monkeypatches the SparseMoeBlock class for a given model type to use
SonicMoE's optimized kernels. Two forward paths are supported:

1. **General routing path** (routing_fn is not None):
   Uses a custom routing function + ``moe_general_routing_inputs``.
   Suitable for models with non-standard routing (softmax->topk, sigmoid->topk).

2. **Fused topk->softmax path** (routing_fn is None):
   Uses ``moe_TC_softmax_topk_layer`` which fuses routing + expert computation.
   Suitable for models with simple topk->softmax routing.

Weight format conversion (interleave/deinterleave) is handled by the
WeightConverter system, so the forward assumes weights are already in
interleaved format.

Shared experts are handled generically: if the block has a ``shared_expert``
or ``shared_experts`` attribute, its output is computed alongside the routed
experts and added to the final output. An optional ``shared_expert_gate``
applies sigmoid gating to the shared expert contribution.
"""

import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.constants import resolve_moe_block_classes
from axolotl.utils.logging import get_logger

from .lora import (
    has_lora,
    materialize_expert_lora,
    unwrap_experts_lora,
    unwrap_gate_lora,
)

LOG = get_logger(__name__)


def _get_expert_weights(experts_module):
    """Extract expert weights, applying LoRA materialization if PEFT is active.

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


def _fix_qwen3_5_moe_text_weight_renaming(model_type: str, base_model_type: str):
    """Strip qwen3_5_moe_text WeightRenaming in VLM mode to preserve custom loaders."""
    if model_type != "qwen3_5_moe_text" or base_model_type == "qwen3_5_moe_text":
        return

    try:
        from transformers.conversion_mapping import (
            get_checkpoint_conversion_mapping,
            register_checkpoint_conversion_mapping,
        )
        from transformers.core_model_loading import WeightRenaming
    except ImportError:
        return

    text_mapping = get_checkpoint_conversion_mapping(model_type)
    if text_mapping and isinstance(text_mapping[0], WeightRenaming):
        text_mapping.pop(0)
        register_checkpoint_conversion_mapping(model_type, text_mapping, overwrite=True)
        LOG.info("Stripped qwen3_5_moe_text WeightRenaming for VLM mode")


def patch_sonicmoe(
    model_type: str,
    torch_compile: bool = False,
    base_model_type: str | None = None,
):
    """Patch SparseMoeBlock for SonicMoE support."""
    from .routing import get_model_moe_config
    from .weight_converter import register_sonicmoe_weight_converter

    _fix_qwen3_5_moe_text_weight_renaming(model_type, base_model_type or model_type)

    routing_fn, activation, router_attr = get_model_moe_config(model_type)

    if torch_compile and routing_fn is not None:
        routing_fn = _try_compile_routing(routing_fn)

    for moe_cls in resolve_moe_block_classes(model_type):
        _patch_forward(moe_cls, routing_fn, activation, router_attr)
    register_sonicmoe_weight_converter(model_type)


def _try_compile_routing(routing_fn):
    """Attempt to torch.compile the routing function, fall back to eager on failure."""
    try:
        compiled_fn = torch.compile(routing_fn, mode="reduce-overhead", dynamic=False)
        LOG.info(f"torch.compile enabled for routing function: {routing_fn.__name__}")
        return compiled_fn
    except Exception as exc:  # pylint: disable=broad-except
        LOG.warning(
            f"torch.compile failed for routing function {routing_fn.__name__}, "
            f"falling back to eager: {exc}"
        )
        return routing_fn


def _patch_forward(moe_cls, routing_fn, activation, router_attr):
    """Monkeypatch the SparseMoeBlock class with a SonicMoE forward.

    The patched forward handles shared experts generically: if
    ``self.shared_expert`` or ``self.shared_experts`` exists, it is computed
    and added to the routed output. If ``self.shared_expert_gate`` also exists,
    it applies sigmoid gating to the shared expert contribution (as in qwen2_moe).

    Args:
        moe_cls: The SparseMoeBlock class to patch.
        routing_fn: Routing function (e.g. softmax_topk_routing), or None
            for the fused moe_TC_softmax_topk_layer path.
        activation: SonicMoE ActivationType enum value.
        router_attr: Name of the router module attribute on the MoE block.
    """
    if hasattr(moe_cls, "_original_forward"):
        LOG.info(f"{moe_cls.__name__}.forward already patched with SonicMoE, skipping")
        return

    original_forward = moe_cls.forward

    if routing_fn is not None:
        _make_general_forward(moe_cls, routing_fn, activation)
    else:
        _make_fused_forward(moe_cls, activation, router_attr)

    moe_cls._original_forward = original_forward
    LOG.info(f"Patched {moe_cls.__name__}.forward with SonicMoE implementation")


def _make_general_forward(moe_cls, routing_fn, activation):
    """Create forward using routing_fn + moe_general_routing_inputs."""

    def sonicmoe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from sonicmoe import moe_general_routing_inputs

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Shared expert (computed early, matching original model ordering)
        shared_expert_output = _compute_shared_expert(self, hidden_states_flat)

        # Routing
        router_scores, token_indices, expert_indices, _router_logits = routing_fn(
            hidden_states_flat, self
        )

        # Unwrap PEFT + optional LoRA materialization, then permute to SonicMoE layout
        gate_up_weight, down_weight = _get_expert_weights(self.experts)
        gate_up_weight = gate_up_weight.to(hidden_states_flat.dtype)
        down_weight = down_weight.to(hidden_states_flat.dtype)
        E = gate_up_weight.shape[-1]

        output, _ = moe_general_routing_inputs(
            hidden_states_flat,
            router_scores,
            token_indices,
            expert_indices,
            gate_up_weight,
            None,  # b1 (no gate/up bias)
            down_weight,
            None,  # b2 (no down bias)
            E,
            torch.cuda.current_stream().cuda_stream,
            activation,
            False,  # is_inference_mode
        )

        # Add shared expert contribution if present
        if shared_expert_output is not None:
            if hasattr(self, "shared_expert_gate"):
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states_flat))
                    * shared_expert_output
                )
            output = output + shared_expert_output

        return output.view(batch_size, sequence_length, hidden_dim)

    moe_cls.forward = sonicmoe_forward


def _make_fused_forward(moe_cls, activation, router_attr):
    """Create forward using moe_TC_softmax_topk_layer (topk -> softmax)."""

    def sonicmoe_fused_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from sonicmoe import moe_TC_softmax_topk_layer

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Shared expert (computed early, matching original model ordering)
        shared_expert_output = _compute_shared_expert(self, hidden_states_flat)

        # Unwrap router for attribute access + optional LoRA delta
        raw_router = getattr(self, router_attr)
        base_router, router_weight, router_lora_delta = unwrap_gate_lora(raw_router)
        if router_lora_delta is not None:
            # Materialize local tensor to avoid DTensor + Tensor add under FSDP
            if hasattr(router_weight, "to_local"):
                router_weight = router_weight.to_local()
            effective_router_weight = router_weight + router_lora_delta
        else:
            effective_router_weight = router_weight

        # Unwrap PEFT + optional LoRA materialization, then permute to SonicMoE layout
        gate_up_weight, down_weight = _get_expert_weights(self.experts)
        gate_up_weight = gate_up_weight.to(hidden_states_flat.dtype)
        down_weight = down_weight.to(hidden_states_flat.dtype)

        output, _router_logits, _expert_freq = moe_TC_softmax_topk_layer(
            hidden_states_flat,
            effective_router_weight,
            gate_up_weight,
            None,  # b1 (no gate/up bias)
            down_weight,
            None,  # b2 (no down bias)
            base_router.top_k,
            torch.cuda.current_stream().cuda_stream,
            activation,
            False,  # is_inference_mode
        )

        # Add shared expert contribution if present
        if shared_expert_output is not None:
            if hasattr(self, "shared_expert_gate"):
                shared_expert_output = (
                    F.sigmoid(self.shared_expert_gate(hidden_states_flat))
                    * shared_expert_output
                )
            output = output + shared_expert_output

        return output.view(batch_size, sequence_length, hidden_dim)

    moe_cls.forward = sonicmoe_fused_forward


def _compute_shared_expert(moe_block, hidden_states_flat):
    """Compute shared expert output if the block has one.

    Handles singular (qwen2_moe: ``shared_expert``), plural
    (glm_moe_dsa/deepseek_v3: ``shared_experts``), and MLP
    (hunyuan_v1_moe: ``shared_mlp``) attribute names.
    """
    shared_expert = (
        getattr(moe_block, "shared_expert", None)
        or getattr(moe_block, "shared_experts", None)
        or getattr(moe_block, "shared_mlp", None)
    )
    if shared_expert is not None:
        return shared_expert(hidden_states_flat)
    return None
