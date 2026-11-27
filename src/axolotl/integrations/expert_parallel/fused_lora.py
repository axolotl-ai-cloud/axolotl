"""optmoe's fused LoRA precisions (``bf16_lora``, ``mxfp8_lora``): frozen BF16 experts plus a
PEFT ``lora_target_parameters`` adapter, trained on the fused kernels.

The kernels run on the merged weights ``W + scaling * B @ A``, held per MoE layer by one
:class:`optmoe.lora.MergedWeightCache` that every optimizer step invalidates (gradient
accumulation reuses the merge). The base stays in the experts module, FSDP-sharded like any
frozen parameter; the forward reads the unsharded tensor the ParamWrapper's FSDP unit
gathered. Adapter gradients come from rank-r GEMMs on the fused backward's pools; the base
gets none. ``expert_parallel_optmoe_fused_lora: false`` restores the dispatch mode for
adapters.
"""

from __future__ import annotations

from typing import Any

from torch.optim.optimizer import register_optimizer_step_post_hook

from axolotl.utils.logging import get_logger

from .nvfp4_lora import (
    _DOWN,
    _GATE_UP,
    _targets_expert_param,
    expert_adapter_scalings,
    is_nvfp4_tensor,
    local_expert_lora,
    map_adapter,
    register_layer,
    unregister_layers,
)

LOG = get_logger(__name__)

PRECISIONS = ("bf16_lora", "mxfp8_lora", "fp8_lora")


def enabled(cfg) -> bool:
    value = getattr(cfg, "expert_parallel_optmoe_fused_lora", None)
    return True if value is None else bool(value)


def config_reason(cfg) -> str | None:
    """Config-only reasons the fused LoRA precisions cannot run (``None`` = allowed)."""
    adapter = getattr(cfg, "adapter", None)
    if adapter != "lora":
        return (
            f"the fused LoRA path trains a single lora adapter, adapter is {adapter!r}"
        )
    if getattr(cfg, "peft_use_dora", None):
        return "the fused LoRA path does not support DoRA (peft_use_dora)"
    if getattr(cfg, "lora_dropout", 0) or 0:
        return "the fused LoRA path needs lora_dropout: 0"
    if not getattr(cfg, "use_sonicmoe", False):
        return (
            "the fused LoRA path needs use_sonicmoe: true (its ParamWrapper fast path hands "
            "the expert adapter factors to the experts module)"
        )
    targets = getattr(cfg, "lora_target_parameters", None) or ()
    if not (
        _targets_expert_param(targets, _GATE_UP)
        and _targets_expert_param(targets, _DOWN)
    ):
        return (
            "the fused LoRA path needs lora_target_parameters on experts.gate_up_proj and "
            "experts.down_proj of every layer (no per-layer targets)"
        )
    return None


def experts_reason(experts, name: str) -> str | None:
    """Model-level reasons one Experts module cannot run a fused LoRA precision."""
    for attr in (_GATE_UP, _DOWN):
        param = getattr(experts, attr, None)
        if param is None:
            return f"{name}.{attr} is missing"
        if is_nvfp4_tensor(param):
            return f"{name}.{attr} is packed NVFP4 (its LoRA precision is nvfp4_lora)"
    return None


class FusedLoraState:
    """Per-process state of the fused LoRA precisions: the shared merged-weight cache, one
    ``LoraExperts`` per MoE layer and the optimizer post-step invalidation."""

    def __init__(self):
        from optmoe.lora import MergedWeightCache

        self.cache = MergedWeightCache()
        self.layers: dict[int, Any] = {}
        self._hook = register_optimizer_step_post_hook(
            lambda *_: self.cache.invalidate()
        )

    def close(self) -> None:
        self._hook.remove()
        unregister_layers(self.layers.values())
        self.layers.clear()
        self.cache.invalidate()

    def layer(self, experts, scaling: float):
        entry = self.layers.get(id(experts))
        if entry is None:
            from optmoe.lora import LoraExperts

            entry = LoraExperts(scaling=scaling)
            self.layers[id(experts)] = entry
            register_layer("optmoe.lora", entry, self.cache)
            if len(self.layers) == 1:
                LOG.info(
                    f"optmoe fused LoRA: merged-weight cache mode {self.cache.mode}"
                )
        elif entry.scaling != float(scaling):
            raise ValueError(
                "fused LoRA: the LoRA scaling changed after the first forward"
            )
        return entry

    def prepare_layers(self, model) -> int:
        """Build and register every adapted layer, so a compiled forward creates none."""
        built = 0
        for experts, scaling in expert_adapter_scalings(model):
            self.layer(experts, scaling)
            built += 1
        return built

    def stats(self) -> dict:
        return self.cache.stats()


_STATE: FusedLoraState | None = None


def get_state() -> FusedLoraState | None:
    return _STATE


def set_state(state: FusedLoraState | None) -> None:
    global _STATE
    previous = _STATE
    _STATE = state
    if previous is not None and previous is not state:
        previous.close()


def forward(runtime, experts, hidden_states, topk_idx, topk_w, w1, w2):
    """The fused forward on base ``w1`` / ``w2`` with the adapter the SonicMoE ParamWrapper
    fast path handed over."""
    lora = getattr(experts, "_sonicmoe_lora", None) or {}
    if _GATE_UP not in lora or _DOWN not in lora:
        raise RuntimeError(
            "fused LoRA: the experts forward received no LoRA factors; the PEFT "
            "lora_target_parameters adapter on experts.gate_up_proj and experts.down_proj "
            "must be active, with use_sonicmoe: true so its ParamWrapper hands the factors "
            "to the experts module"
        )
    state = get_state()
    if state is None:
        raise RuntimeError("fused LoRA: state missing; was the optmoe runtime built?")
    wrappers = getattr(experts, "_sonicmoe_lora_wrappers", None) or {}
    a1, b1, scaling1 = lora[_GATE_UP]
    a2, b2, scaling2 = lora[_DOWN]
    if float(scaling1) != float(scaling2):
        raise ValueError(
            "fused LoRA: gate_up_proj and down_proj adapters must share one LoRA scaling, "
            f"got {scaling1} and {scaling2}"
        )
    a1, b1, num_experts, rank1 = local_expert_lora(
        a1, b1, experts, wrappers.get(_GATE_UP)
    )
    a2, b2, _, rank2 = local_expert_lora(a2, b2, experts, wrappers.get(_DOWN))
    layer = state.layer(experts, float(scaling1))
    a1, b1 = map_adapter(a1, b1, num_experts, rank1, w1.shape[-1])
    a2, b2 = map_adapter(a2, b2, num_experts, rank2, w2.shape[-1])
    return runtime.forward_lora(
        hidden_states,
        topk_idx,
        topk_w,
        a1,
        b1,
        a2,
        b2,
        layer,
        state.cache,
        w1=w1,
        w2=w2,
    )
