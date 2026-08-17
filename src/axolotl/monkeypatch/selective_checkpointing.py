"""Eager selective activation checkpointing (SAC): save chosen ops instead of recomputing.

Uses ``torch.utils.checkpoint.create_selective_checkpoint_contexts`` (no torch.compile).
Ops are matched at the dispatcher level, so custom kernels that are not registered as
torch ops are simply recomputed as usual — only ops we want to *save* need to be
dispatcher-visible.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ATTENTION_GROUP = "attention"

_ATEN_ATTENTION_PACKETS = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
)

# regions to observe before warning that no op ever matched the save policy
_NO_MATCH_WARN_REGIONS = 64


def _aten_attention_ops() -> set:
    ops = set()
    for packet_name in _ATEN_ATTENTION_PACKETS:
        packet = getattr(torch.ops.aten, packet_name, None)
        if packet is not None:
            ops.add(packet.default)
    return ops


def _op_name(op: Any) -> str:
    try:
        return op.name()
    except (AttributeError, TypeError):
        return getattr(op, "__name__", str(op))


def _is_flash_attention_forward(name: str) -> bool:
    lowered = name.lower()
    if "flash_attn" not in lowered and "flash_attention" not in lowered:
        return False
    if "backward" in lowered or "bwd" in lowered:
        return False
    return True


# a bounded LEFT context is what defines SWA; a bounded right side alone can
# just encode causality, so only window_size_left is inspected
_WINDOW_ARG_NAME = "window_size_left"
_window_arg_cache: dict[Any, int | None] = {}


def _window_arg_index(op: Any) -> int | None:
    """Schema position of the sliding-window arg, or None if the op has none."""
    if op in _window_arg_cache:
        return _window_arg_cache[op]
    index: int | None = None
    schema = getattr(op, "_schema", None)
    if schema is not None:
        for i, arg in enumerate(schema.arguments):
            if arg.name == _WINDOW_ARG_NAME:
                index = i
                break
    _window_arg_cache[op] = index
    return index


def _is_sliding_window_call(op: Any, args: tuple, kwargs: dict) -> bool:
    """True when a flash-attention call is bounded by a sliding window.

    flash-attn uses -1 for an unbounded side, so a non-negative window arg
    means SWA. SDPA carries the window only inside the attention mask, so
    hybrid models running through SDPA cannot be discriminated here.
    """
    val = kwargs.get(_WINDOW_ARG_NAME)
    if isinstance(val, int):
        return val >= 0
    index = _window_arg_index(op)
    if index is not None and index < len(args) and isinstance(args[index], int):
        return args[index] >= 0
    return False


DEFAULT_RECOMPUTE_LAYER_TYPES = ("sliding_attention", "chunked_attention")


class SacPolicyState:
    """Bookkeeping shared across checkpoint regions for logging/diagnostics."""

    def __init__(self) -> None:
        self.saved_op_names: set[str] = set()
        self.sliding_op_names: set[str] = set()
        self.regions_seen: int = 0
        self.warned_no_match: bool = False
        # published by decoder-layer hooks; read by the policy. Hooks fire again
        # during checkpoint recompute (on the autograd thread), so forward and
        # replay see the same value.
        self.current_layer_type: str | None = None


def _layer_attention_type(module) -> str | None:
    for obj in (
        module,
        getattr(module, "self_attn", None),
        getattr(module, "attention", None),
    ):
        if obj is None:
            continue
        layer_type = getattr(obj, "layer_type", None)
        if isinstance(layer_type, str):
            return layer_type
        if getattr(obj, "is_sliding", None) is True:
            return "sliding_attention"
    return None


def install_layer_type_hooks(model, state: SacPolicyState) -> int:
    """Publish each checkpointed decoder layer's attention type while it runs.

    Lets the policy skip saving sliding/chunked-window attention in hybrid
    models even under SDPA, where the window lives in the mask and cannot be
    read off the op's arguments.
    """
    from transformers import GradientCheckpointingLayer

    hooked = 0
    if not hasattr(model, "modules"):
        return hooked
    for module in model.modules():
        if not isinstance(module, GradientCheckpointingLayer):
            continue
        layer_type = _layer_attention_type(module)
        if layer_type is None:
            continue

        def _set(mod, args, kwargs=None, _lt=layer_type):
            state.current_layer_type = _lt

        def _clear(mod, args, output):
            state.current_layer_type = None

        module.register_forward_pre_hook(_set)
        module.register_forward_hook(_clear, always_call=True)
        hooked += 1
    if hooked:
        LOG.info(
            f"selective_checkpointing: layer-type hooks on {hooked} decoder layers "
            "(sliding/chunked-window attention will be recomputed)"
        )
    return hooked


def build_sac_policy(
    save: list[str] | None = None,
    state: SacPolicyState | None = None,
    save_sliding_window: bool = False,
    recompute_layer_types: list[str] | None = None,
) -> Callable:
    """Build an eager SAC policy_fn: MUST_SAVE for matching ops, PREFER_RECOMPUTE otherwise.

    ``save`` entries are either the ``"attention"`` group or substrings matched
    against the qualified op name (e.g. ``"aten::mm"``). Unless
    ``save_sliding_window`` is set, hybrid-model attention calls bounded by a
    sliding window keep being recomputed — SWA is cheap and not worth the saved
    memory; only full-attention calls are saved.
    """
    save = save or [ATTENTION_GROUP]
    state = state or SacPolicyState()
    skip_layer_types = (
        set()
        if save_sliding_window
        else set(
            DEFAULT_RECOMPUTE_LAYER_TYPES
            if recompute_layer_types is None
            else recompute_layer_types
        )
    )

    exact_ops: set = set()
    substrings: list[str] = []
    match_flash_attention = False
    for spec in save:
        if spec == ATTENTION_GROUP:
            exact_ops |= _aten_attention_ops()
            match_flash_attention = True
        else:
            substrings.append(spec)

    def _matches(op: Any) -> bool:
        if op in exact_ops:
            return True
        name = _op_name(op)
        if match_flash_attention and _is_flash_attention_forward(name):
            return True
        return any(sub in name for sub in substrings)

    def policy_fn(ctx, op, *args, **kwargs):  # pylint: disable=unused-argument
        if _matches(op):
            name = _op_name(op)
            if state.current_layer_type in skip_layer_types:
                if name not in state.sliding_op_names:
                    state.sliding_op_names.add(name)
                    LOG.info(
                        f"selective_checkpointing: recomputing `{name}` in "
                        f"{state.current_layer_type} layers"
                    )
                return CheckpointPolicy.PREFER_RECOMPUTE
            if not save_sliding_window and _is_sliding_window_call(op, args, kwargs):
                if name not in state.sliding_op_names:
                    state.sliding_op_names.add(name)
                    LOG.info(
                        f"selective_checkpointing: recomputing sliding-window "
                        f"calls of `{name}` (save_sliding_window: false)"
                    )
                return CheckpointPolicy.PREFER_RECOMPUTE
            if name not in state.saved_op_names:
                state.saved_op_names.add(name)
                LOG.info(
                    f"selective_checkpointing: saving `{name}` "
                    "(backward will not recompute it)"
                )
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def build_sac_context_fn(
    save: list[str] | None = None,
    save_sliding_window: bool = False,
    state: SacPolicyState | None = None,
    recompute_layer_types: list[str] | None = None,
) -> Callable:
    """Return a ``context_fn`` for ``torch.utils.checkpoint.checkpoint``."""
    state = state or SacPolicyState()
    policy_fn = build_sac_policy(
        save, state, save_sliding_window, recompute_layer_types
    )

    def context_fn():
        state.regions_seen += 1
        if (
            not state.saved_op_names
            and not state.warned_no_match
            and state.regions_seen >= _NO_MATCH_WARN_REGIONS
        ):
            state.warned_no_match = True
            LOG.warning(
                f"selective_checkpointing: no op matched the save policy after "
                f"{state.regions_seen} checkpoint regions. Your attention "
                "implementation may not be dispatcher-visible (e.g. a custom "
                "kernel not registered via torch.library); everything is being "
                "recomputed as with plain gradient checkpointing."
            )
        return create_selective_checkpoint_contexts(policy_fn)

    return context_fn


def apply_selective_checkpointing(
    model,
    save: list[str] | None = None,
    save_sliding_window: bool = False,
    recompute_layer_types: list[str] | None = None,
    offload: bool = False,
) -> None:
    """Wrap ``model.gradient_checkpointing_enable`` to inject the SAC ``context_fn``.

    Wrapping the instance method covers every enable call site (axolotl's model
    loader, HF Trainer at train() time, PEFT's kbit prep) without placing a
    non-serializable callable into ``TrainingArguments``.
    """
    if getattr(model.gradient_checkpointing_enable, "_axolotl_sac", False):
        return

    state = SacPolicyState()
    skip_layer_types = (
        set()
        if save_sliding_window
        else set(
            DEFAULT_RECOMPUTE_LAYER_TYPES
            if recompute_layer_types is None
            else recompute_layer_types
        )
    )
    if skip_layer_types:
        install_layer_type_hooks(model, state)
    if offload:
        from axolotl.monkeypatch.selective_checkpointing_offload import (
            build_sac_offload_context_fn,
        )

        context_fn = build_sac_offload_context_fn(
            save,
            save_sliding_window,
            state,
            recompute_layer_types=recompute_layer_types,
        )
    else:
        context_fn = build_sac_context_fn(
            save, save_sliding_window, state, recompute_layer_types
        )
    orig_enable = model.gradient_checkpointing_enable

    def enable_with_sac(gradient_checkpointing_kwargs=None, **kwargs):
        gc_kwargs = dict(gradient_checkpointing_kwargs or {})
        gc_kwargs["use_reentrant"] = False
        gc_kwargs["context_fn"] = context_fn
        return orig_enable(gradient_checkpointing_kwargs=gc_kwargs, **kwargs)

    enable_with_sac._axolotl_sac = True
    model.gradient_checkpointing_enable = enable_with_sac
    LOG.info(
        "selective_checkpointing enabled: "
        f"save={save or [ATTENTION_GROUP]} (eager SAC, non-reentrant)"
    )
