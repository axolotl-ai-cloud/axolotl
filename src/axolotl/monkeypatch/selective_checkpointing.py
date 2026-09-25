"""Eager selective activation checkpointing (SAC): save chosen ops instead of recomputing.

Uses ``torch.utils.checkpoint.create_selective_checkpoint_contexts`` (no torch.compile).
Ops are matched at the dispatcher level, so custom kernels that are not registered as
torch ops are simply recomputed as usual — only ops we want to *save* need to be
dispatcher-visible.
"""

from __future__ import annotations

import fnmatch
import inspect
from collections import Counter
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
from torch.utils.checkpoint import (
    CheckpointPolicy,
    checkpoint as _torch_checkpoint,
    create_selective_checkpoint_contexts,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

ATTENTION_GROUP = "attention"

# PyTorch releases before 2.14 reject this keyword.
_SUPPORTS_RESPECT_SAVED_TENSORS_HOOKS = (
    "respect_saved_tensors_hooks" in inspect.signature(_torch_checkpoint).parameters
)

_ATEN_ATTENTION_PACKETS = (
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
)

# regions to observe before warning that no op ever matched the save policy
_NO_MATCH_WARN_REGIONS = 64

# (arg index, dim) locating the contraction dim K of each matmul op the module
# and shape rules may save. aten::bmm is excluded on purpose: math-SDPA's attn@V
# has K = seq_len. aten::addmm_ (in-place) must never be saved.
_RULE_MATMUL_K: dict[str, tuple[int, int]] = {
    "aten::mm": (1, 0),
    "aten::addmm": (2, 0),
    "aten::linear": (1, -1),
    "aten::_grouped_mm": (1, -2),
    "bitsandbytes::gemm_4bit": (0, -1),
}
_RULE_MATMUL_OPS = frozenset(_RULE_MATMUL_K)


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


def _matmul_k(name: str, args: tuple) -> int | None:
    """Contraction dim K of a rule matmul op, or None if it cannot be read."""
    spec = _RULE_MATMUL_K.get(name)
    if spec is None:
        return None
    index, dim = spec
    try:
        if index >= len(args):
            return None
        tensor = args[index]
        if not torch.is_tensor(tensor) or tensor.dim() < 2:
            return None
        return int(tensor.shape[dim])
    except (IndexError, AttributeError, TypeError):
        return None


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
        # same threading assumption as current_layer_type: forward and backward
        # never overlap, so one depth counter per save_modules entry suffices
        self.module_depth: dict[str, int] = {}
        self.scope_stack: dict[str, list[bool]] = {}
        self.hook_fires: dict[str, int] = {}
        self.hook_fires_outside: dict[str, int] = {}
        self.module_targets: dict[str, int] = {}
        self.rule_saves: dict[str, int] = {}
        self.rule_replays: dict[str, int] = {}
        self.recompute_seen: bool = False
        self.warned_dead_saves: bool = False
        self.save_modules: list[str] = []
        self.save_matmul_min_k: int | None = None


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


def _in_sac_region() -> bool:
    """True while a SAC forward or recompute dispatch mode is active."""
    try:
        stack = _get_current_dispatch_mode_stack()
    except Exception:  # pylint: disable=broad-exception-caught
        return False
    return any(callable(getattr(mode, "policy_fn", None)) for mode in stack)


def _checkpointed_layer_names(model) -> list[str]:
    try:
        from transformers import GradientCheckpointingLayer
    except ImportError:
        return []
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, GradientCheckpointingLayer)
    ]


def _encloses(outer: str, names: list[str]) -> str | None:
    for name in names:
        if name != outer and (outer == "" or name.startswith(outer + ".")):
            return name
    return None


def _module_name_matches(name: str, entry: str) -> bool:
    if any(ch in entry for ch in "*?["):
        return fnmatch.fnmatchcase(name, entry)
    return name == entry or name.endswith("." + entry)


def _leaf_name_hint(model, limit: int = 8) -> str:
    counts: Counter = Counter()
    for name, module in model.named_modules():
        if name and next(module.children(), None) is None:
            counts[name.rsplit(".", 1)[-1]] += 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(leaf for leaf, _ in ranked[:limit])


def install_module_scope_hooks(
    model, state: SacPolicyState, save_modules: list[str]
) -> dict[str, int]:
    """Bracket each matched module's forward with a per-entry scope depth counter.

    While any depth is positive the policy saves matmul ops, so the rule reaches
    a specific projection even though the dispatcher never sees module names. A
    PEFT-wrapped match resolves to its ``base_layer`` so LoRA A/B stay out of
    scope. Hooks fire again during recompute, so forward and replay agree.

    A scope only counts when it opens inside a checkpoint region. A module that
    encloses the checkpointed decoder layers (``model``, ``language_model``, a
    bare ``*``) would otherwise be in scope for every forward region but never
    during recompute, desyncing torch's saved-tensor cache; such matches are
    skipped with a warning.
    """
    if not hasattr(model, "named_modules"):
        for entry in save_modules:
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} matched no "
                f"module in {type(model).__name__}; the rule is inert."
            )
        return {}

    layer_names = _checkpointed_layer_names(model)
    hooked: dict[str, int] = {}
    for entry in save_modules:
        state.module_depth.setdefault(entry, 0)
        state.scope_stack.setdefault(entry, [])
        state.hook_fires.setdefault(entry, 0)
        state.hook_fires_outside.setdefault(entry, 0)
        targets: dict[int, torch.nn.Module] = {}
        enclosing: list[tuple[str, str]] = []
        outer: str | None = None
        # named_modules is pre-order, so a match's descendants follow it contiguously
        for name, module in model.named_modules():
            if outer is not None and (outer == "" or name.startswith(outer + ".")):
                continue
            if not _module_name_matches(name, entry):
                continue
            layer = _encloses(name, layer_names)
            if layer is not None:
                enclosing.append((name, layer))
                continue
            outer = name
            base = getattr(module, "base_layer", None)
            target = base if isinstance(base, torch.nn.Module) else module
            targets.setdefault(id(target), target)

        def _pre(mod, args, _entry=entry):
            counted = _in_sac_region()
            state.scope_stack.setdefault(_entry, []).append(counted)
            if counted:
                state.module_depth[_entry] = state.module_depth.get(_entry, 0) + 1
                state.hook_fires[_entry] = state.hook_fires.get(_entry, 0) + 1
            else:
                state.hook_fires_outside[_entry] = (
                    state.hook_fires_outside.get(_entry, 0) + 1
                )

        def _post(mod, args, output, _entry=entry):
            stack = state.scope_stack.get(_entry)
            if stack and stack.pop():
                state.module_depth[_entry] = max(
                    state.module_depth.get(_entry, 0) - 1, 0
                )

        for target in targets.values():
            target.register_forward_pre_hook(_pre)
            target.register_forward_hook(_post, always_call=True)
        hooked[entry] = len(targets)
        state.module_targets[entry] = len(targets)
        if enclosing:
            name, layer = enclosing[0]
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} matches "
                f"{len(enclosing)} module(s) that enclose checkpointed decoder "
                f"layers (e.g. {name or '<root>'!r} contains {layer!r}); their "
                "forward runs outside the checkpoint regions, so they were not "
                "hooked. Name a module inside a decoder layer instead."
            )
        if targets:
            LOG.info(
                f"selective_checkpointing: module-scope hooks on {len(targets)} "
                f"modules for save_modules entry {entry!r}"
            )
        elif not enclosing:
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} matched no "
                f"module in {type(model).__name__}; the rule is inert. Leaf module "
                f"names in this model include: {_leaf_name_hint(model)}"
            )
    return hooked


def build_sac_policy(
    save: list[str] | None = None,
    state: SacPolicyState | None = None,
    save_sliding_window: bool = False,
    recompute_layer_types: list[str] | None = None,
    *,
    save_modules: list[str] | None = None,
    save_matmul_min_k: int | None = None,
) -> Callable:
    """Build an eager SAC policy_fn: MUST_SAVE for matching ops, PREFER_RECOMPUTE otherwise.

    ``save`` entries are either the ``"attention"`` group or substrings matched
    against the qualified op name (e.g. ``"aten::mm"``). Unless
    ``save_sliding_window`` is set, hybrid-model attention calls bounded by a
    sliding window keep being recomputed — SWA is cheap and not worth the saved
    memory; only full-attention calls are saved.

    Two matmul rules follow the save list. ``save_modules``: a matmul dispatched
    while a module-scope hook (``install_module_scope_hooks``) is active is saved.
    ``save_matmul_min_k``: a matmul whose contraction dim K is at least this value
    is saved. Neither rule consults layer types or sliding windows.
    """
    save = save or [ATTENTION_GROUP]
    state = state or SacPolicyState()
    state.save_modules = list(save_modules or [])
    state.save_matmul_min_k = save_matmul_min_k
    state.rule_saves.setdefault("save", 0)
    for entry in state.save_modules:
        state.rule_saves.setdefault(f"module:{entry}", 0)
        state.module_depth.setdefault(entry, 0)
        state.scope_stack.setdefault(entry, [])
        state.hook_fires.setdefault(entry, 0)
        state.hook_fires_outside.setdefault(entry, 0)
    if save_matmul_min_k:
        state.rule_saves.setdefault("shape", 0)
    has_rules = bool(state.save_modules or save_matmul_min_k)
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
                if has_rules:
                    return _rule_policy(ctx, op, args)
                return CheckpointPolicy.PREFER_RECOMPUTE
            if not save_sliding_window and _is_sliding_window_call(op, args, kwargs):
                if name not in state.sliding_op_names:
                    state.sliding_op_names.add(name)
                    LOG.info(
                        f"selective_checkpointing: recomputing sliding-window "
                        f"calls of `{name}` (save_sliding_window: false)"
                    )
                if has_rules:
                    return _rule_policy(ctx, op, args)
                return CheckpointPolicy.PREFER_RECOMPUTE
            if name not in state.saved_op_names:
                state.saved_op_names.add(name)
                LOG.info(
                    f"selective_checkpointing: saving `{name}` "
                    "(backward will not recompute it)"
                )
            if not getattr(ctx, "is_recompute", False):
                state.rule_saves["save"] += 1
            if has_rules:
                # credit a rule whose op the save list already took, so its
                # diagnostics do not report it inert
                _rule_policy(ctx, op, args)
            return CheckpointPolicy.MUST_SAVE
        if has_rules:
            return _rule_policy(ctx, op, args)
        return CheckpointPolicy.PREFER_RECOMPUTE

    def _count(key: str, is_recompute: bool) -> None:
        counts = state.rule_replays if is_recompute else state.rule_saves
        counts[key] = counts.get(key, 0) + 1

    def _rule_policy(ctx, op, args):
        is_recompute = getattr(ctx, "is_recompute", False)
        if is_recompute:
            state.recompute_seen = True
        name = _op_name(op)
        if name not in _RULE_MATMUL_OPS:
            return CheckpointPolicy.PREFER_RECOMPUTE
        active = [e for e, depth in state.module_depth.items() if depth > 0]
        if active:
            for entry in active:
                _count(f"module:{entry}", is_recompute)
                logged = f"module:{entry}:{name}"
                if logged not in state.saved_op_names:
                    state.saved_op_names.add(logged)
                    LOG.info(
                        f"selective_checkpointing: saving `{name}` inside "
                        f"save_modules entry {entry!r} (backward will not "
                        "recompute it)"
                    )
            return CheckpointPolicy.MUST_SAVE
        min_k = state.save_matmul_min_k
        if min_k:
            k = _matmul_k(name, args)
            if k is not None and k >= min_k:
                _count("shape", is_recompute)
                logged = f"shape:{name}"
                if logged not in state.saved_op_names:
                    state.saved_op_names.add(logged)
                    LOG.info(
                        f"selective_checkpointing: saving `{name}` with K={k} >= "
                        f"{min_k} (backward will not recompute it)"
                    )
                return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    return policy_fn


def _warn_unmatched_rules(state: SacPolicyState) -> None:
    """Warn once, per rule, about save rules that never saved a tensor."""
    if state.warned_no_match or state.regions_seen < _NO_MATCH_WARN_REGIONS:
        return
    state.warned_no_match = True
    n = state.regions_seen
    rule_keys = [key for key in state.rule_saves if key != "save"]
    rules_saved = any(state.rule_saves[key] > 0 for key in rule_keys)
    save_list_saved = state.rule_saves.get("save", 0) > 0 or any(
        not name.startswith(("module:", "shape:")) for name in state.saved_op_names
    )
    if not save_list_saved:
        message = (
            f"selective_checkpointing: no op matched the save policy after "
            f"{n} checkpoint regions. Your attention "
            "implementation may not be dispatcher-visible (e.g. a custom "
            "kernel not registered via torch.library)"
        )
        if rules_saved:
            message += ". Module/shape rules did save tensors."
        else:
            message += (
                "; everything is being recomputed as with plain gradient checkpointing."
            )
        LOG.warning(message)
    for entry in state.save_modules:
        if state.rule_saves.get(f"module:{entry}", 0):
            continue
        if state.module_targets.get(entry) == 0:
            continue  # install already warned
        if state.hook_fires.get(entry, 0) == 0 and state.hook_fires_outside.get(
            entry, 0
        ):
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} never saved "
                f"a tensor after {n} checkpoint regions: its module only runs "
                "outside the checkpoint regions (e.g. lm_head, embed_tokens, a "
                "final norm), so the policy never sees its matmuls. The rule is "
                "inert."
            )
        elif state.hook_fires.get(entry, 0) == 0:
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} never saved "
                f"a tensor after {n} checkpoint regions: its module hooks never "
                "fired, so the module forward is bypassed (fused kernel such as "
                "lora_mlp_kernel, or a replaced forward). The rule is inert."
            )
        else:
            LOG.warning(
                f"selective_checkpointing: save_modules entry {entry!r} never saved "
                f"a tensor after {n} checkpoint regions: its hooks fired but the "
                "module forward does not dispatch a visible matmul (fused/custom "
                "kernel?). The rule is inert."
            )
    k = state.save_matmul_min_k
    if k and not state.rule_saves.get("shape", 0):
        LOG.warning(
            f"selective_checkpointing: save_matmul_min_k={k} never matched after "
            f"{n} checkpoint regions: no visible matmul with K >= {k} was "
            "dispatched inside a checkpoint region. Lower the threshold (K is a "
            "linear's in_features) or check that the projections are not fused."
        )


def _rule_label(state: SacPolicyState, key: str) -> str:
    if key == "shape":
        return f"save_matmul_min_k={state.save_matmul_min_k}"
    return f"save_modules entry {key.split(':', 1)[1]!r}"


def _warn_dead_rule_saves(state: SacPolicyState) -> None:
    """Warn once about rules whose saved tensors backward's recompute never read."""
    if (
        state.warned_dead_saves
        or not state.recompute_seen
        or state.regions_seen < _NO_MATCH_WARN_REGIONS
    ):
        return
    state.warned_dead_saves = True
    for key, saves in state.rule_saves.items():
        if key == "save" or not saves or state.rule_replays.get(key, 0):
            continue
        LOG.warning(
            f"selective_checkpointing: {_rule_label(state, key)} saved {saves} "
            "tensors in forward but backward's recompute never read one. Torch's "
            "checkpoint early stop ends the replay at the last tensor backward "
            "needs, and this op runs after it (e.g. a layer's final projection "
            "whose output only feeds a residual add), so the rule costs memory "
            "and saves no compute. Target an earlier projection (o_proj, "
            "gate_proj, up_proj) instead."
        )


def run_rule_diagnostics(state: SacPolicyState) -> None:
    """Per-region hook for both context_fns: one-shot rule warnings."""
    _warn_unmatched_rules(state)
    if state.save_modules or state.save_matmul_min_k:
        _warn_dead_rule_saves(state)


def build_sac_context_fn(
    save: list[str] | None = None,
    save_sliding_window: bool = False,
    state: SacPolicyState | None = None,
    recompute_layer_types: list[str] | None = None,
    *,
    save_modules: list[str] | None = None,
    save_matmul_min_k: int | None = None,
) -> Callable:
    """Return a ``context_fn`` for ``torch.utils.checkpoint.checkpoint``."""
    state = state or SacPolicyState()
    policy_fn = build_sac_policy(
        save,
        state,
        save_sliding_window,
        recompute_layer_types,
        save_modules=save_modules,
        save_matmul_min_k=save_matmul_min_k,
    )

    def context_fn():
        state.regions_seen += 1
        run_rule_diagnostics(state)
        return create_selective_checkpoint_contexts(policy_fn)

    return context_fn


def apply_selective_checkpointing(
    model,
    save: list[str] | None = None,
    save_sliding_window: bool = False,
    recompute_layer_types: list[str] | None = None,
    offload: bool = False,
    *,
    save_modules: list[str] | None = None,
    save_matmul_min_k: int | None = None,
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
    if save_modules:
        install_module_scope_hooks(model, state, save_modules)
    if offload:
        from axolotl.monkeypatch.selective_checkpointing_offload import (
            build_sac_offload_context_fn,
        )

        context_fn = build_sac_offload_context_fn(
            save,
            save_sliding_window,
            state,
            recompute_layer_types=recompute_layer_types,
            save_modules=save_modules,
            save_matmul_min_k=save_matmul_min_k,
        )
    else:
        context_fn = build_sac_context_fn(
            save,
            save_sliding_window,
            state,
            recompute_layer_types,
            save_modules=save_modules,
            save_matmul_min_k=save_matmul_min_k,
        )
    orig_enable = model.gradient_checkpointing_enable

    def enable_with_sac(gradient_checkpointing_kwargs=None, **kwargs):
        gc_kwargs = dict(gradient_checkpointing_kwargs or {})
        gc_kwargs["use_reentrant"] = False
        gc_kwargs["context_fn"] = context_fn
        if _SUPPORTS_RESPECT_SAVED_TENSORS_HOOKS:
            gc_kwargs.setdefault("respect_saved_tensors_hooks", False)
        else:
            gc_kwargs.pop("respect_saved_tensors_hooks", None)
        return orig_enable(gradient_checkpointing_kwargs=gc_kwargs, **kwargs)

    enable_with_sac._axolotl_sac = True
    model.gradient_checkpointing_enable = enable_with_sac
    rules = (
        f", save_modules={save_modules}, save_matmul_min_k={save_matmul_min_k}"
        if save_modules or save_matmul_min_k
        else ""
    )
    LOG.info(
        "selective_checkpointing enabled: "
        f"save={save or [ATTENTION_GROUP]}{rules} (eager SAC, non-reentrant)"
    )
