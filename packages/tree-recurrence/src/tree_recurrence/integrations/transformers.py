# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Make transformers' recurrent mixers follow tree-packed rows.

``install(model)`` hooks the modeling module of every recurrent mixer in ``model`` at
seams transformers already has, and leaves the mixers' forwards alone:

- Gated DeltaNet (transformers >= 5.15): the forward passes its kwargs, the tree
  among them, to the module-level ``causal_conv1d_fn`` and
  ``torch_chunk_gated_delta_rule``. Both are wrapped.
- Mamba2 (Mamba2, Nemotron-H, Bamba, Granite-MoE-Hybrid): the block passes its
  kwargs to the mixer, which passes them to ``mamba2_split_conv1d_scan_combined``
  (training) or to ``causal_conv1d_fn`` and ``mamba2_chunk_scan``. All three are
  wrapped. ``Mamba2ForCausalLM`` drops its kwargs before the backbone, so its
  forward hands the tree on.
- LFM2's decoder layer calls its short conv with ``seq_idx`` only, so the layer's
  forward is wrapped to run the conv on the tree for that call.

A wrapper reads the tree from its kwargs, which gradient checkpointing saves and
replays, and passes everything else through untouched. It calls the function it
wrapped when it runs, so a hub kernel bound later is used too. A tree-packed forward
builds no cache (a row's final state is no sample's), whatever ``use_cache`` the
caller left to the config.

Each modeling module has one owner. TRL's ``patch_tree_gdn``
(huggingface/trl#7686) also wraps the Gated DeltaNet functions: a module it already
patched is left to it, and a module patched here is added to its record, so it does
not wrap the module again.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import logging
import sys
import weakref
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal

import torch

from tree_recurrence._hooks import (
    Hook,
    hook_of,
    install_hook,
    loads_global,
    remove_hook,
)
from tree_recurrence.components.conv import channels_first_conv
from tree_recurrence.components.gated_delta_rule import gated_delta_rule_scan
from tree_recurrence.components.mamba2 import split_conv1d_scan_on_tree, ssd_scan
from tree_recurrence.components.short_conv import gated_short_conv
from tree_recurrence.driver import DEFAULT_POLICY, Policy, tree_conv, tree_scan
from tree_recurrence.plan import tree_kwargs, tree_plan
from tree_recurrence.registry import (
    TreeFamily,
    follows_trees,
    mixer_family,
    register_family,
    tree_family,
)

LOG = logging.getLogger(__name__)

TRL_PACKING = "trl.experimental.async_grpo.packing"
BACKENDS = {"gated_delta_net": ("tree_recurrence", "trl")}

# Model types with recurrent layers; one no family covers gets "none".
RECURRENT_MODEL_TYPES = frozenset(
    {
        "bailing_hybrid",
        "bamba",
        "falcon_h1",
        "falcon_mamba",
        "glm5_next",
        "granitemoehybrid",
        "jamba",
        "kimi_linear",
        "lfm2",
        "lfm2_moe",
        "mamba",
        "mamba2",
        "minimax",
        "nemotron_h",
        "olmo_hybrid",
        "plamo2",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_5_text",
        "qwen3_next",
        "recurrent_gemma",
        "rwkv",
        "xlstm",
        "zamba",
        "zamba2",
    }
)
NON_RECURRENT_LAYER_TYPES = frozenset(
    {
        "attention",
        "chunked_attention",
        "full_attention",
        "mlp",
        "moe",
        "sliding_attention",
    }
)

Mode = Literal["tree", "chains", "none"]


@dataclass(frozen=True)
class Support:
    """How a model may pack rows.

    ``tree``: every recurrent layer follows the trie, so samples may share prefixes.
    ``chains``: some recurrent layer keeps whole samples apart but cannot follow a
    trie, so every sample has to be its own chain. ``none``: some recurrent layer
    would mix packed samples. ``owners`` maps each recurrent mixer class to what makes
    it follow trees: ``tree_recurrence``, ``trl``, or ``model`` (its own forward).
    """

    mode: Mode
    reason: str
    owners: Mapping[str, str] = field(default_factory=dict)

    @property
    def shares_prefixes(self) -> bool:
        return self.mode == "tree"


class SeamMissing(Exception):
    """The modeling module lacks what a family's hooks attach to."""


@dataclass
class InstallContext:
    model: torch.nn.Module
    mixer_classes: tuple[type[torch.nn.Module], ...]
    policy: Policy
    backend: str


@dataclass(eq=False)
class _ModuleState:
    policy: Policy
    hooks: dict[tuple[Any, str], Hook] = field(default_factory=dict)
    # TRL's packing module whose record of patched modules lists this one
    claimed: ModuleType | None = None
    # the installed models that run on these hooks
    models: weakref.WeakSet[torch.nn.Module] = field(default_factory=weakref.WeakSet)

    def add(self, hook: Hook) -> None:
        self.hooks[(hook.owner, hook.name)] = hook


_INSTALLED: dict[str, _ModuleState] = {}
_CACHE_HOOKS: weakref.WeakKeyDictionary[torch.nn.Module, Any] = (
    weakref.WeakKeyDictionary()
)


def install(
    model: torch.nn.Module,
    *,
    policy: Policy | None = None,
    backends: Mapping[str, str] | None = None,
) -> Support:
    """Make ``model``'s recurrent mixers follow tree-packed rows, where they can.

    Hooks are per modeling module, so models of one architecture share them and
    their ``policy`` (the latest install's). Installing again is a no-op beyond
    that, and ``uninstall`` takes the hooks off once no installed model uses them.
    Call it once the model is loaded: a mixer whose forward ``kernelize`` replaced
    cannot be hooked.

    A recurrent layer no family covers makes the answer ``none``: one whose
    ``layer_types`` entry no found family implements, or a module outside every
    covered mixer with a state-space decay (``A_log``) or a causal depthwise conv.

    ``backends`` picks who runs a family's trees: ``{"gated_delta_net": "trl"}`` uses
    TRL's ``patch_tree_gdn`` when TRL has it, and this package otherwise.
    """
    policy = policy or DEFAULT_POLICY
    backends = dict(backends or {})
    for family_name, backend in backends.items():
        if backend not in BACKENDS.get(family_name, ()):
            raise ValueError(f"unknown backend {backend!r} for {family_name!r}")
    model_types = _model_types(model)
    mixers: dict[type[torch.nn.Module], list[torch.nn.Module]] = {}
    for module in model.modules():
        if mixer_family(type(module).__name__) is not None:
            mixers.setdefault(type(module), []).append(module)
    uncovered = _uncovered_recurrence(model, model_types, mixers)
    if uncovered:
        return Support("none", "; ".join(uncovered))
    if not mixers:
        return Support("tree", "no recurrent layers")

    _skip_the_cache(model)
    owners: dict[str, str] = {}
    chains: list[str] = []
    refusals: list[str] = []
    for cls, instances in mixers.items():
        name = cls.__name__
        family = mixer_family(name)
        assert family is not None
        if follows_trees(instances[0]):
            owners[name] = "model"
            continue
        replaced = sum(_forward_replaced(module) for module in instances)
        if replaced:
            refusals.append(
                f"{name}: {replaced} layers had their forward replaced (by kernelize?), "
                "which does not follow trees"
            )
            continue
        if family.install is None:
            if family.support == "chains":
                owners[name] = "model"
                chains.append(name)
            else:
                refusals.append(f"{name}: its forward is not marked as following trees")
            continue
        context = InstallContext(
            model, (cls,), policy, backends.get(family.name, "tree_recurrence")
        )
        try:
            owners[name] = family.install(sys.modules[cls.__module__], context)
        except SeamMissing as error:
            refusals.append(f"{name}: {error}")
    if refusals:
        return Support("none", "; ".join(refusals), owners)
    if chains:
        return Support(
            "chains",
            f"{', '.join(chains)} keep whole samples apart but cannot follow a tree",
            owners,
        )
    return Support("tree", f"{', '.join(sorted(owners))} follow trees", owners)


def uninstall(model: torch.nn.Module | None = None) -> None:
    """Undo ``install`` for ``model``, or for every model.

    A modeling module keeps its hooks while another installed model still uses them.
    """
    if model is None:
        names = set(_INSTALLED)
    else:
        names = {
            type(module).__module__
            for module in model.modules()
            if mixer_family(type(module).__name__) is not None
        }
    modules = list(_CACHE_HOOKS) if model is None else list(model.modules())
    for module in modules:
        handle = _CACHE_HOOKS.pop(module, None)
        if handle is not None:
            handle.remove()
    for name in names:
        state = _INSTALLED.get(name)
        if state is None:
            continue
        if model is not None:
            state.models.discard(model)
            if state.models:
                continue
        del _INSTALLED[name]
        for hook in state.hooks.values():
            remove_hook(hook)
        if state.claimed is not None:
            getattr(state.claimed, "_PATCHED", set()).discard(sys.modules.get(name))


def _model_types(model: torch.nn.Module) -> set[str]:
    config = getattr(model, "config", None)
    if config is None:
        return set()
    types = {getattr(config, "model_type", None)}
    with contextlib.suppress(Exception):
        types.add(config.get_text_config().model_type)
    return {t for t in types if t}


def _looks_recurrent(module: torch.nn.Module) -> bool:
    """A state-space decay, or a causal depthwise short conv."""
    if "A_log" in module._parameters:
        return True
    if not isinstance(module, torch.nn.Conv1d) or not isinstance(module.padding, tuple):
        return False
    width = module.kernel_size[0]
    depthwise = module.groups == module.in_channels > 1
    return depthwise and width > 1 and module.padding[0] == width - 1


def _uncovered_recurrence(
    model: torch.nn.Module,
    model_types: set[str],
    mixers: Mapping[type[torch.nn.Module], list[torch.nn.Module]],
) -> list[str]:
    """Why some recurrent layer of ``model`` follows no tree, if one does not."""
    found = {mixer_family(cls.__name__) for cls in mixers}
    names = "/".join(sorted(model_types)) or type(model).__name__
    reasons = []
    expected = tree_family(model_types)
    if expected is not None and expected not in found:
        reasons.append(
            f"{names} should have {', '.join(expected.mixers)} layers, and none were "
            "found"
        )
    elif not found and model_types & RECURRENT_MODEL_TYPES:
        reasons.append(f"{names} has recurrent layers without tree support")
    covered = {t for family in found if family for t in family.layer_types}
    layers = _recurrent_layer_types(model) - covered
    if layers:
        reasons.append(
            f"{names} has {', '.join(sorted(layers))} layers that no tree family covers"
        )
    inside = {
        id(module)
        for instances in mixers.values()
        for mixer in instances
        for module in mixer.modules()
    }
    loose: dict[str, list[str]] = {}
    for path, module in model.named_modules():
        if id(module) not in inside and _looks_recurrent(module):
            loose.setdefault(type(module).__name__, []).append(path)
    for cls, paths in loose.items():
        reasons.append(
            f"{cls} at {paths[0]}{f' and {len(paths) - 1} more' if len(paths) > 1 else ''}"
            " looks recurrent (a state-space decay or a causal depthwise conv) and no "
            "tree family covers it"
        )
    return reasons


def _recurrent_layer_types(model: torch.nn.Module) -> set[str]:
    config = getattr(model, "config", None)
    try:
        text_config = config.get_text_config()  # type: ignore[union-attr]
    except Exception:
        text_config = config
    layer_types = getattr(text_config, "layer_types", None) or []
    return set(layer_types) - NON_RECURRENT_LAYER_TYPES


def _forward_replaced(module: torch.nn.Module) -> bool:
    bound = vars(module).get("forward")
    if bound is None:
        return False
    # accelerate's hooks keep the forward they wrap
    wrapped = vars(module).get("_old_forward", bound)
    return getattr(wrapped, "__func__", None) is not type(module).forward


def _without_cache(module, args, kwargs):
    if kwargs.get("tree_cu_seqlens_q") is None or kwargs.get("use_cache") is False:
        return None
    return args, {**kwargs, "use_cache": False}


def _skip_the_cache(model: torch.nn.Module) -> None:
    """Tree-packed calls of ``model``'s transformers models run with
    ``use_cache=False``: TRL leaves it to the config, which builds a cache in
    training."""
    for module in model.modules():
        if module in _CACHE_HOOKS or not hasattr(type(module), "base_model_prefix"):
            continue
        if "use_cache" in inspect.signature(type(module).forward).parameters:
            _CACHE_HOOKS[module] = module.register_forward_pre_hook(
                _without_cache, with_kwargs=True
            )


def _policy(module_name: str) -> Policy:
    state = _INSTALLED.get(module_name)
    return DEFAULT_POLICY if state is None else state.policy


def _state(modeling: ModuleType, context: InstallContext) -> _ModuleState:
    state = _INSTALLED.get(modeling.__name__)
    if state is None:
        state = _INSTALLED[modeling.__name__] = _ModuleState(context.policy)
    state.policy = context.policy
    state.models.add(context.model)
    return state


def _require(
    modeling: ModuleType,
    mixer_classes: tuple[type[torch.nn.Module], ...],
    names: tuple[str, ...],
) -> None:
    """Raise unless every mixer forward calls each of ``names`` from the module."""
    for name in names:
        if not callable(getattr(modeling, name, None)):
            raise SeamMissing(f"{modeling.__name__} has no {name}")
        for cls in mixer_classes:
            if not loads_global(cls.forward, name):
                raise SeamMissing(
                    f"{cls.__name__}.forward does not call the module's {name} "
                    "(an older transformers, or a replaced forward)"
                )


def _signature(function: Callable[..., Any]) -> inspect.Signature:
    return inspect.signature(function)


def _conv_signature(hidden_states, weight, bias=None, activation=None, **kwargs):
    pass


def _gated_delta_rule_signature(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    pass


def _ssd_signature(
    hidden_states,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    dt_bias=None,
    initial_states=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    **kwargs,
):
    pass


def _split_conv1d_scan_signature(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    activation="silu",
    rmsnorm_weight=None,
    rmsnorm_eps=1e-6,
    outproj_weight=None,
    outproj_bias=None,
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
    **kwargs,
):
    pass


_CONV = _signature(_conv_signature)
_GATED_DELTA_RULE = _signature(_gated_delta_rule_signature)
_SSD = _signature(_ssd_signature)
_SPLIT_CONV1D_SCAN = _signature(_split_conv1d_scan_signature)


def _bind(signature: inspect.Signature, args, kwargs) -> dict[str, Any]:
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return bound.arguments


def _announce(hook: Hook, kernel: bool, strategy: str) -> None:
    """Say once per hook how it runs trees, loudly when on a PyTorch reference."""
    if hook.announced:
        return
    hook.announced = True
    where = f"{hook.owner.__name__}.{hook.name}"
    if kernel:
        LOG.info(f"tree_recurrence: {where} runs trees on a kernel ({strategy})")
    else:
        LOG.warning(
            f"tree_recurrence: {where} has no kernel, so trees run on its PyTorch "
            "reference, every path as one padded batch (slow)"
        )


def _no_cache(*states: Any) -> None:
    if any(state is not None and state is not False for state in states):
        raise ValueError(
            "a tree-packed row runs without a cache: call the model with "
            "use_cache=False"
        )


def _conv_hook(hook: Hook) -> Callable[..., Any]:
    def causal_conv1d_fn(*args, **kwargs):
        plan = tree_plan(kwargs) if hook.enabled else None
        if plan is None:
            return hook.original(*args, **kwargs)
        bound = _bind(_CONV, args, kwargs)
        weight = bound["weight"]
        conv = channels_first_conv(
            hook.original, weight, bound["bias"], bound["activation"]
        )
        x = bound["hidden_states"].transpose(1, 2)
        return tree_conv(conv, x, plan, weight.shape[-1]).transpose(1, 2)

    return causal_conv1d_fn


def _gated_delta_rule_hook(hook: Hook) -> Callable[..., Any]:
    def torch_chunk_gated_delta_rule(*args, **kwargs):
        plan = tree_plan(kwargs) if hook.enabled else None
        if plan is None:
            return hook.original(*args, **kwargs)
        bound = _bind(_GATED_DELTA_RULE, args, kwargs)
        _no_cache(bound["initial_state"], bound["output_final_state"])
        varlen = hook.kernel_backed()
        _announce(hook, varlen, "forking or rescanning paths by policy")
        scan = gated_delta_rule_scan(
            hook.original,
            use_qk_l2norm_in_kernel=bound["use_qk_l2norm_in_kernel"],
            varlen=varlen,
        )
        inputs = [bound[name] for name in ("query", "key", "value", "g", "beta")]
        policy = _policy(hook.owner.__name__)
        out = tree_scan(scan, inputs, plan, forkable=varlen, policy=policy)
        return out, None

    return torch_chunk_gated_delta_rule


def _ssd_hook(hook: Hook) -> Callable[..., Any]:
    def mamba2_chunk_scan(*args, **kwargs):
        plan = tree_plan(kwargs) if hook.enabled else None
        if plan is None:
            return hook.original(*args, **kwargs)
        bound = _bind(_SSD, args, kwargs)
        _no_cache(bound["initial_states"], bound["return_final_states"])
        if bound["kwargs"].get("z") is not None:
            raise NotImplementedError("a gated scan (z) on a tree")
        varlen = hook.kernel_backed()
        _announce(hook, varlen, "rescanning paths")
        scan = ssd_scan(
            hook.original,
            bound["A"],
            varlen=varlen,
            chunk_size=bound["chunk_size"],
            D=bound["D"],
            dt_bias=bound["dt_bias"],
            dt_softplus=bound["dt_softplus"],
            dt_limit=bound["dt_limit"],
        )
        inputs = [bound[name] for name in ("hidden_states", "dt", "B", "C")]
        return tree_scan(scan, inputs, plan, forkable=False)

    return mamba2_chunk_scan


_SPLIT_CONV1D_SCAN_PARAMS = tuple(
    name
    for name in _SPLIT_CONV1D_SCAN.parameters
    if name not in ("zxbcdt", "initial_states", "return_final_states", "kwargs")
)


def _split_conv1d_scan_hook(hook: Hook) -> Callable[..., Any]:
    def mamba2_split_conv1d_scan_combined(*args, **kwargs):
        plan = tree_plan(kwargs) if hook.enabled else None
        # transformers' reference returns None: the mixer then runs the unfused path
        if plan is None or not hook.kernel_backed():
            return hook.original(*args, **kwargs)
        _announce(hook, True, "the fused mixer over every path")
        bound = _bind(_SPLIT_CONV1D_SCAN, args, kwargs)
        _no_cache(bound["initial_states"], bound["return_final_states"])
        params = {name: bound[name] for name in _SPLIT_CONV1D_SCAN_PARAMS}
        return split_conv1d_scan_on_tree(hook.original, bound["zxbcdt"], plan, **params)

    return mamba2_split_conv1d_scan_combined


@contextlib.contextmanager
def _forward_for_call(
    module: torch.nn.Module, forward: Callable[..., Any]
) -> Iterator[None]:
    """``module`` runs ``forward`` until the block exits."""
    previous = vars(module).get("forward")
    module.forward = forward  # type: ignore[method-assign]
    try:
        yield
    finally:
        if previous is None:
            del module.forward
        else:
            module.forward = previous  # type: ignore[method-assign]


def _tree_kwargs_into(attribute: str) -> Callable[[Hook], Callable[..., Any]]:
    """A forward that hands the tree kwargs to its ``attribute`` submodule."""

    def build(hook: Hook) -> Callable[..., Any]:
        def forward(self, *args, **kwargs):
            tree = tree_kwargs(kwargs) if hook.enabled else {}
            if not tree:
                return hook.original(self, *args, **kwargs)
            for name in tree:
                del kwargs[name]
            inner = getattr(self, attribute)
            with _forward_for_call(inner, functools.partial(inner.forward, **tree)):
                return hook.original(self, *args, **kwargs)

        return forward

    return build


def _short_conv_layer_hook(
    modeling: ModuleType, attribute: str, hook: Hook
) -> Callable[..., Any]:
    def run(mixer, plan, hidden_states, past_key_values=None, **kwargs):
        _no_cache(past_key_values)
        return gated_short_conv(mixer, modeling.causal_conv1d_fn, hidden_states, plan)

    def forward(self, *args, **kwargs):
        plan = tree_plan(kwargs) if hook.enabled else None
        mixer = getattr(self, attribute, None)
        if plan is None or mixer is None:
            return hook.original(self, *args, **kwargs)
        with _forward_for_call(mixer, functools.partial(run, mixer, plan)):
            return hook.original(self, *args, **kwargs)

    return forward


def _trl_packing() -> ModuleType | None:
    return sys.modules.get(TRL_PACKING)


def _trl_owns(modeling: ModuleType, packing: ModuleType | None) -> bool:
    if packing is None or hook_of(modeling, "torch_chunk_gated_delta_rule"):
        return False
    current = getattr(modeling, "torch_chunk_gated_delta_rule", None)
    return modeling in getattr(packing, "_PATCHED", ()) or (
        getattr(current, "__module__", None) == TRL_PACKING
    )


def _install_gated_delta_net(modeling: ModuleType, context: InstallContext) -> str:
    names = ("causal_conv1d_fn", "torch_chunk_gated_delta_rule")
    _require(modeling, context.mixer_classes, names)
    packing = _trl_packing()
    if _trl_owns(modeling, packing):
        return "trl"
    if context.backend == "trl":
        try:
            packing = importlib.import_module(TRL_PACKING)
        except ImportError:
            packing = None
        patch = getattr(packing, "patch_tree_gdn", None)
        if patch is not None:
            state = _INSTALLED.get(modeling.__name__)
            if state is not None:  # hand the module over
                for name in names:
                    hook = state.hooks.pop((modeling, name), None)
                    if hook is not None:
                        remove_hook(hook)
                if state.claimed is not None:
                    getattr(state.claimed, "_PATCHED", set()).discard(modeling)
                    state.claimed = None
            if patch(context.model):
                return "trl"
        LOG.warning(
            "TRL's patch_tree_gdn is unavailable or declined the model; Gated "
            "DeltaNet follows trees through tree_recurrence"
        )
    state = _state(modeling, context)
    state.add(install_hook(modeling, "causal_conv1d_fn", _conv_hook))
    state.add(
        install_hook(modeling, "torch_chunk_gated_delta_rule", _gated_delta_rule_hook)
    )
    patched = getattr(packing, "_PATCHED", None)
    if isinstance(patched, set):
        patched.add(modeling)
        state.claimed = packing
    return "tree_recurrence"


# causal LMs whose forward drops its kwargs before the backbone
_KWARGS_DROPPING = {"Mamba2ForCausalLM": "backbone"}


def _install_mamba2(modeling: ModuleType, context: InstallContext) -> str:
    names: tuple[str, ...] = ("causal_conv1d_fn", "mamba2_chunk_scan")
    fused = "mamba2_split_conv1d_scan_combined"
    if any(loads_global(cls.forward, fused) for cls in context.mixer_classes):
        names += (fused,)
    _require(modeling, context.mixer_classes, names)
    state = _state(modeling, context)
    state.add(install_hook(modeling, "causal_conv1d_fn", _conv_hook))
    state.add(install_hook(modeling, "mamba2_chunk_scan", _ssd_hook))
    if fused in names:
        state.add(install_hook(modeling, fused, _split_conv1d_scan_hook))
    for class_name, attribute in _KWARGS_DROPPING.items():
        cls = getattr(modeling, class_name, None)
        if isinstance(cls, type):
            state.add(install_hook(cls, "forward", _tree_kwargs_into(attribute)))
    return "tree_recurrence"


def _install_short_conv(modeling: ModuleType, context: InstallContext) -> str:
    if not callable(getattr(modeling, "causal_conv1d_fn", None)):
        raise SeamMissing(f"{modeling.__name__} has no causal_conv1d_fn")
    layers = {
        (type(parent), attribute)
        for parent in context.model.modules()
        for attribute, child in parent.named_children()
        if isinstance(child, context.mixer_classes)
    }
    for cls, _ in layers:
        if cls.__module__ != modeling.__name__:
            raise SeamMissing(f"{cls.__name__} is not from {modeling.__name__}")
    state = _state(modeling, context)
    for cls, attribute in layers:
        state.add(
            install_hook(
                cls,
                "forward",
                functools.partial(_short_conv_layer_hook, modeling, attribute),
            )
        )
    return "tree_recurrence"


register_family(
    TreeFamily(
        "gated_delta_net",
        model_types=(
            "olmo_hybrid",
            "qwen3_5",
            "qwen3_5_moe",
            "qwen3_5_moe_text",
            "qwen3_5_text",
            "qwen3_next",
        ),
        mixers=(
            "OlmoHybridGatedDeltaNet",
            "Qwen3NextGatedDeltaNet",
            "Qwen3_5GatedDeltaNet",
            "Qwen3_5MoeGatedDeltaNet",
        ),
        layer_types=("linear_attention",),
        install=_install_gated_delta_net,
    )
)
register_family(
    TreeFamily(
        "mamba2",
        model_types=("bamba", "granitemoehybrid", "mamba2", "nemotron_h"),
        mixers=(
            "BambaMixer",
            "GraniteMoeHybridMambaLayer",
            "Mamba2Mixer",
            "NemotronHMamba2Mixer",
        ),
        layer_types=("linear_attention", "mamba"),
        install=_install_mamba2,
    )
)
register_family(
    TreeFamily(
        "short_conv",
        model_types=("lfm2", "lfm2_moe"),
        mixers=("Lfm2MoeShortConv", "Lfm2ShortConv"),
        layer_types=("conv",),
        install=_install_short_conv,
    )
)
