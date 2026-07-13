"""Composable model-family templates and per-model profiles."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Mapping, overload

from .base import Capability, ModelSupport

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import (
        PretrainedConfig,
        PreTrainedModel,
        PreTrainedTokenizerBase,
        ProcessorMixin,
    )

    from axolotl.processing_strategies import ProcessingStrategy
    from axolotl.utils.dict import DictDefault


class ModelHookPhase(str, Enum):
    """Stable lifecycle boundaries available to model-specific hooks."""

    BEFORE_CONFIG_LOAD = "before_config_load"
    CONFIGURE_RUN = "configure_run"
    BEFORE_TOKENIZER_LOAD = "before_tokenizer_load"
    BEFORE_MODEL_BUILD = "before_model_build"
    AFTER_BASE_MODEL_BUILD = "after_base_model_build"
    AFTER_ADAPTER_LOAD = "after_adapter_load"


@dataclass(frozen=True)
class ModelHookContext:
    """Inputs available to a model hook at its lifecycle phase."""

    cfg: DictDefault
    model_config: PretrainedConfig | Mapping[str, Any] | None = None
    tokenizer: PreTrainedTokenizerBase | None = None
    processor: ProcessorMixin | None = None
    model: PreTrainedModel | PeftModel | None = None
    inference: bool = False
    reference_model: bool = False


ModelHook = Callable[[ModelHookContext], None]
AutoModelClassProvider = Callable[[], type | None]
ProcessingStrategyClassProvider = Callable[[], type["ProcessingStrategy"] | None]
ConfigMatcher = Callable[["DictDefault"], bool]
ProcessorMatcher = Callable[["ProcessorMixin"], bool]

_ACTIVE_LEGACY_HOOK: ContextVar[tuple[int, ModelHookPhase] | None] = ContextVar(
    "model_support_legacy_hook",
    default=None,
)


@dataclass(frozen=True)
class ModelStrategies:
    """Typed component providers selected by a family or model profile."""

    auto_model_cls: AutoModelClassProvider | None = None
    processing_strategy_cls: ProcessingStrategyClassProvider | None = None

    def with_overrides(self, overrides: ModelStrategies) -> ModelStrategies:
        return ModelStrategies(
            auto_model_cls=(
                overrides.auto_model_cls
                if overrides.auto_model_cls is not None
                else self.auto_model_cls
            ),
            processing_strategy_cls=(
                overrides.processing_strategy_cls
                if overrides.processing_strategy_cls is not None
                else self.processing_strategy_cls
            ),
        )


@dataclass(frozen=True)
class ModelMatchers:
    """Typed discovery functions for pre-config and processor dispatch."""

    cfg: ConfigMatcher | None = None
    processor: ProcessorMatcher | None = None

    def with_overrides(self, overrides: ModelMatchers) -> ModelMatchers:
        return ModelMatchers(
            cfg=overrides.cfg if overrides.cfg is not None else self.cfg,
            processor=(
                overrides.processor
                if overrides.processor is not None
                else self.processor
            ),
        )


@dataclass(frozen=True)
class ModelHooks:
    """Model-specific hooks grouped by explicit lifecycle phase."""

    by_phase: Mapping[ModelHookPhase, tuple[ModelHook, ...]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        hooks = {
            phase: tuple(phase_hooks)
            for phase, phase_hooks in dict(self.by_phase).items()
        }
        if any(not isinstance(phase, ModelHookPhase) for phase in hooks):
            raise TypeError("ModelHooks keys must be ModelHookPhase values")
        object.__setattr__(self, "by_phase", MappingProxyType(hooks))

    def with_additions(self, additions: ModelHooks) -> ModelHooks:
        phases = dict.fromkeys((*self.by_phase, *additions.by_phase))
        return ModelHooks(
            {
                phase: self.for_phase(phase) + additions.for_phase(phase)
                for phase in phases
            }
        )

    def for_phase(self, phase: ModelHookPhase) -> tuple[ModelHook, ...]:
        return self.by_phase.get(phase, ())


@dataclass(frozen=True)
class ModelFamilyTemplate:
    """Shared defaults for a family that follows the vanilla training path."""

    name: str
    is_multimodal: bool = False
    capabilities: Mapping[str, Capability] = field(default_factory=dict)
    strategies: ModelStrategies = field(default_factory=ModelStrategies)
    matchers: ModelMatchers = field(default_factory=ModelMatchers)
    hooks: ModelHooks = field(default_factory=ModelHooks)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )


@dataclass(frozen=True)
class ModelProfile:
    """Per-model declarations layered over a reusable family template."""

    family: ModelFamilyTemplate
    is_multimodal: bool | None = None
    capabilities: Mapping[str, Capability | None] = field(default_factory=dict)
    strategies: ModelStrategies = field(default_factory=ModelStrategies)
    matchers: ModelMatchers = field(default_factory=ModelMatchers)
    hooks: ModelHooks = field(default_factory=ModelHooks)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )


@dataclass(frozen=True)
class ResolvedModelProfile:
    """Immutable effective behavior for one registered model descriptor."""

    model_types: tuple[str, ...]
    family: str | None
    is_multimodal: bool
    capabilities: Mapping[str, Capability]
    strategies: ModelStrategies
    matchers: ModelMatchers
    hooks: ModelHooks

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "capabilities", MappingProxyType(dict(self.capabilities))
        )


def _declared_value(support: ModelSupport, name: str) -> tuple[bool, Any]:
    if name in vars(support):
        return True, vars(support)[name]
    for cls in type(support).__mro__:
        if cls is ModelSupport:
            return False, None
        if name in cls.__dict__:
            return True, cls.__dict__[name]
    return False, None


def _legacy_cfg_hook(
    support: ModelSupport,
    phase: ModelHookPhase,
    method: Callable[[DictDefault], None],
) -> ModelHook:
    def hook(context: ModelHookContext) -> None:
        token = _ACTIVE_LEGACY_HOOK.set((id(support), phase))
        try:
            method(context.cfg)
        finally:
            _ACTIVE_LEGACY_HOOK.reset(token)

    return hook


def _legacy_post_load_hook(
    support: ModelSupport,
    method: Callable[[DictDefault, Any], None],
) -> ModelHook:
    def hook(context: ModelHookContext) -> None:
        if context.model is None:
            raise ValueError("AFTER_ADAPTER_LOAD requires a model instance")
        token = _ACTIVE_LEGACY_HOOK.set(
            (id(support), ModelHookPhase.AFTER_ADAPTER_LOAD)
        )
        try:
            method(context.cfg, context.model)
        finally:
            _ACTIVE_LEGACY_HOOK.reset(token)

    return hook


def _resolve_declarative_model_support(
    support: ModelSupport,
) -> ResolvedModelProfile:
    profile = support.profile
    if profile is None:
        return ResolvedModelProfile(
            model_types=support.model_types,
            family=None,
            is_multimodal=False,
            capabilities={},
            strategies=ModelStrategies(),
            matchers=ModelMatchers(),
            hooks=ModelHooks(),
        )

    family = profile.family
    is_multimodal = (
        profile.is_multimodal
        if profile.is_multimodal is not None
        else family.is_multimodal
    )
    capabilities = dict(family.capabilities)
    for name, capability in profile.capabilities.items():
        if capability is None:
            capabilities.pop(name, None)
        else:
            capabilities[name] = capability

    return ResolvedModelProfile(
        model_types=support.model_types,
        family=family.name,
        is_multimodal=is_multimodal,
        capabilities=capabilities,
        strategies=family.strategies.with_overrides(profile.strategies),
        matchers=family.matchers.with_overrides(profile.matchers),
        hooks=family.hooks.with_additions(profile.hooks),
    )


def _run_model_profile_hooks(
    support: ModelSupport,
    phase: ModelHookPhase,
    context: ModelHookContext,
) -> None:
    if _ACTIVE_LEGACY_HOOK.get() == (id(support), phase):
        return
    resolved = _resolve_declarative_model_support(support)
    for hook in resolved.hooks.for_phase(phase):
        hook(context)


@overload
def resolve_model_support(support: None) -> None: ...


@overload
def resolve_model_support(support: ModelSupport) -> ResolvedModelProfile: ...


def resolve_model_support(
    support: ModelSupport | None,
) -> ResolvedModelProfile | None:
    """Resolve family defaults, profile overrides, and legacy declarations."""
    if support is None:
        return None

    declarative = _resolve_declarative_model_support(support)
    is_multimodal = declarative.is_multimodal
    capabilities = dict(declarative.capabilities)
    strategies = declarative.strategies
    matchers = declarative.matchers
    hooks = declarative.hooks

    declares_multimodal, _ = _declared_value(support, "is_multimodal")
    if declares_multimodal:
        is_multimodal = bool(support.is_multimodal)
    declares_capabilities, _ = _declared_value(support, "capabilities")
    if declares_capabilities:
        capabilities.update(support.capabilities)
    declares_auto_model, _ = _declared_value(support, "get_auto_model_cls")
    if declares_auto_model:
        strategies = strategies.with_overrides(
            ModelStrategies(auto_model_cls=support.get_auto_model_cls)
        )
    declares_processing, _ = _declared_value(support, "get_processing_strategy_cls")
    if declares_processing:
        strategies = strategies.with_overrides(
            ModelStrategies(processing_strategy_cls=support.get_processing_strategy_cls)
        )
    declares_cfg_matcher, _ = _declared_value(support, "matches_cfg")
    if declares_cfg_matcher:
        matchers = matchers.with_overrides(ModelMatchers(cfg=support.matches_cfg))
    declares_processor_matcher, _ = _declared_value(support, "matches_processor")
    if declares_processor_matcher:
        matchers = matchers.with_overrides(
            ModelMatchers(processor=support.matches_processor)
        )

    legacy_hooks: dict[ModelHookPhase, tuple[ModelHook, ...]] = {}
    legacy_phases = (
        (ModelHookPhase.BEFORE_CONFIG_LOAD, "pre_config_load"),
        (ModelHookPhase.CONFIGURE_RUN, "validate_cfg"),
        (ModelHookPhase.BEFORE_TOKENIZER_LOAD, "pre_tokenizer_load"),
        (ModelHookPhase.BEFORE_MODEL_BUILD, "pre_model_load"),
    )
    for phase, method_name in legacy_phases:
        declares_method, _ = _declared_value(support, method_name)
        if declares_method:
            legacy_hooks[phase] = (
                _legacy_cfg_hook(support, phase, getattr(support, method_name)),
            )
    declares_ready_hook, _ = _declared_value(support, "post_model_load")
    if declares_ready_hook:
        legacy_hooks[ModelHookPhase.AFTER_ADAPTER_LOAD] = (
            _legacy_post_load_hook(support, support.post_model_load),
        )
    hooks = hooks.with_additions(ModelHooks(legacy_hooks))

    return ResolvedModelProfile(
        model_types=support.model_types,
        family=declarative.family,
        is_multimodal=is_multimodal,
        capabilities=capabilities,
        strategies=strategies,
        matchers=matchers,
        hooks=hooks,
    )


def run_model_support_hooks(
    support: ModelSupport | None,
    phase: ModelHookPhase,
    context: ModelHookContext,
) -> None:
    """Run the effective hooks for one explicit lifecycle phase."""
    if support is None:
        return
    resolved = resolve_model_support(support)
    for hook in resolved.hooks.for_phase(phase):
        hook(context)
