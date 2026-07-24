"""Per-architecture model support descriptors.

Each supported architecture gets one directory here containing a
`ModelSupport` descriptor (declarative capabilities + lifecycle hooks) and any
model-specific code it needs (processing strategy, patches). Features query
the registry instead of hardcoding ``model_type`` checks.
"""

from .base import (
    Capability,
    Experimental,
    ModelSupport,
    Supported,
    Unsupported,
    check_capability,
)
from .profile import (
    AutoModelClassProvider,
    ConfigMatcher,
    ModelFamilyTemplate,
    ModelHook,
    ModelHookContext,
    ModelHookPhase,
    ModelHooks,
    ModelMatchers,
    ModelProfile,
    ModelStrategies,
    ModelStrategyOverrides,
    ProcessingStrategyClassProvider,
    ProcessorMatcher,
    ResolvedModelProfile,
    resolve_model_support,
    run_model_support_hooks,
)
from .registry import (
    get_model_support,
    get_model_support_for_cfg,
    get_model_support_for_processor,
    register_model_support,
)
from .templates import IMAGE_TEXT_TO_TEXT, VANILLA_CAUSAL_LM

__all__ = [
    "AutoModelClassProvider",
    "Capability",
    "ConfigMatcher",
    "Experimental",
    "ModelSupport",
    "ModelFamilyTemplate",
    "ModelHook",
    "ModelHookContext",
    "ModelHookPhase",
    "ModelHooks",
    "ModelMatchers",
    "ModelProfile",
    "ModelStrategyOverrides",
    "ModelStrategies",
    "ProcessingStrategyClassProvider",
    "ProcessorMatcher",
    "ResolvedModelProfile",
    "Supported",
    "Unsupported",
    "check_capability",
    "get_model_support",
    "get_model_support_for_cfg",
    "get_model_support_for_processor",
    "register_model_support",
    "resolve_model_support",
    "run_model_support_hooks",
    "IMAGE_TEXT_TO_TEXT",
    "VANILLA_CAUSAL_LM",
]
