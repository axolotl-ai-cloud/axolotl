"""Registry mapping ``model_type`` to its `ModelSupport` descriptor."""

import importlib
import threading
from collections.abc import Iterator
from importlib.metadata import entry_points

from axolotl.utils.logging import get_logger

from .base import ModelSupport

LOG = get_logger(__name__)

MODEL_SUPPORT_ENTRY_POINT_GROUP = "axolotl.model_support"

# Built-in descriptors, imported lazily on first lookup so that importing
# axolotl.model_support stays cycle-free and cheap.
_BUILTIN_MODULES = (
    "axolotl.model_support.bailing_hybrid",
    "axolotl.model_support.cohere_compass",
    "axolotl.model_support.diffusion_gemma",
    "axolotl.model_support.dream",
    "axolotl.model_support.glm4_moe_lite",
    "axolotl.model_support.k2_horizon",
    "axolotl.model_support.kimi_linear",
    "axolotl.model_support.mamba",
    "axolotl.model_support.muse_glimmer",
    "axolotl.model_support.nemotron_diffusion",
    "axolotl.model_support.nemotron_diffusion_vlm",
    "axolotl.model_support.paddleocr_vl",
    "axolotl.model_support.qwen3_5_moe",
    "axolotl.model_support.qwen4_exp",
)

_REGISTRY: dict[str, ModelSupport] = {}
_builtins_loaded = False
_loading_builtins = False
_builtins_lock = threading.RLock()


def _is_builtin_support_class(support_cls: type[ModelSupport]) -> bool:
    return any(
        support_cls.__module__ == module
        or support_cls.__module__.startswith(f"{module}.")
        for module in _BUILTIN_MODULES
    )


def _is_registered_model_support(support_cls: type[ModelSupport]) -> bool:
    model_types = _validate_model_types(support_cls)
    with _builtins_lock:
        return all(
            type(_REGISTRY.get(model_type)) is support_cls for model_type in model_types
        )


def _ensure_builtins() -> None:
    global _builtins_loaded, _loading_builtins  # pylint: disable=global-statement
    # Import outside the lock: holding it across imports deadlocks against a
    # thread mid-import of a builtin that re-enters via @register_model_support.
    with _builtins_lock:
        if _builtins_loaded or _loading_builtins:
            return
        _loading_builtins = True

    try:
        for module in _BUILTIN_MODULES:
            importlib.import_module(module)
        for entry_point in entry_points(group=MODEL_SUPPORT_ENTRY_POINT_GROUP):
            try:
                support_cls = entry_point.load()
            except Exception:  # pylint: disable=broad-exception-caught
                LOG.warning(
                    "Could not import model-support entry point '%s'; skipping it",
                    entry_point.value,
                    exc_info=True,
                )
                continue
            if not _is_registered_model_support(support_cls):
                register_model_support(support_cls)
    except Exception:
        # Leave partial registrations intact so the failed import can be retried.
        with _builtins_lock:
            _loading_builtins = False
        raise

    with _builtins_lock:
        _builtins_loaded = True
        _loading_builtins = False


def _validate_model_types(support_cls: type[ModelSupport]) -> tuple[str, ...]:
    # A non-descriptor class would poison registry-wide matcher scans for
    # unrelated models, so reject it at registration time.
    if not (isinstance(support_cls, type) and issubclass(support_cls, ModelSupport)):
        raise TypeError(
            f"register_model_support requires a ModelSupport subclass, "
            f"got {support_cls!r}"
        )
    # normalize legacy list declarations to the documented tuple
    raw_model_types = support_cls.model_types
    if not isinstance(raw_model_types, (list, tuple)) or not raw_model_types:
        raise ValueError(
            f"{support_cls.__name__}.model_types must be a non-empty tuple"
        )
    model_types = tuple(raw_model_types)
    if any(
        not isinstance(model_type, str) or not model_type.strip()
        for model_type in model_types
    ):
        raise ValueError(
            f"{support_cls.__name__}.model_types must contain non-empty strings"
        )
    if len(set(model_types)) != len(model_types):
        raise ValueError(f"{support_cls.__name__}.model_types must be unique")
    support_cls.model_types = model_types
    return model_types


def _iter_unique_support() -> Iterator[ModelSupport]:
    with _builtins_lock:
        supports = tuple(_REGISTRY.values())
    seen: set[int] = set()
    for support in supports:
        identity = id(support)
        if identity in seen:
            continue
        seen.add(identity)
        yield support


def _one_match(matches: list[ModelSupport], subject: str) -> ModelSupport | None:
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    names = ", ".join(type(support).__name__ for support in matches)
    raise ValueError(
        f"Ambiguous model support for {subject}: {names}. Narrow the "
        f"overlapping matchers, or register under an exact model_type "
        f"(exact lookup bypasses matcher discovery)."
    )


def register_model_support(support_cls: type[ModelSupport]) -> type[ModelSupport]:
    """Class decorator registering a descriptor under each of its `model_types`.

    Out-of-tree architectures can call this from a plugin or any imported
    module; registering an already-covered ``model_type`` overrides the
    built-in descriptor.
    """
    model_types = _validate_model_types(support_cls)

    # Loading built-ins first makes last-registration-wins deterministic for plugins.
    _ensure_builtins()

    with _builtins_lock:
        protected_model_types = {
            model_type
            for model_type in model_types
            if _builtins_loaded
            and _is_builtin_support_class(support_cls)
            and model_type in _REGISTRY
            and not _is_builtin_support_class(type(_REGISTRY[model_type]))
        }
    if len(protected_model_types) == len(model_types):
        return support_cls

    instance = support_cls()
    with _builtins_lock:
        for model_type in model_types:
            if model_type in protected_model_types:
                continue
            if model_type in _REGISTRY:
                LOG.warning(
                    "Overriding model support for %s with %s",
                    model_type,
                    support_cls.__name__,
                )
            _REGISTRY[model_type] = instance
    return support_cls


def get_model_support(model_type: str | None) -> ModelSupport | None:
    """Look up the descriptor for a ``model_type``; `None` if unregistered."""
    if not model_type:
        return None
    _ensure_builtins()
    return _REGISTRY.get(model_type)


def get_model_support_for_processor(processor) -> ModelSupport | None:
    """Look up a descriptor by multimodal processor instance."""
    _ensure_builtins()
    return _one_match(
        [
            support
            for support in _iter_unique_support()
            if support.matches_processor(processor)
        ],
        f"processor {type(processor).__name__}",
    )


def get_model_support_for_cfg(cfg) -> ModelSupport | None:
    """Look up a descriptor before the model config is loaded.

    An exact resolved ``model_config_type`` wins. Before that is available,
    descriptors can match on config fields such as the model name.
    """
    model_type = cfg.model_config_type
    if model_type:
        support = get_model_support(model_type)
        if support is not None:
            return support

    _ensure_builtins()
    return _one_match(
        [support for support in _iter_unique_support() if support.matches_cfg(cfg)],
        "configuration",
    )
