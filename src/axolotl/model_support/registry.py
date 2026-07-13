"""Registry mapping ``model_type`` to its `ModelSupport` descriptor."""

import importlib
import threading
from collections.abc import Iterator

from axolotl.utils.logging import get_logger

from .base import ModelSupport

LOG = get_logger(__name__)

# Built-in descriptors, imported lazily on first lookup so that importing
# axolotl.model_support stays cycle-free and cheap.
_BUILTIN_MODULES = (
    "axolotl.model_support.kimi_linear",
    "axolotl.model_support.paddleocr_vl",
)

_REGISTRY: dict[str, ModelSupport] = {}
_builtins_loaded = False
_loading_builtins = False
_builtins_lock = threading.RLock()


def _ensure_builtins() -> None:
    global _builtins_loaded, _loading_builtins  # pylint: disable=global-statement
    with _builtins_lock:
        if _builtins_loaded or _loading_builtins:
            return

        _loading_builtins = True
        try:
            for module in _BUILTIN_MODULES:
                importlib.import_module(module)
        except Exception:
            # Leave partial registrations intact so the failed import can be retried.
            raise
        else:
            _builtins_loaded = True
        finally:
            _loading_builtins = False


def _validate_model_types(support_cls: type[ModelSupport]) -> tuple[str, ...]:
    model_types = support_cls.model_types
    if not isinstance(model_types, tuple) or not model_types:
        raise ValueError(
            f"{support_cls.__name__}.model_types must be a non-empty tuple"
        )
    if any(
        not isinstance(model_type, str) or not model_type.strip()
        for model_type in model_types
    ):
        raise ValueError(
            f"{support_cls.__name__}.model_types must contain non-empty strings"
        )
    if len(set(model_types)) != len(model_types):
        raise ValueError(f"{support_cls.__name__}.model_types must be unique")
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
    raise ValueError(f"Ambiguous model support for {subject}: {names}")


def register_model_support(support_cls: type[ModelSupport]) -> type[ModelSupport]:
    """Class decorator registering a descriptor under each of its `model_types`.

    Out-of-tree architectures can call this from a plugin or any imported
    module; registering an already-covered ``model_type`` overrides the
    built-in descriptor.
    """
    model_types = _validate_model_types(support_cls)

    # Loading built-ins first makes last-registration-wins deterministic for plugins.
    _ensure_builtins()

    instance = support_cls()
    with _builtins_lock:
        for model_type in model_types:
            if model_type in _REGISTRY:
                LOG.warning(
                    "Overriding model support for %s with %s",
                    model_type,
                    support_cls.__name__,
                )
        _REGISTRY.update(dict.fromkeys(model_types, instance))
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
    model_type = getattr(cfg, "model_config_type", None)
    if model_type:
        support = get_model_support(model_type)
        if support is not None:
            return support

    _ensure_builtins()
    return _one_match(
        [support for support in _iter_unique_support() if support.matches_cfg(cfg)],
        "configuration",
    )
