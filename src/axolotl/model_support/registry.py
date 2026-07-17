"""Registry mapping ``model_type`` to its `ModelSupport` descriptor."""

import importlib

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


def _ensure_builtins() -> None:
    global _builtins_loaded  # pylint: disable=global-statement
    if _builtins_loaded:
        return
    _builtins_loaded = True
    for module in _BUILTIN_MODULES:
        importlib.import_module(module)


def register_model_support(support_cls: type[ModelSupport]) -> type[ModelSupport]:
    """Class decorator registering a descriptor under each of its `model_types`.

    Out-of-tree architectures can call this from a plugin or any imported
    module; registering an already-covered ``model_type`` overrides the
    built-in descriptor.
    """
    if not support_cls.model_types:
        raise ValueError(f"{support_cls.__name__} must define `model_types`")

    instance = support_cls()
    for model_type in support_cls.model_types:
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
    for support in dict.fromkeys(_REGISTRY.values()):
        if support.matches_processor(processor):
            return support
    return None


def get_model_support_for_cfg(cfg) -> ModelSupport | None:
    """Look up a descriptor before the model config is loaded.

    Used for pre-config/pre-tokenizer patching, where ``model_type`` is not
    yet available and descriptors match on the config (e.g. model name).
    """
    _ensure_builtins()
    for support in dict.fromkeys(_REGISTRY.values()):
        if support.matches_cfg(cfg):
            return support
    return None
