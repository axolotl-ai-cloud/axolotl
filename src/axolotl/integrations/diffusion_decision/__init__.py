"""Typed decision training for native diffusion language models.

Exports resolve lazily so that importing a submodule (for example the light
``canvas`` path used by serving) does not import ``plugin`` and, through it,
axolotl's training stack.
"""

import importlib

__ci_config_keys__ = ("diffusion_decision",)

_LAZY_IMPORTS = {
    "DecisionCanvas": ".records",
    "DiffusionDecisionArgs": ".args",
    "DiffusionDecisionConfig": ".args",
    "DiffusionDecisionPlugin": ".plugin",
    "SlotInit": ".slots",
}

__all__ = [
    "DecisionCanvas",
    "DiffusionDecisionArgs",
    "DiffusionDecisionConfig",
    "DiffusionDecisionPlugin",
    "SlotInit",
]


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
