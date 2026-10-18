"""Compose the selected trainer class with feature and plugin mixins at build time.

Mixins go before the selected class in the MRO, so their overrides wrap the trainer's
own (including Axolotl's). They must call ``super()`` for anything they override. Core
features register config-gated mixins with :func:`register_trainer_mixin`; plugins
return theirs from ``BasePlugin.get_trainer_mixins``. Core mixins come first
(outermost), then plugin mixins in ``plugins:`` order.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from axolotl.utils.import_helper import get_cls_from_module_str
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MixinSpec = type | str

_CORE_MIXINS: list[tuple[MixinSpec, Callable[[Any], bool]]] = []
_COMPOSED: dict[tuple[type, tuple[type, ...]], type] = {}


def register_trainer_mixin(
    mixin: MixinSpec, when: Callable[[Any], bool] = lambda cfg: True
) -> None:
    """Add ``mixin`` to every trainer whose config satisfies ``when``."""
    if not any(existing is mixin for existing, _ in _CORE_MIXINS):
        _CORE_MIXINS.append((mixin, when))


def _resolve(spec: MixinSpec) -> type:
    mixin = get_cls_from_module_str(spec) if isinstance(spec, str) else spec
    if not isinstance(mixin, type):
        raise TypeError(f"Trainer mixin {spec!r} is not a class")
    return mixin


def trainer_mixins_for(cfg: Any) -> list[type]:
    """Core mixins enabled by ``cfg``, then every plugin's, in order."""
    from axolotl.integrations.base import PluginManager

    specs: list[MixinSpec] = [mixin for mixin, when in _CORE_MIXINS if when(cfg)]
    specs.extend(PluginManager.get_instance().get_trainer_mixins(cfg))
    return [_resolve(spec) for spec in specs]


def compose_trainer_cls(base: type, mixins: Iterable[type]) -> type:
    """Subclass ``base`` with ``mixins`` placed before it in the MRO.

    Mixins already in ``base``'s MRO are skipped and duplicates are dropped. The same
    base and mixins always give the same class, so ``isinstance`` and class identity
    stay stable across builds.
    """
    ordered: Sequence[type] = tuple(
        mixin for mixin in dict.fromkeys(mixins) if mixin not in base.__mro__
    )
    if not ordered:
        return base
    key = (base, tuple(ordered))
    if key in _COMPOSED:
        return _COMPOSED[key]
    try:
        composed = type(
            base.__name__,
            (*ordered, base),
            {
                "__module__": base.__module__,
                "__qualname__": base.__qualname__,
                "__doc__": base.__doc__,
                "_axolotl_composed_from": (base, tuple(ordered)),
            },
        )
    except TypeError as exc:
        names = ", ".join(mixin.__qualname__ for mixin in ordered)
        raise TypeError(
            f"Cannot compose {base.__qualname__} with trainer mixins {names}: {exc}"
        ) from exc
    LOG.info(
        f"Composed {base.__qualname__} with trainer mixins: "
        + ", ".join(mixin.__qualname__ for mixin in ordered)
    )
    _COMPOSED[key] = composed
    return composed
