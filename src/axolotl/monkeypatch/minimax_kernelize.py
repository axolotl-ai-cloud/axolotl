"""Fix for transformers' MiniMax M2 ``kernelize()`` crash under ``use_kernels``.

MiniMax M2 attention registers ``apply_rotary_pos_emb`` (a plain function) in
``_hidden_kernels``, so ``kernelize()`` tries to ``register_module()`` a
non-``nn.Module`` and raises ``ValueError``. ``forward`` calls the module-level
``apply_rotary_pos_emb`` directly, so dropping that entry is numerically neutral.

The patch wraps the attention ``__init__`` to strip non-Module ``_hidden_kernels``
entries; Module entries are left intact. Idempotent; install before model build.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_TARGETS = [
    ("transformers.models.minimax_m2.modeling_minimax_m2", "MiniMaxM2Attention"),
]

_PATCHED_CLASSES: list[type] = []


def _strip_non_module_kernels(orig_init):
    """Wrap ``__init__`` to drop non-Module ``_hidden_kernels`` after construction."""
    import torch.nn as nn

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        hidden_kernels = self.__dict__.get("_hidden_kernels") or {}
        for name in [
            n for n, fn in hidden_kernels.items() if not isinstance(fn, nn.Module)
        ]:
            del hidden_kernels[name]

    init._axolotl_original = orig_init  # type: ignore[attr-defined]
    return init


def patch_minimax_kernelize() -> bool:
    """Strip dead non-Module ``_hidden_kernels`` entries on MiniMax attention.

    Returns ``True`` if at least one target class was patched (or already was),
    ``False`` if none could be imported (e.g. transformers predates MiniMax).
    """
    import importlib

    patched_any = False
    for module_path, attr in _TARGETS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(module, attr, None)
        if cls is None:
            continue
        patched_any = True
        if cls in _PATCHED_CLASSES:
            continue

        cls.__init__ = _strip_non_module_kernels(cls.__init__)
        _PATCHED_CLASSES.append(cls)
        LOG.info(
            "minimax_kernelize: patched %s to drop non-Module _hidden_kernels "
            "entries so use_kernels/kernelize() does not crash",
            attr,
        )

    return patched_any


def unpatch_minimax_kernelize() -> None:
    """Restore the original attention ``__init__``\\ s. Useful for tests."""
    for cls in list(_PATCHED_CLASSES):
        original = getattr(cls.__init__, "_axolotl_original", None)
        if original is not None:
            cls.__init__ = original
        _PATCHED_CLASSES.remove(cls)
