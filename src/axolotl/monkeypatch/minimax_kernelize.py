"""Fix for transformers' MiniMax ``kernelize()`` crash under ``use_kernels``.

MiniMax attention decorates with ``@use_kernelized_func(apply_rotary_pos_emb)``
over a plain function, so ``kernelize()`` tries to ``register_module()`` a
non-``nn.Module`` and raises ``ValueError``. ``forward`` calls the module-level
``apply_rotary_pos_emb`` directly, so dropping the dead ``_hidden_kernels`` entry
is numerically neutral. Same bug as :mod:`axolotl.monkeypatch.gemma4_kernelize`.

The patch wraps each attention ``__init__`` to strip non-Module ``_hidden_kernels``
entries; Module entries are left intact. Idempotent; install before model build.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# (module path, attribute) pairs covering the MiniMax MoE line. ``minimax_m2`` is
# also the architecture for the M2.x point releases and MiniMax M3.
_TARGETS = [
    ("transformers.models.minimax_m2.modeling_minimax_m2", "MiniMaxM2Attention"),
    ("transformers.models.minimax.modeling_minimax", "MiniMaxAttention"),
]

_PATCHED_CLASSES: list[type] = []


def patch_minimax_kernelize() -> bool:
    """Strip dead non-Module ``_hidden_kernels`` entries on MiniMax attention.

    Returns ``True`` if at least one target class was patched (or already was),
    ``False`` if none could be imported (e.g. transformers predates MiniMax).
    """
    import importlib

    import torch.nn as nn

    patched_any = False
    for module_path, attr in _TARGETS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue

        cls = getattr(module, attr, None)
        if cls is None or cls in _PATCHED_CLASSES:
            patched_any = patched_any or cls in _PATCHED_CLASSES
            continue

        orig_init = cls.__init__

        def init(self, *args, _orig_init=orig_init, **kwargs):
            _orig_init(self, *args, **kwargs)
            hidden_kernels = self.__dict__.get("_hidden_kernels")
            if hidden_kernels:
                stale = [
                    name
                    for name, fn in hidden_kernels.items()
                    if not isinstance(fn, nn.Module)
                ]
                for name in stale:
                    del hidden_kernels[name]

        init._axolotl_original = orig_init  # type: ignore[attr-defined]
        cls.__init__ = init
        _PATCHED_CLASSES.append(cls)
        patched_any = True
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
