"""Fix for transformers' Gemma 4 ``kernelize()`` crash under ``use_kernels``.

In transformers, ``Gemma4VisionAttention`` is decorated with
``@use_kernelized_func(apply_rotary_pos_emb)`` where ``apply_rotary_pos_emb``
is a **plain function**, not a ``@use_kernel_func_from_hub``-wrapped kernel
layer. That decorator stashes the bare function in each instance's
``_hidden_kernels`` dict.

When ``use_kernels=True`` (which axolotl's ``KernelsArgs`` force-enables for the
ScatterMoE path), ``from_pretrained`` calls ``model.kernelize()``, whose
``attach_hidden_kernels`` step does ``module.register_module(name, fn)`` for each
``_hidden_kernels`` entry. ``register_module`` rejects a non-``nn.Module``::

    TypeError: ...apply_rotary_pos_emb is not a Module subclass

and the ``finally``-block cleanup then raises the visible::

    AttributeError: 'Gemma4VisionAttention' object has no attribute apply_rotary_pos_emb

This is a transformers bug, not Gemma4-specific in spirit (qwen3_moe avoids it
by wrapping the func with ``@use_kernel_func_from_hub`` so a Module-like ``Func``
is registered). Notably, ``Gemma4VisionAttention.forward`` calls
``apply_multidimensional_rope`` and never references ``apply_rotary_pos_emb``, so
the registered entry is dead weight — dropping the non-Module ``_hidden_kernels``
entries makes ``kernelize()`` a no-op for vision attention with zero behavior
change.

The patch wraps ``Gemma4VisionAttention.__init__`` to strip any non-``nn.Module``
``_hidden_kernels`` entries after construction. Properly-wrapped (Module) entries,
including ones a fixed transformers might introduce, are left intact, so the patch
is forward-compatible. Idempotent; install before the model is built.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCH_APPLIED = False


def patch_gemma4_kernelize() -> bool:
    """Strip dead non-Module ``_hidden_kernels`` entries on ``Gemma4VisionAttention``.

    Returns ``True`` if the patch is installed (or already was), ``False`` if the
    target class could not be imported (e.g. transformers predates Gemma 4) — in
    which case nothing is done and the caller can continue unaffected.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return True

    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        LOG.debug(
            "gemma4_kernelize: transformers.models.gemma4 not importable, "
            "skipping. This is fine for non-Gemma4 training."
        )
        return False

    cls = getattr(modeling_gemma4, "Gemma4VisionAttention", None)
    if cls is None:
        LOG.warning(
            "gemma4_kernelize: modeling_gemma4 has no 'Gemma4VisionAttention', "
            "skipping. Transformers API may have changed."
        )
        return False

    import torch.nn as nn

    orig_init = cls.__init__

    def init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        hidden_kernels = self.__dict__.get("_hidden_kernels")
        if hidden_kernels:
            stale = [
                name
                for name, fn in hidden_kernels.items()
                if not isinstance(fn, nn.Module)
            ]
            for name in stale:
                del hidden_kernels[name]

    # Preserve the original for teardown / idempotency checks.
    init._axolotl_original = orig_init  # type: ignore[attr-defined]
    cls.__init__ = init
    _PATCH_APPLIED = True
    LOG.info(
        "gemma4_kernelize: patched Gemma4VisionAttention to drop non-Module "
        "_hidden_kernels entries so use_kernels/kernelize() does not crash"
    )
    return True


def unpatch_gemma4_kernelize() -> None:
    """Restore the original ``Gemma4VisionAttention.__init__``. Useful for tests."""
    global _PATCH_APPLIED
    if not _PATCH_APPLIED:
        return
    try:
        from transformers.models.gemma4 import modeling_gemma4
    except ImportError:
        _PATCH_APPLIED = False
        return
    cls = getattr(modeling_gemma4, "Gemma4VisionAttention", None)
    if cls is not None:
        original = getattr(cls.__init__, "_axolotl_original", None)
        if original is not None:
            cls.__init__ = original
    _PATCH_APPLIED = False
