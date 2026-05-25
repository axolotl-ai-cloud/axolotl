"""Unit tests for the runtime version-gate warning in ``protrain.check``.

Mirrors the CI workflow gate (``.github/workflows/protrain-version-check.yml``)
on the runtime side: the plugin's ``pre_model_load`` calls
``warn_on_unvalidated_versions`` so users running against a transformers /
peft combo above the validated bounds see a ``UserWarning`` at startup.
"""

from __future__ import annotations

import sys
import warnings

import pytest

from axolotl.integrations.protrain.check import (
    VALIDATED_PEFT_MAX,
    VALIDATED_TRANSFORMERS_MAX,
    warn_on_unvalidated_versions,
)


def _patch_versions(
    monkeypatch: pytest.MonkeyPatch,
    transformers_version: str,
    peft_version: str,
) -> None:
    """Patch ``__version__`` on the live ``sys.modules`` entries.

    The shared ``cleanup_monkeypatches`` fixture (``tests/conftest.py``) calls
    ``importlib.reload`` on ``transformers`` between tests, swapping
    ``sys.modules['transformers']`` for a fresh module object. Patching a
    test-module-level ``import transformers`` therefore mutates a stale
    reference that ``warn_on_unvalidated_versions``'s inner ``import
    transformers`` never sees. Resolve the live module via ``sys.modules``
    instead.
    """
    transformers_mod = sys.modules["transformers"]
    peft_mod = sys.modules["peft"]
    monkeypatch.setattr(
        transformers_mod, "__version__", transformers_version, raising=False
    )
    monkeypatch.setattr(peft_mod, "__version__", peft_version, raising=False)


def test_warn_on_unvalidated_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched future transformers version triggers the matching UserWarning."""
    # Pin peft below its validated max so only the transformers branch fires.
    _patch_versions(monkeypatch, "5.9.0", "0.19.0")
    with pytest.warns(UserWarning, match="transformers 5.9.0 exceeds validated"):
        warn_on_unvalidated_versions()


def test_warn_on_unvalidated_peft(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patched future peft version triggers the matching UserWarning."""
    _patch_versions(monkeypatch, "5.5.4", "0.21.0")
    with pytest.warns(UserWarning, match="peft 0.21.0 exceeds validated"):
        warn_on_unvalidated_versions()


def test_no_warn_at_validated_version(monkeypatch: pytest.MonkeyPatch) -> None:
    """Known-good (below max) versions emit no warnings."""
    _patch_versions(monkeypatch, "5.5.4", "0.19.1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unvalidated_versions()


def test_validated_constants_present() -> None:
    """The CI gate reads these constants from check.py; surface them as a sanity check."""
    assert VALIDATED_TRANSFORMERS_MAX == "5.9"
    assert VALIDATED_PEFT_MAX == "0.21"
