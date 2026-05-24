"""Tests for ``assert_supported_peft_transformers_surface``.

Probes the startup-time guard that fails loud at config time if PEFT or
transformers drift away from the API surface ProTrain depends on
(``LoraLayer.adapter_layer_names`` and ``Trainer._load_from_checkpoint``).
"""

from __future__ import annotations

import pytest

from axolotl.integrations.protrain.check import (
    assert_supported_peft_transformers_surface,
)


def test_assert_passes_with_installed_versions() -> None:
    """Sanity: installed peft / transformers expose the expected surface."""
    assert_supported_peft_transformers_surface()


def test_assert_raises_on_missing_lora_layer_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing ``LoraLayer.adapter_layer_names`` must surface as RuntimeError."""
    from peft.tuners.lora import LoraLayer

    # The attribute is also defined on BaseTunerLayer (LoraLayer's MRO parent);
    # delete from every class in the MRO that owns it so hasattr returns False.
    for cls in LoraLayer.__mro__:
        if "adapter_layer_names" in cls.__dict__:
            monkeypatch.delattr(cls, "adapter_layer_names", raising=True)

    with pytest.raises(RuntimeError) as excinfo:
        assert_supported_peft_transformers_surface()

    msg = str(excinfo.value)
    assert "peft.tuners.lora.LoraLayer.adapter_layer_names" in msg
    assert "Validated upper bounds" in msg
    assert "peft=" in msg
    assert "transformers=" in msg


def test_assert_raises_on_missing_trainer_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing ``Trainer._load_from_checkpoint`` must surface as RuntimeError."""
    from transformers import Trainer

    monkeypatch.delattr(Trainer, "_load_from_checkpoint", raising=True)

    with pytest.raises(RuntimeError) as excinfo:
        assert_supported_peft_transformers_surface()

    msg = str(excinfo.value)
    assert "transformers.Trainer._load_from_checkpoint" in msg
    assert "Validated upper bounds" in msg
