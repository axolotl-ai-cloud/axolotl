"""Tests for the auto-deferral of the load-time fp32 embedding upcast.

ProTrain handles the embed / lm_head fp32 upcast lazily during forward, so
the load-time upcast in
``axolotl.loaders.model.ModelLoader._convert_embedding_modules_dtype`` is
redundant under ProTrain and OOMs 27B + 4-bit on a single 24 GiB card.

When ProTrain is registered in ``cfg.plugins`` the loader now skips the
load-time upcast automatically — the legacy ``embeddings_skip_upcast: true``
YAML knob is no longer required for that path.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from axolotl.loaders.model import ModelLoader
from axolotl.loaders.utils import is_protrain_active

_PROTRAIN_PLUGIN_ID = "axolotl.integrations.protrain.ProTrainPlugin"


class _ConfigSpy:
    """Minimal cfg-shaped object that supports both attribute access and ``get``.

    ``ModelLoader._configure_embedding_dtypes`` reads ``self.cfg`` via dotted
    attribute access. Only the attributes it touches need to be present.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _StubLoader(ModelLoader):
    """Subclass that overrides the FSDP-detection properties so the test
    doesn't need to construct a real model or accelerator state."""

    is_fsdp_enabled = False  # type: ignore[assignment]
    is_qlora_and_fsdp_enabled = False  # type: ignore[assignment]


def _make_loader(cfg) -> ModelLoader:
    """Build a stubbed ``ModelLoader`` without invoking ``__init__``."""
    loader = _StubLoader.__new__(_StubLoader)
    loader.cfg = cfg
    loader.model = MagicMock()
    loader.model_config = SimpleNamespace(model_type="llama")
    return loader


# ----- is_protrain_active --------------------------------------------------


def test_is_protrain_active_true_when_plugin_listed() -> None:
    cfg = _ConfigSpy(plugins=[_PROTRAIN_PLUGIN_ID])
    assert is_protrain_active(cfg) is True


def test_is_protrain_active_false_when_plugin_absent() -> None:
    cfg = _ConfigSpy(plugins=["some.other.Plugin"])
    assert is_protrain_active(cfg) is False


def test_is_protrain_active_false_with_no_plugins_attr() -> None:
    cfg = _ConfigSpy()
    assert is_protrain_active(cfg) is False


def test_is_protrain_active_false_with_none_plugins() -> None:
    cfg = _ConfigSpy(plugins=None)
    assert is_protrain_active(cfg) is False


def test_is_protrain_active_rejects_non_string_entries() -> None:
    cfg = _ConfigSpy(plugins=[object(), 42])
    assert is_protrain_active(cfg) is False


# ----- ModelLoader._configure_embedding_dtypes -----------------------------


def _capture_upcast_call(loader: ModelLoader):
    """Replace ``_convert_embedding_modules_dtype`` with a recorder and run
    ``_configure_embedding_dtypes``; return the recorded calls."""
    calls: list[dict] = []

    def _record(embedding_modules, dist_dtype, before_kbit_train_or_finetune):
        calls.append(
            {
                "embedding_modules": list(embedding_modules),
                "dist_dtype": dist_dtype,
                "before_kbit_train_or_finetune": before_kbit_train_or_finetune,
            }
        )

    loader._convert_embedding_modules_dtype = _record  # type: ignore[assignment]
    loader._prepare_model_for_quantization = lambda: None  # type: ignore[assignment]
    loader._set_z3_leaf_modules = lambda: None  # type: ignore[assignment]
    loader._configure_embedding_dtypes()
    return calls


def test_auto_defer_when_protrain_in_plugins() -> None:
    """ProTrain in plugins + 4-bit => load-time upcast list is empty."""
    cfg = _ConfigSpy(
        plugins=[_PROTRAIN_PLUGIN_ID],
        load_in_4bit=True,
        embeddings_skip_upcast=None,  # legacy knob NOT set
        model_config_type="llama",
        adapter="qlora",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        flash_attention=False,
        flex_attention=False,
        sage_attention=False,
        cut_cross_entropy=False,
        torch_dtype="bfloat16",
    )
    loader = _make_loader(cfg)
    calls = _capture_upcast_call(loader)

    pre_kbit = [c for c in calls if c["before_kbit_train_or_finetune"]]
    assert pre_kbit, "expected at least one pre-kbit upcast invocation"
    assert pre_kbit[0]["embedding_modules"] == [], (
        "ProTrain-active path must clear the embedding-module list so the "
        "load-time fp32 upcast is a no-op"
    )


def test_existing_upcast_still_runs_without_protrain() -> None:
    """Without ProTrain (and without the legacy knob), the upcast still runs."""
    cfg = _ConfigSpy(
        plugins=[],
        load_in_4bit=True,
        embeddings_skip_upcast=None,
        model_config_type="llama",
        adapter="qlora",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        flash_attention=False,
        flex_attention=False,
        sage_attention=False,
        cut_cross_entropy=False,
        torch_dtype="bfloat16",
    )
    loader = _make_loader(cfg)
    calls = _capture_upcast_call(loader)

    pre_kbit = [c for c in calls if c["before_kbit_train_or_finetune"]]
    assert pre_kbit and pre_kbit[0]["embedding_modules"], (
        "default path (no ProTrain, no explicit skip) must still upcast embeds"
    )


def test_explicit_skip_knob_still_honored_without_protrain() -> None:
    """``embeddings_skip_upcast: true`` keeps working for non-ProTrain users."""
    cfg = _ConfigSpy(
        plugins=[],
        load_in_4bit=True,
        embeddings_skip_upcast=True,
        model_config_type="llama",
        adapter="qlora",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        flash_attention=False,
        flex_attention=False,
        sage_attention=False,
        cut_cross_entropy=False,
        torch_dtype="bfloat16",
    )
    loader = _make_loader(cfg)
    calls = _capture_upcast_call(loader)

    pre_kbit = [c for c in calls if c["before_kbit_train_or_finetune"]]
    assert pre_kbit and pre_kbit[0]["embedding_modules"] == [], (
        "explicit skip knob must clear embed-module list (back-compat)"
    )


def test_no_skip_when_4bit_disabled_even_with_protrain() -> None:
    """Auto-defer is 4-bit-specific; bf16 paths still upcast under ProTrain."""
    cfg = _ConfigSpy(
        plugins=[_PROTRAIN_PLUGIN_ID],
        load_in_4bit=False,
        embeddings_skip_upcast=None,
        model_config_type="llama",
        adapter="lora",
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        flash_attention=False,
        flex_attention=False,
        sage_attention=False,
        cut_cross_entropy=False,
        torch_dtype="bfloat16",
    )
    loader = _make_loader(cfg)
    calls = _capture_upcast_call(loader)

    pre_kbit = [c for c in calls if c["before_kbit_train_or_finetune"]]
    assert pre_kbit and pre_kbit[0]["embedding_modules"], (
        "bf16 ProTrain path must still upcast embeds at load time"
    )


# ----- PatchManager._apply_adapter_patches ---------------------------------


def test_patch_manager_fires_peft_patch_under_protrain(monkeypatch) -> None:
    """The peft prep-code patch fires when ProTrain is active (mirrors the
    historical ``embeddings_skip_upcast`` gate so the embed/lm_head upcast
    inside ``prepare_model_for_kbit_training`` is also skipped)."""
    from axolotl.loaders import patch_manager as pm_mod

    called: list[bool] = []
    monkeypatch.setattr(
        "axolotl.monkeypatch.peft.utils.patch_peft_prep_code",
        lambda: called.append(True),
    )

    pm = pm_mod.PatchManager.__new__(pm_mod.PatchManager)
    pm.cfg = _ConfigSpy(  # type: ignore[assignment]
        adapter="qlora",
        plugins=[_PROTRAIN_PLUGIN_ID],
        embeddings_skip_upcast=None,
    )
    pm._apply_adapter_patches()
    assert called == [True]


def test_patch_manager_skips_peft_patch_without_gate(monkeypatch) -> None:
    """No ProTrain, no explicit knob => peft prep-code patch does NOT fire."""
    from axolotl.loaders import patch_manager as pm_mod

    called: list[bool] = []
    monkeypatch.setattr(
        "axolotl.monkeypatch.peft.utils.patch_peft_prep_code",
        lambda: called.append(True),
    )

    pm = pm_mod.PatchManager.__new__(pm_mod.PatchManager)
    pm.cfg = _ConfigSpy(  # type: ignore[assignment]
        adapter="qlora",
        plugins=[],
        embeddings_skip_upcast=None,
    )
    pm._apply_adapter_patches()
    assert called == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-x", "-q"]))
