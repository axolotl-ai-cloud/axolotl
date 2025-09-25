"""Tests for PatchManager context parallel patch selection."""

import addict

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault


def _stub_transformers_patches(monkeypatch):
    """Replace trainer loss patchers with no-ops for isolation."""
    monkeypatch.setattr(
        "axolotl.monkeypatch.transformers.trainer_loss_calc.patch_evaluation_loop",
        lambda: None,
    )
    monkeypatch.setattr(
        "axolotl.monkeypatch.transformers.trainer_loss_calc.patch_maybe_log_save_evaluate",
        lambda: None,
    )


def test_patch_manager_applies_flash_cp_patch(monkeypatch):
    """When flash attention is enabled, we patch Trainer for CP."""
    _stub_transformers_patches(monkeypatch)

    patch_calls = {"count": 0}

    def stub_patch():
        patch_calls["count"] += 1

    monkeypatch.setattr(
        "axolotl.monkeypatch.transformers.trainer_context_parallel.patch_prepare_context_parallel_inputs",
        stub_patch,
    )

    cfg = DictDefault(
        {
            "context_parallel_size": 2,
            "flash_attention": True,
            "sdp_attention": False,
        }
    )

    manager = PatchManager(cfg, addict.Dict())
    manager._apply_transformers_patches()

    assert patch_calls["count"] == 1


def test_patch_manager_skips_flash_patch_for_sdpa(monkeypatch):
    """When only SDPA is requested, we should not patch Trainer."""
    _stub_transformers_patches(monkeypatch)

    patch_calls = {"count": 0}

    def stub_patch():
        patch_calls["count"] += 1

    monkeypatch.setattr(
        "axolotl.monkeypatch.transformers.trainer_context_parallel.patch_prepare_context_parallel_inputs",
        stub_patch,
    )

    cfg = DictDefault(
        {
            "context_parallel_size": 2,
            "flash_attention": False,
            "sdp_attention": True,
        }
    )

    manager = PatchManager(cfg, addict.Dict())
    manager._apply_transformers_patches()

    assert patch_calls["count"] == 0
