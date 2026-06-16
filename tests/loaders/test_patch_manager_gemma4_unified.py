"""Tests for ``PatchManager`` fused-attn dispatch over the gemma4 / gemma4_unified
families. These pin the unified-specific routing (namespace branch + auto-on with
no ``fused_attn_kernel`` gate) introduced when the two blocks were consolidated."""

import logging
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault


def _make_patch_manager(model_config_type, *, gradient_checkpointing, inference=False):
    """Minimal cfg that reaches the fused-attn dispatch. ``context_parallel_size``
    must be an int (the pre-CUDA span compares ``> 1``) and ``fused_attn_kernel``
    stays falsy to prove the unified path is auto-on without it."""
    cfg = DictDefault(
        {
            "model_config_type": model_config_type,
            "context_parallel_size": 1,
            "sample_packing": False,
            "gradient_checkpointing": gradient_checkpointing,
            "activation_offloading": False,
            "fused_attn_kernel": False,
            "fsdp_config": None,
        }
    )
    return PatchManager(cfg, MagicMock(), inference=inference)


@pytest.fixture
def spy_fused_patches(monkeypatch):
    """Replace the two fused-attn patch entry points (imported inside the
    dispatch) with call recorders, and force the CUDA-gated block to run."""
    import axolotl.monkeypatch.models.gemma4.fused_attn as regular_fa
    import axolotl.monkeypatch.models.gemma4_unified.fused_attn as unified_fa

    calls = {"unified": [], "regular": []}
    monkeypatch.setattr(
        unified_fa,
        "patch_gemma4_unified_fused_attn",
        lambda **kw: calls["unified"].append(kw),
    )
    monkeypatch.setattr(
        regular_fa,
        "patch_gemma4_fused_attn",
        lambda **kw: calls["regular"].append(kw),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return calls


class TestFusedAttnDispatch:
    @pytest.mark.parametrize(
        "model_config_type",
        ["gemma4_unified", "gemma4_unified_text"],
    )
    def test_unified_routes_to_unified_patch_auto_on(
        self, spy_fused_patches, model_config_type
    ):
        """Unified dispatches to the unified namespace and runs even though
        ``fused_attn_kernel`` is False (auto-on, like standard gemma4)."""
        pm = _make_patch_manager(model_config_type, gradient_checkpointing=True)
        pm._apply_model_specific_patches()

        assert spy_fused_patches["unified"] == [{"install_shared_kv_workaround": True}]
        assert spy_fused_patches["regular"] == []

    @pytest.mark.parametrize("model_config_type", ["gemma4", "gemma4_text"])
    def test_standard_gemma4_routes_to_regular_patch(
        self, spy_fused_patches, model_config_type
    ):
        pm = _make_patch_manager(model_config_type, gradient_checkpointing=False)
        pm._apply_model_specific_patches()

        assert spy_fused_patches["regular"] == [{"install_shared_kv_workaround": False}]
        assert spy_fused_patches["unified"] == []

    def test_shared_kv_workaround_off_without_checkpointing(self, spy_fused_patches):
        pm = _make_patch_manager("gemma4_unified", gradient_checkpointing=False)
        pm._apply_model_specific_patches()

        assert spy_fused_patches["unified"] == [{"install_shared_kv_workaround": False}]

    def test_shared_kv_workaround_off_during_inference(self, spy_fused_patches):
        pm = _make_patch_manager(
            "gemma4_unified", gradient_checkpointing=True, inference=True
        )
        pm._apply_model_specific_patches()

        assert spy_fused_patches["unified"] == [{"install_shared_kv_workaround": False}]


class TestWarnIfFusedAttnUnsupported:
    """``_FUSED_ATTN_KERNEL_SUPPORTED`` must include the unified model types so
    ``fused_attn_kernel: true`` does not warn (it is not a no-op for them)."""

    def _capture(self, caplog, cfg):
        logger = logging.getLogger("axolotl.loaders.patch_manager")
        logger.addHandler(caplog.handler)
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            PatchManager._warn_if_fused_attn_unsupported(cfg)
        finally:
            logger.removeHandler(caplog.handler)
            logger.setLevel(previous_level)

    @pytest.mark.parametrize(
        "model_config_type",
        ["gemma4_unified", "gemma4_unified_text"],
    )
    def test_no_warn_for_unified_types(self, caplog, model_config_type):
        self._capture(
            caplog,
            DictDefault(
                {"fused_attn_kernel": True, "model_config_type": model_config_type}
            ),
        )
        assert caplog.text == "", f"unexpected warning: {caplog.text}"

    def test_warns_for_unsupported_type(self, caplog):
        self._capture(
            caplog,
            DictDefault({"fused_attn_kernel": True, "model_config_type": "llama"}),
        )
        assert "llama" in caplog.text and "no-op" in caplog.text
