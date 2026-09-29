"""GLM capability declarations and fused-attention lifecycle gates."""

from unittest.mock import Mock

import pytest
import torch

from axolotl.loaders import patch_manager
from axolotl.model_support import (
    VANILLA_CAUSAL_LM,
    ModelProfile,
    ModelSupport,
    Supported,
    Unsupported,
    get_model_support,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault


def test_verified_capabilities():
    support = get_model_support("glm4_moe_lite")
    assert support is not None
    profile = resolve_model_support(support)
    for name in ("liger", "cut_cross_entropy", "fused_attn_kernel"):
        assert isinstance(profile.capabilities[name], Supported)


def test_cpu_skips_patch(monkeypatch):
    from axolotl.monkeypatch.models.glm4_moe_lite import fused_attn

    patch = Mock()
    monkeypatch.setattr(fused_attn, "patch_glm4_moe_lite_fused_attn", patch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = DictDefault(model_config_type="glm4_moe_lite", fused_attn_kernel=True)
    patch_manager.PatchManager(cfg, None)._apply_model_support_pre_load_hook()
    patch.assert_not_called()


@pytest.mark.parametrize(
    "model_type, warns", [("glm4_moe_lite", False), ("llama", True)]
)
def test_warning_consults_profile_then_legacy_fallback(monkeypatch, model_type, warns):
    warning = Mock()
    monkeypatch.setattr(patch_manager.LOG, "warning", warning)
    cfg = DictDefault(model_config_type=model_type, fused_attn_kernel=True)
    patch_manager.PatchManager._warn_if_fused_attn_unsupported(cfg)
    assert warning.called is warns


def test_unsupported_fusion_rejected_before_hooks(monkeypatch):
    class NoFusion(ModelSupport):
        model_types = ("test_no_fusion",)
        profile = ModelProfile(
            family=VANILLA_CAUSAL_LM,
            capabilities={"fused_attn_kernel": Unsupported("No MLA patch")},
        )

    monkeypatch.setattr(patch_manager, "get_model_support", lambda _: NoFusion())
    hooks = Mock()
    monkeypatch.setattr(patch_manager, "run_model_support_hooks", hooks)
    cfg = DictDefault(model_config_type="test_no_fusion", fused_attn_kernel=True)
    manager = patch_manager.PatchManager(cfg, None)
    with pytest.raises(ValueError, match="No MLA patch"):
        manager._apply_model_support_pre_load_hook()
    hooks.assert_not_called()
    with pytest.raises(ValueError, match="No MLA patch"):
        manager._warn_if_fused_attn_unsupported(cfg)
    cfg.fused_attn_kernel = False
    manager._apply_model_support_pre_load_hook()
    hooks.assert_called_once()
