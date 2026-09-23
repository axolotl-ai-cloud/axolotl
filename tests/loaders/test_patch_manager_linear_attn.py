"""Tests that PatchManager threads packed-document boundaries into recurrent mixers."""

from unittest.mock import MagicMock

import pytest
import torch

from axolotl.loaders.patch_manager import PatchManager
from axolotl.utils.dict import DictDefault

GDN_PATCHES = {
    "qwen3_next": "axolotl.monkeypatch.models.qwen3_next.modeling.patch_qwen3_next_modeling_packing",
    "qwen3_5": "axolotl.monkeypatch.models.qwen3_5.modeling.patch_qwen3_5_modeling_packing",
    "qwen3_5_text": "axolotl.monkeypatch.models.qwen3_5.modeling.patch_qwen3_5_modeling_packing",
    "qwen3_5_moe": "axolotl.monkeypatch.models.qwen3_5.modeling.patch_qwen3_5_moe_modeling_packing",
    "qwen3_5_moe_text": "axolotl.monkeypatch.models.qwen3_5.modeling.patch_qwen3_5_moe_modeling_packing",
}

SEQ_IDX_INJECTED = {
    "lfm2": "Lfm2Model",
    "lfm2_moe": "Lfm2MoeModel",
    "bamba": "BambaModel",
}


def _manager(model_type, **overrides):
    cfg = DictDefault(
        {
            "model_config_type": model_type,
            "sample_packing": False,
            "batch_flattening": False,
            "context_parallel_size": 1,
            **overrides,
        }
    )
    return PatchManager(cfg, MagicMock())


@pytest.mark.parametrize("model_type,target", GDN_PATCHES.items())
@pytest.mark.parametrize("mode", ["sample_packing", "batch_flattening"])
def test_gdn_patch_applies_without_cuda(monkeypatch, model_type, target, mode):
    """The patch must not depend on a CUDA device being visible at load time."""
    calls = []
    monkeypatch.setattr(target, lambda: calls.append(model_type))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    _manager(model_type, **{mode: True})._apply_model_specific_patches()

    assert calls == [model_type]


@pytest.mark.parametrize("model_type,target", GDN_PATCHES.items())
def test_gdn_patch_skipped_when_unpacked(monkeypatch, model_type, target):
    calls = []
    monkeypatch.setattr(target, lambda: calls.append(model_type))

    _manager(model_type)._apply_model_specific_patches()

    assert calls == []


@pytest.mark.parametrize("model_type,cls_name", SEQ_IDX_INJECTED.items())
def test_seq_idx_injected_for_short_conv_and_mamba2_models(
    monkeypatch, model_type, cls_name
):
    pytest.importorskip(f"transformers.models.{model_type}")
    patched = []
    monkeypatch.setattr(
        "axolotl.monkeypatch.models.mamba_utils.patch_model_forward_seq_idx",
        lambda cls: patched.append(cls.__name__),
    )

    _manager(model_type, sample_packing=True)._apply_model_specific_patches()

    assert patched == [cls_name]


def test_ssm_hybrid_patch_applies_for_context_parallel_without_packing(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "axolotl.monkeypatch.models.falcon_h1.modeling.patch_falcon_h1_modeling_packing",
        lambda: calls.append("falcon_h1"),
    )

    _manager("falcon_h1", context_parallel_size=2)._apply_model_specific_patches()

    assert calls == ["falcon_h1"]
