"""Tests for Muon parameter tagging helpers."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.muonclip import MuonClipController, tag_parameters_for_muon
from axolotl.utils.callbacks.muonclip import MuonClipCallback
from axolotl.utils.schemas.muon import MuonClipConfig


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.linear = nn.Linear(4, 4, bias=True)
        self.proj = nn.Linear(4, 2, bias=False)


class LlamaSdpaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.weight = nn.Parameter(torch.ones(1))


class _TinyLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = LlamaSdpaAttention()


class _DummyBuilder(TrainerBuilderBase):
    def build(self, total_num_steps):  # pragma: no cover - not used in tests
        return None


def _make_cfg(**overrides):
    defaults = {
        "muonclip": MuonClipConfig(),
        "optimizer": "muon",
        "learning_rate": 0.01,
        "gc_steps": None,
        "use_wandb": False,
        "use_mlflow": False,
        "use_comet": False,
        "use_otel_metrics": False,
        "save_first_step": False,
        "profiler_steps": None,
        "plugins": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_default_tagging_prefers_ndim_and_excludes_embed():
    model = _TinyModel()
    cfg = MuonClipConfig()

    metadata, summary = tag_parameters_for_muon(model, cfg)

    assert summary.total == len(list(model.named_parameters()))
    assert metadata["linear.weight"].use_muon
    assert metadata["proj.weight"].use_muon
    assert metadata["linear.bias"].use_muon is False
    assert metadata["linear.bias"].reason.startswith("ndim<")
    assert metadata["embed.weight"].use_muon is False
    assert metadata["embed.weight"].reason.startswith("exclude:")
    assert hasattr(model.linear.weight, "use_muon") and model.linear.weight.use_muon


def test_apply_to_forces_parameters_into_muon_bucket():
    model = _TinyModel()
    cfg = MuonClipConfig(apply_to=["bias"], exclude=[])

    metadata, summary = tag_parameters_for_muon(model, cfg, min_ndim=2)

    assert summary.muon == 4  # embed weight, linear weight, proj weight, forced bias
    assert metadata["linear.bias"].use_muon
    assert metadata["linear.bias"].reason.startswith("forced:")


def test_trainer_builder_initializes_muon_tags_once():
    cfg = _make_cfg()
    builder = _DummyBuilder(cfg, _TinyModel(), tokenizer=None)

    metadata_first, summary_first = builder._ensure_muon_param_tags()
    assert summary_first.muon >= 2
    assert "linear.weight" in metadata_first
    assert metadata_first["linear.weight"].use_muon

    metadata_second, summary_second = builder._ensure_muon_param_tags()
    assert metadata_first is metadata_second
    assert summary_first is summary_second


def test_builder_provides_muon_state_store():
    cfg = _make_cfg()
    builder = _DummyBuilder(cfg, _TinyModel(), tokenizer=None)

    store_first = builder._get_muon_state_store()
    assert store_first is builder._get_muon_state_store()


def test_builder_controller_lazily_constructed():
    cfg = _make_cfg(muonclip=MuonClipConfig(enabled=True))
    builder = _DummyBuilder(cfg, _TinyModel(), tokenizer=None)

    controller = builder._get_muon_controller()
    assert isinstance(controller, MuonClipController)
    assert controller is builder._get_muon_controller()


def test_builder_auto_registers_attention_modules_when_qk_clip_enabled():
    cfg = _make_cfg(muonclip=MuonClipConfig(enabled=True, qk_clip=True))
    builder = _DummyBuilder(cfg, _TinyLlamaModel(), tokenizer=None)

    controller = builder._get_muon_controller()
    assert controller.attention_trackers


def test_get_callbacks_includes_muonclip_callback():
    cfg = _make_cfg(muonclip=MuonClipConfig(enabled=True, qk_clip=False))
    builder = _DummyBuilder(cfg, _TinyModel(), tokenizer=None)

    callbacks = builder.get_callbacks()
    assert any(isinstance(cb, MuonClipCallback) for cb in callbacks)


def test_qk_clip_warns_when_no_supported_attention(caplog):
    cfg = _make_cfg(muonclip=MuonClipConfig(enabled=True, qk_clip=True))
    builder = _DummyBuilder(cfg, _TinyModel(), tokenizer=None)

    with caplog.at_level("WARNING"):
        builder._get_muon_controller()

    assert any("no supported attention modules" in rec.message for rec in caplog.records)
