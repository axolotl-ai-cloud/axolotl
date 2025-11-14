"""Smoke tests for the MuonClipController skeleton."""

import torch
import torch.nn as nn
import pytest

from axolotl.muonclip import MuonClipController
from axolotl.muonclip.attention import BUFFER_NAME
from axolotl.utils.schemas.muon import MuonClipConfig


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.num_heads = 2


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.attn = _FakeAttention()


class _DummyOptimizer:
    def __init__(self, params, lr: float):
        self.param_groups = [
            {
                "params": params,
                "lr": lr,
            }
        ]


def test_controller_initializes_metadata_and_state_store():
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True)

    controller = MuonClipController(model, cfg)

    assert controller.metadata, "Expected parameter metadata to be populated"
    assert controller.state_store is not None
    tracker = controller.register_attention(model.linear, name="attn", num_heads=1)
    assert tracker.name == "attn"
    assert hasattr(model.linear, BUFFER_NAME)
    controller.post_optimizer_step()  # currently a noop


def test_muon_update_applies_change():
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, momentum=0.9)
    controller = MuonClipController(model, cfg, learning_rate=0.01)

    before = model.linear.weight.clone()
    model.linear.weight.grad = torch.ones_like(model.linear.weight)
    controller.post_optimizer_step()
    after = model.linear.weight
    assert not torch.allclose(before, after)


def test_qk_clip_scales_projections():
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, qk_clip=True, qk_clip_tau=1.0, qk_clip_alpha=0.5)
    controller = MuonClipController(model, cfg, learning_rate=0.0)

    tracker = controller.register_attention(model.attn, name="attn", num_heads=2)
    buffer = getattr(model.attn, tracker.buffer_name)
    buffer.copy_(torch.tensor([2.0, 0.5]))

    before_q = model.attn.q_proj.weight.clone()
    controller.post_optimizer_step()
    after_q = model.attn.q_proj.weight
    assert not torch.allclose(before_q, after_q)


def test_qk_clip_deactivates_after_max_steps():
    model = _TinyModel()
    cfg = MuonClipConfig(
        enabled=True,
        qk_clip=True,
        qk_clip_tau=1.0,
        qk_clip_alpha=0.5,
        qk_clip_max_steps=1,
    )
    controller = MuonClipController(model, cfg, learning_rate=0.0)

    tracker = controller.register_attention(model.attn, name="attn", num_heads=2)
    buffer = getattr(model.attn, tracker.buffer_name)
    buffer.copy_(torch.tensor([2.0, 0.5]))

    before = model.attn.q_proj.weight.clone()
    controller.post_optimizer_step()
    after_first = model.attn.q_proj.weight.clone()
    assert not torch.allclose(before, after_first)

    buffer.copy_(torch.tensor([2.0, 0.5]))
    controller.post_optimizer_step()
    after_second = model.attn.q_proj.weight.clone()
    assert torch.allclose(after_first, after_second), "QK-Clip should be inactive after max steps"


def _run_muon_step_with_lr(lr_value: float) -> float:
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, momentum=0.0, weight_decay=0.0)
    controller = MuonClipController(model, cfg, learning_rate=1.0)
    param = model.linear.weight
    param.data.fill_(0.5)
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    optimizer = _DummyOptimizer([param], lr_value)
    controller.post_optimizer_step(optimizer=optimizer)
    delta = before - param.detach()
    return delta.norm().item()


def _run_muon_step_without_optimizer() -> float:
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, momentum=0.0, weight_decay=0.0)
    controller = MuonClipController(model, cfg, learning_rate=1.0)
    param = model.linear.weight
    param.data.fill_(0.5)
    param.grad = torch.ones_like(param)
    before = param.detach().clone()
    controller.post_optimizer_step()
    delta = before - param.detach()
    return delta.norm().item()


def test_controller_uses_optimizer_learning_rate_overrides():
    slow = _run_muon_step_with_lr(0.01)
    fast = _run_muon_step_with_lr(0.1)

    assert slow > 0
    assert fast > slow
    assert fast == pytest.approx(slow * 10, rel=1e-5)


def test_controller_uses_fallback_lr_when_optimizer_group_zero():
    baseline = _run_muon_step_without_optimizer()
    zero_lr = _run_muon_step_with_lr(0.0)

    assert baseline > 0
    assert zero_lr > 0
    assert zero_lr == pytest.approx(baseline, rel=1e-6)


def test_controller_state_dict_round_trip():
    model = _TinyModel()
    cfg = MuonClipConfig(enabled=True, momentum=0.9)
    controller = MuonClipController(model, cfg, learning_rate=0.05)

    param = model.linear.weight
    param.grad = torch.ones_like(param)
    controller.post_optimizer_step()

    saved_buffers = controller.state_dict()
    assert any(key.endswith(":momentum") for key in saved_buffers)

    momentum_before = controller.state_store.get_or_create(param).momentum.clone()
    controller.state_store.get_or_create(param).momentum.zero_()
    controller.load_state_dict(saved_buffers)
    momentum_after = controller.state_store.get_or_create(param).momentum

    assert torch.allclose(momentum_before, momentum_after)
