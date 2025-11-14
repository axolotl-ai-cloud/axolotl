"""MuonClip controller handling Muon orthogonalization and QK-Clip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn

from axolotl.muonclip.attention import AttentionTracker, register_attention_module
from axolotl.muonclip.math import muon_orthogonal_update
from axolotl.muonclip.parameters import tag_parameters_for_muon
from axolotl.muonclip.state import MuonStateStore
from axolotl.muonclip.zero_utils import gather_full_param
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.muon import MuonClipConfig

LOG = get_logger(__name__)


@dataclass
class MuonClipContext:
    """Holds model, config, and tagging metadata for the controller."""

    model: nn.Module
    config: MuonClipConfig
    metadata: Mapping[str, object]
    state_store: MuonStateStore


class MuonClipController:
    """Responsible for executing Muon orthogonalization and QK-Clip after optimizer.step()."""

    def __init__(
        self,
        model: nn.Module,
        config: MuonClipConfig,
        *,
        state_store: MuonStateStore | None = None,
        learning_rate: float | None = None,
    ):
        self.model = model
        self.config = config
        metadata, _ = tag_parameters_for_muon(model, config)
        self.metadata = metadata
        ref_param = next(model.parameters())
        self.state_store = state_store or MuonStateStore(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        self.attention_trackers: dict[str, AttentionTracker] = {}
        self.attention_modules: dict[str, nn.Module] = {}
        self.learning_rate = learning_rate or 1e-4
        self._steps = 0
        self._qk_clip_active = bool(config.qk_clip)

    def post_optimizer_step(self):
        if not self.config.enabled:
            return

        self._steps += 1
        self._apply_muon_updates()
        if self._qk_clip_active and self.attention_trackers:
            self._apply_qk_clip()
            self._maybe_deactivate_qk_clip()

    def register_attention(self, module: nn.Module, *, name: str, num_heads: int):
        tracker = register_attention_module(module, name=name, num_heads=num_heads)
        self.attention_trackers[name] = tracker
        self.attention_modules[name] = module
        return tracker

    def _apply_muon_updates(self):
        for name, param in self.model.named_parameters():
            info = self.metadata.get(name)
            if not info or not info.use_muon or param.grad is None:
                continue

            with gather_full_param(param):
                state = self.state_store.get_or_create(param)
                update = muon_orthogonal_update(
                    param.grad.detach(),
                    state.momentum,
                    beta=self.config.momentum,
                    ns_steps=self.config.ns_steps,
                    rms_scale=self.config.rms_scale,
                )
                if self.config.weight_decay:
                    param.data.mul_(1 - self.learning_rate * self.config.weight_decay)
                param.data.add_(update, alpha=-self.learning_rate)

    def _apply_qk_clip(self):
        tau = self.config.qk_clip_tau
        alpha = self.config.qk_clip_alpha
        for name, tracker in self.attention_trackers.items():
            if not tracker.active:
                continue
            module = self.attention_modules.get(name)
            if module is None:
                continue
            buffer = getattr(module, tracker.buffer_name, None)
            if buffer is None:
                continue
            max_logits = buffer.detach()
            eta = (tau / max_logits.clamp(min=1e-6)).clamp(max=1.0)
            if torch.allclose(eta, torch.ones_like(eta)):
                buffer.zero_()
                continue
            if hasattr(module, "q_proj"):
                with gather_full_param(module.q_proj.weight):
                    self._scale_projection(module.q_proj.weight, eta, alpha)
            if hasattr(module, "k_proj"):
                with gather_full_param(module.k_proj.weight):
                    self._scale_projection(module.k_proj.weight, eta, 1 - alpha)
            buffer.zero_()

    def _scale_projection(self, weight: torch.Tensor, eta: torch.Tensor, exponent: float):
        if exponent == 0:
            return
        num_heads = eta.numel()
        head_dim = weight.size(0) // num_heads
        with torch.no_grad():
            reshaped = weight.view(num_heads, head_dim, weight.size(1))
            scale = eta.view(num_heads, 1, 1).pow(exponent)
            reshaped.mul_(scale)

    def _maybe_deactivate_qk_clip(self):
        max_steps = self.config.qk_clip_max_steps
        if not self._qk_clip_active or not max_steps:
            return
        if self._steps >= max_steps:
            self._deactivate_qk_clip(reason=f"reached qk_clip_max_steps={max_steps}")

    def _deactivate_qk_clip(self, *, reason: str | None = None):
        if not self._qk_clip_active:
            return
        self._qk_clip_active = False
        for tracker in self.attention_trackers.values():
            tracker.active = False
        if reason:
            LOG.info("MuonClip QK-Clip deactivated (%s)", reason)
        else:
            LOG.info("MuonClip QK-Clip deactivated")
