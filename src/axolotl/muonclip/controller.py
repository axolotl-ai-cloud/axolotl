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
        try:
            ref_param = next(model.parameters())
        except StopIteration:  # pragma: no cover - models without parameters
            ref_param = None
        self.state_store = state_store or MuonStateStore(
            device=ref_param.device if ref_param else None,
            dtype=ref_param.dtype if ref_param else None,
        )
        self.attention_trackers: dict[str, AttentionTracker] = {}
        self.attention_modules: dict[str, nn.Module] = {}
        self.learning_rate = 1e-4 if learning_rate is None else learning_rate
        self._steps = 0
        self._qk_clip_active = bool(config.qk_clip)

    def post_optimizer_step(self, *, optimizer: torch.optim.Optimizer | None = None):
        if not self.config.enabled:
            return

        self._steps += 1
        lr_map = self._build_lr_map(optimizer)
        self._apply_muon_updates(lr_map)
        if self._qk_clip_active and self.attention_trackers:
            self._apply_qk_clip()
            self._maybe_deactivate_qk_clip()

    def register_attention(self, module: nn.Module, *, name: str, num_heads: int):
        tracker = register_attention_module(module, name=name, num_heads=num_heads)
        self.attention_trackers[name] = tracker
        self.attention_modules[name] = module
        return tracker

    def _build_lr_map(self, optimizer) -> dict[int, float] | None:
        if optimizer is None:
            return None
        param_groups = getattr(optimizer, "param_groups", None)
        if not param_groups:
            return None

        lr_map: dict[int, float] = {}
        for group in param_groups:
            lr = group.get("lr")
            if lr is None:
                continue
            try:
                lr_value = float(lr)
            except (TypeError, ValueError):
                continue
            for param in group.get("params", []):
                if param is None:
                    continue
                if getattr(param, "use_muon", False) and lr_value == 0:
                    continue
                lr_map[id(param)] = lr_value
        return lr_map or None

    def _resolve_learning_rate(
        self, param: torch.nn.Parameter, lr_map: dict[int, float] | None
    ) -> float:
        if lr_map:
            lr = lr_map.get(id(param))
            if lr is not None:
                return lr
        return self.learning_rate

    def _apply_muon_updates(self, lr_map: dict[int, float] | None):
        for name, param in self.model.named_parameters():
            info = self.metadata.get(name)
            if not info or not info.use_muon or param.grad is None:
                continue

            lr = self._resolve_learning_rate(param, lr_map)
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
                    param.data.mul_(1 - lr * self.config.weight_decay)
                param.data.add_(update, alpha=-lr)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Serialize Muon optimizer buffers keyed by parameter name.
        """

        result: dict[str, torch.Tensor] = {}
        params = dict(self.model.named_parameters())
        for name, info in self.metadata.items():
            if not info.use_muon:
                continue
            param = params.get(name)
            if param is None:
                continue
            state = self.state_store.peek(param)
            if state is None:
                continue
            if state.momentum is not None:
                result[f"{name}:momentum"] = self._clone_to_cpu(state.momentum)
            if state.rms is not None:
                result[f"{name}:rms"] = self._clone_to_cpu(state.rms)
        return result

    def load_state_dict(self, buffers: Mapping[str, torch.Tensor]) -> None:
        """
        Restore Muon optimizer buffers from `buffers`.
        """

        if not buffers:
            return

        params = dict(self.model.named_parameters())
        for key, tensor in buffers.items():
            try:
                name, kind = key.rsplit(":", 1)
            except ValueError:
                continue
            param = params.get(name)
            if param is None:
                continue
            with gather_full_param(param):
                state = self.state_store.get_or_create(param, with_rms=(kind == "rms"))
                target = state.momentum if kind == "momentum" else state.rms
                if target is None:
                    continue
                target.copy_(tensor.to(target.device, dtype=target.dtype))

    @staticmethod
    def _clone_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        clone = tensor.detach().clone()
        if clone.device.type != "cpu":
            clone = clone.to("cpu")
        return clone

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
