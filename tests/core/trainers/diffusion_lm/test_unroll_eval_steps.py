"""``diffusion_lm.unroll.eval_steps``: a fixed read count whenever the model is not training."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from axolotl.core.trainers.diffusion_lm.trainer import AxolotlDiffusionTrainer
from axolotl.utils.schemas.diffusion import DiffusionLMConfig, DiffusionUnrollConfig


class _Trainer:
    _native_unroll_steps = AxolotlDiffusionTrainer._native_unroll_steps

    def __init__(self, eval_steps, training=True, sampled=2):
        self._unroll = SimpleNamespace(k_max=2, eval_steps=eval_steps)
        self.model = torch.nn.Linear(1, 1).train(training)
        self._sampled = sampled

    def _native_value(self, name, default=None):
        return self._unroll if name == "unroll" else default

    def _sample_native_unroll_steps(self, k_max, device):
        del k_max, device
        return self._sampled


def test_eval_steps_pins_reads_only_when_not_training():
    device = torch.device("cpu")
    assert _Trainer(1)._native_unroll_steps(2, device, training=True) == 2
    assert _Trainer(1)._native_unroll_steps(2, device, training=False) == 1
    assert _Trainer(2, sampled=1)._native_unroll_steps(2, device, False) == 2
    # Unset keeps the training draw in eval too (previous behaviour).
    assert _Trainer(None)._native_unroll_steps(2, device, training=False) == 2
    # Without an explicit flag the trainer's model mode decides.
    assert _Trainer(1, training=False)._native_unroll_steps(2, device) == 1
    assert _Trainer(1, training=True)._native_unroll_steps(2, device) == 2


def test_eval_steps_beyond_k_max_is_refused():
    with pytest.raises(ValueError, match="eval_steps"):
        _Trainer(3)._native_unroll_steps(2, torch.device("cpu"), training=False)


def test_eval_steps_schema_bounds():
    assert DiffusionUnrollConfig(k_max=2, eval_steps=1).eval_steps == 1
    assert DiffusionUnrollConfig(k_max=2).eval_steps is None
    for bad in (0, 3):
        with pytest.raises(ValidationError):
            DiffusionUnrollConfig(k_max=2, eval_steps=bad)
    cfg = DiffusionLMConfig(unroll={"k_max": 2, "eval_steps": 1})
    assert cfg.unroll.eval_steps == 1
