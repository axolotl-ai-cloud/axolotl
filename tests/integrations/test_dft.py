"""Unit tests for DFT (Dynamic Fine-Tuning) integration."""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.dft.args import DFTArgs
from axolotl.integrations.dft.dft_utils import apply_dft_weighting, compute_dft_loss
from axolotl.integrations.dft.patch import patch_compute_loss_for_dft


class TestDFTArgs:
    def test_defaults(self):
        args = DFTArgs()
        assert args.enable_dft_loss is False

    def test_custom(self):
        args = DFTArgs(enable_dft_loss=True)
        assert args.enable_dft_loss is True


class TestDFTUtils:
    def test_apply_dft_weighting_values(self):
        loss = torch.tensor([0.0, 1.0, 10.0], requires_grad=True)
        out = apply_dft_weighting(loss)

        expected = loss.detach() * torch.exp(-loss.detach())
        assert torch.allclose(out.detach(), expected, atol=1e-6)

        out.sum().backward()
        assert loss.grad is not None

    def test_compute_dft_loss_matches_manual(self):
        log4 = math.log(4.0)
        logits = torch.tensor(
            [
                [
                    [0.0, log4],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ],
            requires_grad=True,
        )
        labels = torch.tensor([[0, 1, 0, -100]])

        loss = compute_dft_loss(logits, labels)

        l0 = -math.log(0.8) * 0.8
        l1 = -math.log(0.5) * 0.5
        expected = (l0 + l1) / 2.0

        assert loss.item() == pytest.approx(expected, abs=1e-6)
        loss.backward()
        assert logits.grad is not None

    def test_compute_dft_loss_all_ignored_is_zero(self):
        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.full((1, 4), -100)

        loss = compute_dft_loss(logits, labels)
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

        loss.backward()
        assert logits.grad is not None
        assert torch.all(logits.grad == 0)


class TestDFTPatch:
    def test_patch_compute_loss_removes_labels(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()

        sentinel = object()
        original_compute_loss = MagicMock(return_value=sentinel)
        trainer.compute_loss = original_compute_loss

        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        logits = torch.zeros(1, 4, 2, requires_grad=True)
        labels = torch.tensor([[0, 1, 0, -100]])

        class DummyModel(torch.nn.Module):
            def forward(self, **kwargs):
                assert "labels" not in kwargs
                return SimpleNamespace(logits=logits)

        loss, outputs = trainer.compute_loss(
            DummyModel(),
            {"input_ids": torch.zeros(1, 4, dtype=torch.long), "labels": labels},
            return_outputs=True,
        )

        assert outputs.logits is logits
        assert isinstance(loss, torch.Tensor)
        assert original_compute_loss.call_count == 0

    def test_patch_compute_loss_falls_back_when_disabled(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=False,
            include_tkps=False,
            label_smoothing_factor=0.0,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()

        sentinel = object()
        original_compute_loss = MagicMock(return_value=sentinel)
        trainer.compute_loss = original_compute_loss
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        out = trainer.compute_loss(model=MagicMock(), inputs={})
        assert out is sentinel
        assert original_compute_loss.call_count == 1

    def test_patch_compute_loss_raises_on_label_smoothing(self):
        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            enable_dft_loss=True,
            include_tkps=False,
            label_smoothing_factor=0.1,
            orpo_alpha=None,
        )
        trainer.state = SimpleNamespace()
        original_compute_loss = MagicMock(return_value=None)
        trainer.compute_loss = original_compute_loss
        patch_compute_loss_for_dft(trainer, cfg=MagicMock())

        with pytest.raises(ValueError, match="label smoothing"):
            trainer.compute_loss(model=MagicMock(), inputs={"labels": torch.zeros(1, 2)})
        assert original_compute_loss.call_count == 0
