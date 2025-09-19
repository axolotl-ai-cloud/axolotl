"""Tests for diffusion generation callback dataloader selection and triggering."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from axolotl.integrations.diffusion import DiffusionGenerationCallback


class DummyTrainer:
    """Minimal trainer double with required attributes/methods for the callback."""

    def __init__(self, use_eval: bool):
        # Config used by callback
        self.cfg = SimpleNamespace(
            diffusion=SimpleNamespace(
                generation_interval=1,
                num_generation_samples=1,
                generation_max_length=32,
                generation_steps=4,
                generation_temperature=0.0,
                mask_token_id=16,
            ),
            use_wandb=False,
        )

        # Model/tokenizer are passed through to generate_samples; not used here
        self.model = Mock()
        self.processing_class = Mock()

        # Datasets and loaders
        self.eval_dataset = object() if use_eval else None
        self._train_loader = object()
        self._eval_loader = object()

        # State for world process check
        self.state = SimpleNamespace(is_world_process_zero=True)

        # Track which loader was requested
        self.requested: list[str] = []

    def get_train_dataloader(self):
        self.requested.append("train")
        return self._train_loader

    def get_eval_dataloader(self):
        self.requested.append("eval")
        return self._eval_loader


@pytest.mark.parametrize("use_eval", [False, True])
def test_callback_uses_correct_dataloader(monkeypatch, use_eval):
    trainer = DummyTrainer(use_eval=use_eval)
    callback = DiffusionGenerationCallback(trainer)

    captured = {}

    # Patch generate_samples in the callback module's namespace
    def fake_generate_samples(**kwargs):
        captured["dataloader"] = kwargs.get("dataloader")
        # Return one dummy sample to exercise logging path
        return [
            {
                "original": "o",
                "masked": "m",
                "generated": "g",
                "mask_ratio": 0.5,
                "masked_tokens": 1,
                "total_tokens": 2,
            }
        ]

    monkeypatch.setattr(
        "axolotl.integrations.diffusion.callbacks.generate_samples",
        fake_generate_samples,
    )

    # Trigger at step 1 (interval=1)
    args = SimpleNamespace()
    state = SimpleNamespace(global_step=1)
    control = SimpleNamespace()

    callback.on_step_end(args=args, state=state, control=control)

    # Assert the expected dataloader path was used
    if use_eval:
        assert trainer.requested[0] == "eval"
        assert captured["dataloader"] is trainer._eval_loader
    else:
        assert trainer.requested[0] == "train"
        assert captured["dataloader"] is trainer._train_loader
