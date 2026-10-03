"""Tests for core diffusion generation callbacks."""

from types import SimpleNamespace

from axolotl.integrations.diffusion.lm.callbacks import DiffusionGenerationCallback


class CanonicalTrainer:
    """Minimal callback trainer with only canonical diffusion settings."""

    def __init__(self):
        self.axolotl_cfg = SimpleNamespace(
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
        self.model = object()
        self.processing_class = object()
        self.eval_dataset = None
        self.state = SimpleNamespace(is_world_process_zero=True)
        self.train_loader = object()

    def get_train_dataloader(self):
        return self.train_loader


def test_callback_uses_canonical_diffusion_config(monkeypatch):
    trainer = CanonicalTrainer()
    captured = {}

    def fake_generate_samples(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "axolotl.integrations.diffusion.lm.callbacks.generate_samples",
        fake_generate_samples,
    )

    DiffusionGenerationCallback(trainer).on_step_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(global_step=1),
        control=SimpleNamespace(),
    )

    assert captured["dataloader"] is trainer.train_loader
    assert captured["mask_token_id"] == 16
