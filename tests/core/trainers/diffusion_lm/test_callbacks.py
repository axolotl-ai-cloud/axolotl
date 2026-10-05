"""Tests for core diffusion generation callbacks."""

from types import SimpleNamespace

import torch

from axolotl.core.trainers.diffusion_lm.callbacks import DiffusionGenerationCallback
from axolotl.core.trainers.diffusion_lm.collator import DiffusionCollator
from axolotl.utils.schemas.diffusion import DiffusionLMConfig


class CanonicalTrainer:
    """Minimal callback trainer with only canonical diffusion settings."""

    def __init__(self):
        self.axolotl_cfg = SimpleNamespace(
            diffusion_lm=SimpleNamespace(
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
        "axolotl.core.trainers.diffusion_lm.callbacks.generate_samples",
        fake_generate_samples,
    )

    DiffusionGenerationCallback(trainer).on_step_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(global_step=1),
        control=SimpleNamespace(),
    )

    assert captured["dataloader"] is trainer.train_loader
    assert captured["mask_token_id"] == 16


class _Tokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        return ":".join(map(str, ids))


def _native_callback_trainer(model, batch):
    trainer = CanonicalTrainer()
    trainer.axolotl_cfg.diffusion_lm.generation_max_length = 16
    trainer.model = model
    trainer.processing_class = _Tokenizer()
    trainer.train_loader = [batch] * 10
    return trainer


def _sft_features():
    return [
        {
            "input_ids": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "labels": [-100] * 8 + [10, 11, 12],
        }
    ]


def _run_callback(trainer, monkeypatch):
    logged = []
    callback = DiffusionGenerationCallback(trainer)
    monkeypatch.setattr(
        callback, "_log_samples", lambda samples, step: logged.extend(samples)
    )
    callback.on_step_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(global_step=1),
        control=SimpleNamespace(),
    )
    return logged


def test_native_callback_generates_from_encoder_canvas_collator_batch(monkeypatch):
    calls = []

    class Gemma(torch.nn.Module):
        config = SimpleNamespace(model_type="diffusion_gemma")

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def generate(self, **kwargs):
            calls.append(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[20, 21, 22]])], 1)

    batch = DiffusionCollator(0, 4)(_sft_features())
    assert "input_ids" not in batch
    samples = _run_callback(_native_callback_trainer(Gemma(), batch), monkeypatch)

    assert len(samples) == 1
    assert calls[0]["input_ids"].tolist() == [[2, 3, 4, 5, 6, 7, 8, 9]]
    assert calls[0]["max_new_tokens"] == 3
    assert samples[0]["generated_ids"] == [2, 3, 4, 5, 6, 7, 8, 9, 20, 21, 22]


def test_native_callback_generates_from_full_sequence_collator_batch(monkeypatch):
    calls = []

    class Nemotron(torch.nn.Module):
        config = SimpleNamespace(
            model_type="nemotron_labs_diffusion", block_size=1, eos_token_id=1
        )

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def generate_with_denoising_steps(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            tail = torch.full((1, kwargs["max_new_tokens"]), 30)
            return torch.cat([prompt, tail], 1), None

    batch = DiffusionCollator(0, layout="full_sequence")(_sft_features())
    samples = _run_callback(_native_callback_trainer(Nemotron(), batch), monkeypatch)

    assert len(samples) == 1
    assert calls[0][0].tolist() == [[2, 3, 4, 5, 6, 7, 8, 9]]
    assert calls[0][1]["max_new_tokens"] == 3
    assert samples[0]["generated_ids"] == [2, 3, 4, 5, 6, 7, 8, 9, 30, 30, 30]


def test_generate_samples_defaults_off_for_native_and_on_for_causal_lm():
    assert DiffusionLMConfig().generate_samples is False
    assert DiffusionLMConfig(from_causal_lm=True).generate_samples is True
    assert DiffusionLMConfig(generate_samples=True).generate_samples is True
    assert (
        DiffusionLMConfig(from_causal_lm=True, generate_samples=False).generate_samples
        is False
    )
