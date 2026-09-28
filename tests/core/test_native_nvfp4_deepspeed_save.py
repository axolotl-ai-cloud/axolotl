"""Scoped native-NVFP4 DeepSpeed checkpoint saving."""

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer

from axolotl.core.trainers.base import AxolotlTrainer


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("raises", [False, True])
@pytest.mark.parametrize("instance_override", [False, True])
def test_native_save_excludes_frozen_and_restores_method(
    monkeypatch, tmp_path, marked, raises, instance_override
):
    calls = []

    class Engine:
        def save_checkpoint(self, path, exclude_frozen_parameters=False):
            calls.append(exclude_frozen_parameters)
            if raises:
                raise RuntimeError("save failed")

    engine = Engine()
    if instance_override:
        engine.save_checkpoint = engine.save_checkpoint
    before = engine.__dict__.copy()
    model = torch.nn.Linear(2, 2)
    model.weight.requires_grad_(False)
    model._axolotl_native_nvfp4_deepspeed_prepared = marked
    trainer = object.__new__(AxolotlTrainer)
    trainer.state = SimpleNamespace(global_step=1)
    trainer.args = SimpleNamespace(include_tkps=False)
    trainer.is_deepspeed_enabled = True
    trainer.model_wrapped = engine
    trainer._get_output_dir = lambda trial: str(tmp_path)
    trainer._save_fsdp2_quantized_lora_adapter = lambda *args: False
    trainer._is_fsdp2_quantized_param = lambda p: p is model.weight

    def save(self, model, trial, **kwargs):
        self.model_wrapped.save_checkpoint(str(tmp_path))
        return "saved"

    monkeypatch.setattr(Trainer, "_save_checkpoint", save)
    if raises:
        with pytest.raises(RuntimeError, match="save failed"):
            trainer._save_checkpoint(model, None)
    else:
        assert trainer._save_checkpoint(model, None) == "saved"
    assert calls == [marked]
    assert engine.__dict__ == before
