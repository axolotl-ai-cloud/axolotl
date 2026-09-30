"""F4: FSDP2 + quantized-base LoRA checkpoints must stay resumable.

The DCP sharded model save fails on the NVFP4/Float8 frozen base, so the trainer saves just the
adapter in place of the model save, and the rest of HF's checkpoint (optimizer/scheduler/RNG,
trainer state, rotation) still runs.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import peft
import pytest
import torch
from transformers import Trainer

from axolotl.core.trainers.base import AxolotlTrainer


@pytest.mark.parametrize("adapter_handled", [True, False])
def test_gathered_adapter_replaces_only_the_model_save(
    monkeypatch, tmp_path, adapter_handled
):
    model_saves = []
    monkeypatch.setattr(
        AxolotlTrainer, "save_model", lambda self, *a, **k: model_saves.append(a)
    )

    def hf_save_checkpoint(self, model, trial, **kwargs):
        self.save_model(str(tmp_path), _internal_call=True)
        return "rest of the checkpoint"

    monkeypatch.setattr(Trainer, "_save_checkpoint", hf_save_checkpoint)
    trainer = object.__new__(AxolotlTrainer)
    trainer.state = SimpleNamespace(global_step=5)
    trainer.args = SimpleNamespace(include_tkps=False)
    trainer.is_deepspeed_enabled = False
    trainer.model_wrapped = None
    trainer._get_output_dir = lambda trial=None: str(tmp_path)
    trainer._save_gathered_lora_adapter = MagicMock(return_value=adapter_handled)

    out = trainer._save_checkpoint(model=torch.nn.Linear(2, 2), trial=None)

    assert out == "rest of the checkpoint"
    trainer._save_gathered_lora_adapter.assert_called_once()
    assert len(model_saves) == (0 if adapter_handled else 1)
    assert "save_model" not in trainer.__dict__


def test_fsdp2_checkpoint_save_uses_axolotl_cfg_when_trainer_flag_unset():
    stub = SimpleNamespace(
        is_fsdp_enabled=False,
        axolotl_cfg=SimpleNamespace(
            fsdp_version=2,
            fsdp_config={"state_dict_type": "SHARDED_STATE_DICT"},
        ),
    )
    assert AxolotlTrainer._is_fsdp2_checkpoint_save_enabled(stub)


def test_fsdp2_checkpoint_save_ignores_cfg_without_fsdp_config():
    stub = SimpleNamespace(
        is_fsdp_enabled=False,
        axolotl_cfg=SimpleNamespace(fsdp_config=None),
    )
    assert not AxolotlTrainer._is_fsdp2_checkpoint_save_enabled(stub)


def test_fsdp2_quantized_param_detector_checks_dtensor_local_tensor():
    NVFP4Tensor = type("NVFP4Tensor", (), {})
    param = SimpleNamespace(_local_tensor=NVFP4Tensor())
    assert AxolotlTrainer._is_fsdp2_quantized_param(param)


def test_fsdp2_quantized_param_detector_checks_parameter_data():
    MXTensor = type("MXTensor", (), {})
    param = SimpleNamespace(data=MXTensor())
    assert AxolotlTrainer._is_fsdp2_quantized_param(param)


def test_quantized_lora_checkpoint_uses_ep_adapter_save(monkeypatch, tmp_path):
    class NVFP4Tensor:
        pass

    class FakePeftModel:
        def parameters(self):
            return [SimpleNamespace(_local_tensor=NVFP4Tensor())]

    model = FakePeftModel()
    stub = SimpleNamespace(
        is_fsdp_enabled=True,
        axolotl_cfg=SimpleNamespace(expert_parallel_size=2),
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
        _is_fsdp2_quantized_param=AxolotlTrainer._is_fsdp2_quantized_param,
    )
    # _save_gathered_lora_adapter gates on these helpers; bind the real
    # implementations so the test exercises actual enablement + quant detection.
    stub._is_fsdp2_checkpoint_save_enabled = lambda: (
        AxolotlTrainer._is_fsdp2_checkpoint_save_enabled(stub)
    )

    monkeypatch.setattr(peft, "PeftModel", FakePeftModel)

    from axolotl.integrations.expert_parallel import shard
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    resolve_ep_group = MagicMock(return_value=object())
    save_ep_lora_adapter = MagicMock(return_value=True)
    save_fsdp2_lora_adapter = MagicMock(return_value=True)
    monkeypatch.setattr(ExpertParallelPlugin, "_resolve_ep_group", resolve_ep_group)
    monkeypatch.setattr(shard, "save_ep_lora_adapter", save_ep_lora_adapter)
    monkeypatch.setattr(shard, "save_fsdp2_lora_adapter", save_fsdp2_lora_adapter)

    handled = AxolotlTrainer._save_gathered_lora_adapter(stub, model, str(tmp_path))

    assert handled is True
    resolve_ep_group.assert_called_once_with(stub.axolotl_cfg)
    save_ep_lora_adapter.assert_called_once()
    save_fsdp2_lora_adapter.assert_not_called()
