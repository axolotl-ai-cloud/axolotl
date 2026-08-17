"""F4: FSDP2 + quantized-base LoRA checkpoints must stay resumable.

The DCP sharded model save fails on the NVFP4/Float8 frozen base, so the trainer saves just the
adapter — but it must STILL persist optimizer/scheduler/scaler/RNG and trainer_state so periodic
checkpoints can resume. These bind ``AxolotlTrainer._save_checkpoint`` to a stub (no Trainer/GPU)
and assert the resume artifacts are written after the adapter save.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import peft

from axolotl.core.trainers.base import AxolotlTrainer


def _stub(tmp_path, *, save_only_model=False, should_save=True, adapter_handled=True):
    return SimpleNamespace(
        state=SimpleNamespace(global_step=5, save_to_json=MagicMock()),
        args=SimpleNamespace(save_only_model=save_only_model, should_save=should_save),
        _get_output_dir=lambda trial=None: str(tmp_path),
        _save_fsdp2_quantized_lora_adapter=MagicMock(return_value=adapter_handled),
        _save_optimizer_and_scheduler=MagicMock(),
        _save_scaler=MagicMock(),
        _save_rng_state=MagicMock(),
    )


def test_quantized_lora_checkpoint_persists_resume_state(tmp_path):
    stub = _stub(tmp_path)
    out = AxolotlTrainer._save_checkpoint(stub, model=object(), trial=None)
    assert out is None
    stub._save_fsdp2_quantized_lora_adapter.assert_called_once()
    # the F4 fix: optimizer/scheduler/scaler/RNG + trainer_state all written (resumable)
    stub._save_optimizer_and_scheduler.assert_called_once()
    stub._save_scaler.assert_called_once()
    stub._save_rng_state.assert_called_once()
    stub.state.save_to_json.assert_called_once()


def test_save_only_model_skips_optimizer_but_writes_trainer_state(tmp_path):
    stub = _stub(tmp_path, save_only_model=True)
    AxolotlTrainer._save_checkpoint(stub, model=object(), trial=None)
    stub._save_optimizer_and_scheduler.assert_not_called()
    stub._save_rng_state.assert_not_called()
    stub.state.save_to_json.assert_called_once()  # trainer_state still written for resume bookkeeping


def test_resume_state_failure_keeps_adapter_and_does_not_raise(tmp_path):
    # If an FSDP2 resume-artifact save raises, keep the (already-written) adapter rather than aborting.
    stub = _stub(tmp_path)
    stub._save_optimizer_and_scheduler = MagicMock(
        side_effect=RuntimeError("dcp optim fail")
    )
    out = AxolotlTrainer._save_checkpoint(stub, model=object(), trial=None)
    assert out is None  # did not raise
    stub._save_fsdp2_quantized_lora_adapter.assert_called_once()


def test_fsdp2_checkpoint_save_uses_axolotl_cfg_when_trainer_flag_unset():
    stub = SimpleNamespace(
        is_fsdp_enabled=False,
        axolotl_cfg=SimpleNamespace(
            fsdp_version=2,
            fsdp_config={"state_dict_type": "SHARDED_STATE_DICT"},
            fsdp=None,
        ),
    )
    assert AxolotlTrainer._is_fsdp2_checkpoint_save_enabled(stub)


def test_fsdp2_checkpoint_save_ignores_non_fsdp2_cfg():
    stub = SimpleNamespace(
        is_fsdp_enabled=False,
        axolotl_cfg=SimpleNamespace(fsdp_version=1, fsdp_config={}, fsdp=None),
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
    # _save_fsdp2_quantized_lora_adapter gates on these helpers; bind the real
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

    handled = AxolotlTrainer._save_fsdp2_quantized_lora_adapter(
        stub, model, str(tmp_path)
    )

    assert handled is True
    resolve_ep_group.assert_called_once_with(stub.axolotl_cfg)
    save_ep_lora_adapter.assert_called_once()
    save_fsdp2_lora_adapter.assert_not_called()
