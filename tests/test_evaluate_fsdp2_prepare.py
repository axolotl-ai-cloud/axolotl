"""Tests for preparing the model and optimizer together under FSDP2 before evaluation."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from axolotl.evaluate import prepare_fsdp2_model_for_eval


def _make_trainer(is_fsdp_enabled=True, fsdp_version=2, prepared_models=()):
    trainer = MagicMock()
    trainer.is_fsdp_enabled = is_fsdp_enabled
    trainer.model = object()
    trainer.model_wrapped = trainer.model
    trainer.optimizer = None

    optimizer = object()

    def create_optimizer():
        trainer.optimizer = optimizer

    trainer.create_optimizer.side_effect = create_optimizer

    accelerator = trainer.accelerator
    accelerator._models = list(prepared_models)
    plugin = SimpleNamespace(fsdp_version=fsdp_version) if fsdp_version else None
    accelerator.state = SimpleNamespace(fsdp_plugin=plugin)

    sharded_model = object()
    prepared_optimizer = object()
    accelerator.prepare.return_value = (sharded_model, prepared_optimizer)

    return trainer, optimizer, sharded_model, prepared_optimizer


def test_fsdp2_prepares_model_with_optimizer():
    trainer, optimizer, sharded_model, prepared_optimizer = _make_trainer()
    original_model = trainer.model

    prepare_fsdp2_model_for_eval(trainer)

    trainer.create_optimizer.assert_called_once_with()
    trainer.accelerator.prepare.assert_called_once_with(original_model, optimizer)
    assert trainer.model is sharded_model
    assert trainer.model_wrapped is sharded_model
    assert trainer.optimizer is prepared_optimizer


def test_fsdp2_skips_when_model_already_prepared():
    trainer, *_ = _make_trainer(prepared_models=(object(),))

    prepare_fsdp2_model_for_eval(trainer)

    trainer.create_optimizer.assert_not_called()
    trainer.accelerator.prepare.assert_not_called()


def test_fsdp1_leaves_eval_loop_to_prepare_model():
    trainer, *_ = _make_trainer(fsdp_version=1)

    prepare_fsdp2_model_for_eval(trainer)

    trainer.create_optimizer.assert_not_called()
    trainer.accelerator.prepare.assert_not_called()


def test_no_fsdp_is_untouched():
    trainer, *_ = _make_trainer(is_fsdp_enabled=False, fsdp_version=None)
    original_model = trainer.model

    prepare_fsdp2_model_for_eval(trainer)

    trainer.create_optimizer.assert_not_called()
    trainer.accelerator.prepare.assert_not_called()
    assert trainer.model is original_model


def test_callbacks_can_switch_on_their_forward_for_evaluation():
    from types import SimpleNamespace

    from axolotl.evaluate import enable_callbacks_for_evaluation

    class Gated:
        enabled = False

        def enable_for_evaluation(self):
            self.enabled = True

    gated = Gated()
    trainer = SimpleNamespace(
        callback_handler=SimpleNamespace(callbacks=[object(), gated])
    )
    assert enable_callbacks_for_evaluation(trainer) == ["Gated"]
    assert gated.enabled
    assert enable_callbacks_for_evaluation(SimpleNamespace()) == []


def test_merge_aware_schedule_enables_for_evaluation(monkeypatch):
    from axolotl.integrations.kernels import merge_aware_callback as mac
    from axolotl.integrations.kernels.libs import sonicmoe

    calls = []
    monkeypatch.setattr(sonicmoe, "set_merge_aware_enabled", calls.append)
    callback = mac.MergeAwareScheduleCallback(start_step=0.5)
    callback.enable_for_evaluation()
    callback.enable_for_evaluation()
    assert calls == [True]
