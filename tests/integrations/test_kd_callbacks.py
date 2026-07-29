"""
Tests for the KD temperature scheduler callback.
"""

from types import SimpleNamespace

from axolotl.integrations.kd.callbacks import KDTemperatureSchedulerCallback
from axolotl.integrations.kd.kernels.liger import LigerFusedLinearKLTopKLogprobLoss


def _trainer():
    return SimpleNamespace(
        data_collator=SimpleNamespace(kd_temperature=4.0),
        _kd_loss_fn=LigerFusedLinearKLTopKLogprobLoss(
            weight_hard_loss=0.1,
            weight_soft_loss=0.9,
            temperature=4.0,
        ),
    )


def test_scheduler_updates_both_collator_and_loss_fn():
    """Regression: the schedule was applied to the collator only, so the loss kept the
    temperature it was constructed with."""
    trainer = _trainer()
    callback = KDTemperatureSchedulerCallback(4.0, 1.0, trainer)

    callback.on_step_end(None, SimpleNamespace(global_step=5, max_steps=10), None)

    assert trainer.data_collator.kd_temperature == callback.temperature
    assert trainer._kd_loss_fn.temperature == callback.temperature
    assert 1.0 < callback.temperature < 4.0


def test_scheduler_reaches_minimum_temperature():
    trainer = _trainer()
    callback = KDTemperatureSchedulerCallback(4.0, 1.0, trainer)

    callback.on_step_end(None, SimpleNamespace(global_step=10, max_steps=10), None)

    assert callback.temperature == 1.0
    assert trainer._kd_loss_fn.temperature == 1.0
