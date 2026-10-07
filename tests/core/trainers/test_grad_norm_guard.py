"""Tests for the gradient-norm outlier guard."""

from statistics import median
from types import SimpleNamespace

import pytest
import torch

from axolotl.core.trainers.mixins.grad_norm_guard import GradNormGuardMixin
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class _Base:
    """Stands in for the trainer chain below the mixin."""

    def _get_grad_norm(self, model, grad_norm=None):
        return grad_norm

    def training_step(self, model, inputs, num_items_in_batch=None):
        return torch.tensor(inputs)

    def log(self, logs, start_time=None):
        self.logged = dict(logs)


class _Trainer(GradNormGuardMixin, _Base):
    def __init__(self, zscore=10.0, window=20, action="scale"):
        self.args = SimpleNamespace(
            step_outlier_grad_norm_zscore=zscore,
            step_outlier_loss_zscore=None,
            step_outlier_window=window,
            step_outlier_action=action,
        )
        self.state = SimpleNamespace(global_step=0)


def _model():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3)
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    return model


def _history(n=20):
    # deterministic, spread-out norms: median 50, MAD 5
    return [45.0 + (i % 11) for i in range(n)]


def _feed(trainer, model, values):
    for value in values:
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        trainer._get_grad_norm(model, torch.tensor(value))
        trainer.state.global_step += 1


def test_disabled_is_a_no_op():
    trainer = _Trainer(zscore=None)
    model = _model()
    assert trainer._get_grad_norm(model, torch.tensor(1e6)).item() == 1e6
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    assert not hasattr(trainer, "_step_outlier_history")


def test_waits_for_a_full_window():
    trainer = _Trainer(window=20)
    model = _model()
    _feed(trainer, model, _history(19) + [1e6])
    assert trainer._step_outlier_count == 0
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())


def test_scale_multiplies_by_threshold_over_norm():
    trainer = _Trainer(window=20, zscore=10.0)
    model = _model()
    history = _history()
    _feed(trainer, model, history)
    center = median(history)
    spread = 1.4826 * median(abs(x - center) for x in history)
    threshold = center + 10.0 * spread
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    returned = trainer._get_grad_norm(model, torch.tensor(1000.0))
    assert returned.item() == 1000.0  # the logged norm stays raw
    for param in model.parameters():
        torch.testing.assert_close(
            param.grad, torch.full_like(param, threshold / 1000.0)
        )
    assert trainer._step_outlier_count == 1
    assert trainer._step_outlier_skipped == 0
    assert 1000.0 not in trainer._step_outlier_history["grad_norm"]


def test_values_at_or_below_threshold_pass_and_enter_history():
    trainer = _Trainer(window=20, zscore=10.0)
    model = _model()
    _feed(trainer, model, _history() + [60.0])
    assert trainer._step_outlier_count == 0
    assert trainer._step_outlier_history["grad_norm"][-1] == 60.0


def test_skip_leaves_weights_and_adam_state_unchanged():
    trainer = _Trainer(window=20, action="skip")
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    for value in _history():
        for param in model.parameters():
            param.grad = torch.randn_like(param)
        trainer._get_grad_norm(model, torch.tensor(value))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    weights = [p.detach().clone() for p in model.parameters()]
    states = [
        {k: v.clone() for k, v in optimizer.state[p].items()}
        for p in model.parameters()
    ]
    for param in model.parameters():
        param.grad = torch.randn_like(param) * 100
    trainer._get_grad_norm(model, torch.tensor(5000.0))
    optimizer.step()
    assert all(p.grad is None for p in model.parameters())
    for param, before, state in zip(model.parameters(), weights, states, strict=True):
        assert torch.equal(param, before)
        for key, value in state.items():
            assert torch.equal(optimizer.state[param][key], value)
    assert trainer._step_outlier_skipped == 1


def test_non_finite_norm_is_skipped_even_before_the_window_fills():
    trainer = _Trainer(window=20, action="scale")
    model = _model()
    trainer._get_grad_norm(model, torch.tensor(float("nan")))
    assert all(p.grad is None for p in model.parameters())
    assert trainer._step_outlier_skipped == 1
    assert len(trainer._step_outlier_history["grad_norm"]) == 0


def test_constant_history_never_triggers():
    trainer = _Trainer(window=20)
    model = _model()
    _feed(trainer, model, [50.0] * 20 + [1e6])
    assert trainer._step_outlier_count == 0


def test_log_reports_counts_on_train_logs_only():
    trainer = _Trainer(window=20)
    model = _model()
    _feed(trainer, model, _history() + [1000.0])
    trainer.log({"loss": 1.0})
    assert trainer.logged["step_outlier/steps"] == 1
    assert trainer.logged["step_outlier/skipped"] == 0
    trainer.log({"eval_loss": 1.0})
    assert "step_outlier/steps" not in trainer.logged


def test_trainer_mro_wraps_distributed_mixin():
    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.trainers.mixins import DistributedParallelMixin

    mro = AxolotlTrainer.__mro__
    assert mro.index(GradNormGuardMixin) < mro.index(DistributedParallelMixin)


def test_config_defaults_and_validation(min_base_cfg):
    cfg = validate_config(
        min_base_cfg
        | DictDefault(step_outlier_grad_norm_zscore=10, step_outlier_loss_zscore=6)
    )
    assert cfg.step_outlier_grad_norm_zscore == 10
    assert cfg.step_outlier_loss_zscore == 6
    assert cfg.step_outlier_window == 100
    assert cfg.step_outlier_action == "scale"
    with pytest.raises(ValueError, match="DeepSpeed"):
        validate_config(
            min_base_cfg
            | DictDefault(
                step_outlier_loss_zscore=6,
                deepspeed="deepspeed_configs/zero2.json",
            )
        )
    with pytest.raises(ValueError):
        validate_config(
            min_base_cfg
            | DictDefault(step_outlier_grad_norm_zscore=10, step_outlier_action="drop")
        )


class _RatioTrainer(GradNormGuardMixin, _Base):
    def __init__(self, ratio=2.0, beta=0.9):
        self.args = SimpleNamespace(
            step_outlier_grad_norm_zscore=None,
            step_outlier_loss_zscore=None,
            grad_clip_norm_ratio=ratio,
            grad_clip_norm_ratio_beta=beta,
        )
        self.state = SimpleNamespace(global_step=0)


def _two_tensor_model():
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(4))
    model.b = torch.nn.Parameter(torch.zeros(4))
    return model


def _set_grads(model, a_norm, b_norm):
    model.a.grad = torch.full((4,), a_norm / 2.0)  # norm of 4 equal entries x is 2x
    model.b.grad = torch.full((4,), b_norm / 2.0)


def test_ratio_first_step_initialises_without_clipping():
    trainer = _RatioTrainer(ratio=2.0)
    model = _two_tensor_model()
    _set_grads(model, 1.0, 3.0)
    trainer._get_grad_norm(model, torch.tensor(1.0))
    torch.testing.assert_close(model.a.grad.norm(), torch.tensor(1.0))
    torch.testing.assert_close(model.b.grad.norm(), torch.tensor(3.0))
    assert int(trainer._grad_clip_last_clipped) == 0


def test_ratio_clips_only_the_spiking_tensor_and_tracks_clipped_norm():
    trainer = _RatioTrainer(ratio=2.0, beta=0.9)
    model = _two_tensor_model()
    for _ in range(5):
        _set_grads(model, 1.0, 1.0)
        trainer._get_grad_norm(model, torch.tensor(1.0))
    _set_grads(model, 1.0, 10.0)
    trainer._get_grad_norm(model, torch.tensor(1.0))
    torch.testing.assert_close(model.a.grad.norm(), torch.tensor(1.0))
    torch.testing.assert_close(
        model.b.grad.norm(), torch.tensor(2.0), rtol=1e-4, atol=1e-4
    )
    # the average moves toward the clipped norm 2.0, not the raw 10.0
    torch.testing.assert_close(
        trainer._grad_clip_ema[1], torch.tensor(1.1), rtol=1e-4, atol=1e-4
    )
    trainer.log({"loss": 1.0})
    assert trainer.logged["grad_clip_ratio/clipped_tensors"] == 1
    assert trainer.logged["grad_clip_ratio/clipping_rate"] == 0.5


def test_ratio_beta_defaults_to_the_optimizers_larger_beta():
    trainer = _RatioTrainer(beta=None)
    model = _two_tensor_model()
    trainer.optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95))
    assert trainer._grad_clip_beta() == 0.95
    del trainer.optimizer
    assert trainer._grad_clip_beta() == 0.99


def test_ratio_does_not_run_on_a_skipped_step():
    trainer = _RatioTrainer()
    trainer.args.step_outlier_grad_norm_zscore = 10.0
    trainer.args.step_outlier_window = 20
    model = _two_tensor_model()
    _set_grads(model, 1.0, 1.0)
    trainer._get_grad_norm(model, torch.tensor(float("inf")))
    assert model.a.grad is None and model.b.grad is None
    assert getattr(trainer, "_grad_clip_ema", None) is None


def test_ratio_config_rejects_sharded_gradients(min_base_cfg):
    cfg = validate_config(min_base_cfg | DictDefault(grad_clip_norm_ratio=2.0))
    assert cfg.grad_clip_norm_ratio == 2.0
    with pytest.raises(ValueError, match="unsharded"):
        validate_config(
            min_base_cfg
            | DictDefault(
                grad_clip_norm_ratio=2.0,
                fsdp_version=2,
                fsdp_config={"reshard_after_forward": True},
            )
        )


def _loss_trainer(window=20, loss_z=6.0, action="scale"):
    trainer = _Trainer(zscore=None, window=window, action=action)
    trainer.args.step_outlier_loss_zscore = loss_z
    return trainer


def _feed_losses(trainer, model, losses, grad_norm=1.0, micro=2):
    for loss in losses:
        for param in model.parameters():
            param.grad = torch.ones_like(param)
        for _ in range(micro):
            trainer.training_step(model, loss)
        trainer._get_grad_norm(model, torch.tensor(grad_norm))
        trainer.state.global_step += 1


def test_loss_trigger_scales_by_loss_threshold():
    trainer = _loss_trainer()
    model = _model()
    losses = [1.0 + 0.01 * (i % 11) for i in range(20)]
    _feed_losses(trainer, model, losses)
    center = median(losses)
    spread = 1.4826 * median(abs(x - center) for x in losses)
    threshold = center + 6.0 * spread
    _feed_losses(trainer, model, [3.0])
    for param in model.parameters():
        torch.testing.assert_close(param.grad, torch.full_like(param, threshold / 3.0))
    assert trainer._step_outlier_count == 1
    assert 3.0 not in trainer._step_outlier_history["loss"]


def test_loss_is_averaged_over_micro_batches_and_reset_each_step():
    trainer = _loss_trainer()
    model = _model()
    for param in model.parameters():
        param.grad = torch.ones_like(param)
    trainer.training_step(model, 1.0)
    trainer.training_step(model, 3.0)
    trainer._get_grad_norm(model, torch.tensor(1.0))
    assert trainer._step_outlier_history["loss"][-1] == 2.0
    assert trainer._step_loss_sum is None and trainer._step_loss_micro == 0


def test_non_finite_loss_is_skipped():
    trainer = _loss_trainer()
    model = _model()
    _feed_losses(trainer, model, [float("nan")])
    assert all(p.grad is None for p in model.parameters())
    assert trainer._step_outlier_skipped == 1


def test_scale_uses_the_smallest_factor_when_both_signals_trigger():
    trainer = _loss_trainer(loss_z=6.0)
    trainer.args.step_outlier_grad_norm_zscore = 10.0
    model = _model()
    losses = [1.0 + 0.01 * (i % 11) for i in range(20)]
    norms = _history()
    for loss, norm in zip(losses, norms, strict=True):
        _feed_losses(trainer, model, [loss], grad_norm=norm)
    lc = median(losses)
    ls = 1.4826 * median(abs(x - lc) for x in losses)
    gc = median(norms)
    gs = 1.4826 * median(abs(x - gc) for x in norms)
    loss_factor = (lc + 6.0 * ls) / 3.0
    norm_factor = (gc + 10.0 * gs) / 5000.0
    _feed_losses(trainer, model, [3.0], grad_norm=5000.0)
    expected = min(loss_factor, norm_factor)
    for param in model.parameters():
        torch.testing.assert_close(param.grad, torch.full_like(param, expected))
    assert trainer._step_outlier_count == 1


def test_loss_not_recorded_when_loss_trigger_disabled():
    trainer = _Trainer(window=20)
    model = _model()
    trainer.training_step(model, 1.0)
    assert getattr(trainer, "_step_loss_sum", None) is None
