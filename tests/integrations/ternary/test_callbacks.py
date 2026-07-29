"""CPU-only tests for the ternary training callbacks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import TrainerControl, TrainerState

from axolotl.integrations.ternary import callbacks as callbacks_mod
from axolotl.integrations.ternary.callbacks import (
    DistillAnchorCallback,
    LambdaScheduleCallback,
    TernaryMonitorCallback,
    WdAnnealCallback,
)

ARGS = SimpleNamespace()
CONTROL = TrainerControl()


def _state(global_step: int = 0, max_steps: int = 100) -> TrainerState:
    return TrainerState(global_step=global_step, max_steps=max_steps)


# reference packing/stats mirroring the `quant` contract, so these tests exercise
# only the callbacks' own scheduling and aggregation logic
def _pack(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(-1).to(torch.int16) + 1
    pad = -flat.numel() % 4
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    lanes = flat.reshape(-1, 4)
    packed = lanes[:, 0] | (lanes[:, 1] << 2) | (lanes[:, 2] << 4) | (lanes[:, 3] << 6)
    return packed.to(torch.uint8)


def _unpack(packed: torch.Tensor, numel: int) -> torch.Tensor:
    lanes = packed.to(torch.int16).reshape(-1, 1)
    codes = torch.cat([(lanes >> shift) & 3 for shift in (0, 2, 4, 6)], dim=1)
    return codes.reshape(-1)[:numel].to(torch.int8) - 1


def _ref_flip_count(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    numel = previous.numel() * 4
    diff = _unpack(previous, numel) != _unpack(current, numel)
    return diff.sum().to(torch.int64)


def _ref_zero_fraction(packed: torch.Tensor, numel: int) -> torch.Tensor:
    return (_unpack(packed, numel) == 0).sum().to(torch.float32) / numel


class _FakeTernary(nn.Module):
    """Stand-in for `TernaryLinear` exposing only what the callbacks touch."""

    def __init__(self, codes: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(codes.to(torch.float32))
        self.codes = codes
        self.lambda_ = 0.0

    def set_lambda(self, value: float) -> None:
        self.lambda_ = value

    def code_snapshot(self) -> torch.Tensor:
        return _pack(self.codes)


def _iter_fakes(model):
    for name, module in model.named_modules():
        if isinstance(module, _FakeTernary):
            yield name, module


@pytest.fixture(autouse=True)
def stub_ternary_deps(monkeypatch):
    monkeypatch.setattr(callbacks_mod, "iter_ternary_modules", _iter_fakes)
    monkeypatch.setattr(callbacks_mod, "iter_quantized_modules", _iter_fakes)
    monkeypatch.setattr(callbacks_mod, "flip_count", _ref_flip_count)
    monkeypatch.setattr(callbacks_mod, "zero_fraction", _ref_zero_fraction)
    monkeypatch.setattr(
        callbacks_mod, "module_family", lambda name: name.split(".")[-1]
    )


def _model(**specs: torch.Tensor) -> nn.Module:
    block = nn.Module()
    for name, codes in specs.items():
        block.add_module(name, _FakeTernary(codes))
    model = nn.Module()
    model.add_module("block", block)
    return model


def _codes(zeros: int, plus: int, minus: int) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros(zeros, dtype=torch.int8),
            torch.ones(plus, dtype=torch.int8),
            -torch.ones(minus, dtype=torch.int8),
        ]
    )


def test_lambda_at_linear_boundaries():
    assert LambdaScheduleCallback.lambda_at(0, 100, "linear") == 0.0
    assert LambdaScheduleCallback.lambda_at(25, 100, "linear") == 0.25
    assert LambdaScheduleCallback.lambda_at(100, 100, "linear") == 1.0
    assert LambdaScheduleCallback.lambda_at(1000, 100, "linear") == 1.0


def test_lambda_at_none_schedule_and_zero_warmup():
    assert LambdaScheduleCallback.lambda_at(0, 100, "none") == 1.0
    assert LambdaScheduleCallback.lambda_at(0, 0, "linear") == 1.0
    assert LambdaScheduleCallback.lambda_at(0, 0, "sigmoid") == 1.0


def test_lambda_at_sigmoid_is_a_normalized_monotone_ramp():
    assert LambdaScheduleCallback.lambda_at(0, 100, "sigmoid") == pytest.approx(0.0)
    assert LambdaScheduleCallback.lambda_at(100, 100, "sigmoid") == pytest.approx(1.0)
    assert LambdaScheduleCallback.lambda_at(50, 100, "sigmoid") == pytest.approx(0.5)
    values = [
        LambdaScheduleCallback.lambda_at(step, 100, "sigmoid")
        for step in range(0, 101, 10)
    ]
    assert values == sorted(values)
    # slower than linear in the first half, faster in the second
    assert values[2] < 0.2
    assert LambdaScheduleCallback.lambda_at(80, 100, "sigmoid") > 0.8


def test_lambda_callback_pushes_lambda_onto_modules():
    model = _model(q_proj=_codes(4, 2, 2), down_proj=_codes(2, 1, 1))
    callback = LambdaScheduleCallback(model, schedule="linear", warmup_steps=10)

    callback.on_train_begin(ARGS, _state(0), CONTROL)
    assert [module.lambda_ for _, module in _iter_fakes(model)] == [0.0, 0.0]

    callback.on_step_begin(ARGS, _state(5), CONTROL)
    assert [module.lambda_ for _, module in _iter_fakes(model)] == [0.5, 0.5]

    callback.on_step_begin(ARGS, _state(50), CONTROL)
    assert [module.lambda_ for _, module in _iter_fakes(model)] == [1.0, 1.0]


def test_lambda_callback_resolves_fractional_warmup_against_max_steps():
    model = _model(q_proj=_codes(4, 2, 2))
    callback = LambdaScheduleCallback(model, warmup_steps=0.1)

    callback.on_step_begin(ARGS, _state(10, max_steps=200), CONTROL)
    assert callback.current_lambda == pytest.approx(0.5)

    callback.on_step_begin(ARGS, _state(20, max_steps=200), CONTROL)
    assert callback.current_lambda == 1.0


def test_lambda_callback_reads_whole_float_warmup_as_the_whole_run():
    """`1.0` is the documented upper bound of the fractional form, not one step."""
    model = _model(q_proj=_codes(4, 2, 2))
    callback = LambdaScheduleCallback(model, warmup_steps=1.0)

    callback.on_step_begin(ARGS, _state(2500, max_steps=10000), CONTROL)
    assert callback.current_lambda == pytest.approx(0.25)

    callback.on_step_begin(ARGS, _state(10000, max_steps=10000), CONTROL)
    assert callback.current_lambda == 1.0


def test_lambda_callback_reads_one_int_step_as_one_step():
    model = _model(q_proj=_codes(4, 2, 2))
    callback = LambdaScheduleCallback(model, warmup_steps=1)

    callback.on_step_begin(ARGS, _state(1, max_steps=10000), CONTROL)
    assert callback.current_lambda == 1.0


def test_lambda_callback_never_resolves_a_fraction_to_zero_steps():
    """A fraction smaller than one step must not divide by zero in `lambda_at`."""
    model = _model(q_proj=_codes(4, 2, 2))
    callback = LambdaScheduleCallback(model, warmup_steps=0.001)

    callback.on_step_begin(ARGS, _state(0, max_steps=10), CONTROL)
    assert callback.current_lambda == 0.0

    callback.on_step_begin(ARGS, _state(1, max_steps=10), CONTROL)
    assert callback.current_lambda == 1.0


def test_lambda_callback_resumes_at_the_checkpoint_step():
    model = _model(q_proj=_codes(4, 2, 2))
    callback = LambdaScheduleCallback(model, warmup_steps=100)

    callback.on_train_begin(ARGS, _state(60), CONTROL)
    assert callback.current_lambda == pytest.approx(0.6)


def test_lambda_callback_is_a_noop_without_ternary_modules():
    model = nn.Sequential(nn.Linear(4, 4))
    callback = LambdaScheduleCallback(model, warmup_steps=10)

    callback.on_train_begin(ARGS, _state(0), CONTROL)
    callback.on_step_begin(ARGS, _state(5), CONTROL)

    assert callback.ternary_modules == []
    assert callback.current_lambda == -1.0


class _MultiplierSpy:
    """Records every multiplier the anchor callback pushes."""

    def __init__(self) -> None:
        self.pushed: list[float] = []

    def set_distill_multiplier(self, value: float) -> None:
        self.pushed.append(value)


def _multipliers(max_steps: int, anchor_start: float, steps=None) -> list[float]:
    steps = range(max_steps) if steps is None else steps
    return [
        DistillAnchorCallback.multiplier_at(step, max_steps, anchor_start)
        for step in steps
    ]


def test_anchor_multiplier_is_zero_before_the_anchor_and_one_after_the_ramp():
    # max_steps 100 -> a 2-step ramp opening at step 90
    assert _multipliers(100, 0.9, steps=[0, 45, 89]) == [0.0, 0.0, 0.0]
    assert DistillAnchorCallback.multiplier_at(90, 100, 0.9) == pytest.approx(0.5)
    assert _multipliers(100, 0.9, steps=[91, 95, 99]) == [1.0, 1.0, 1.0]


def test_the_kd_phase_is_the_configured_fraction_of_the_run():
    """The anchor is the first KD step, so the tail is exactly `1 - anchor_start` long."""
    for max_steps, anchor_start in ((1000, 0.9), (100, 0.9), (200, 0.5), (20, 0.9)):
        values = _multipliers(max_steps, anchor_start)
        assert sum(value > 0.0 for value in values) == round(
            (1.0 - anchor_start) * max_steps
        )


def test_anchor_ramp_is_monotone_and_never_a_single_step_jump():
    """A discrete jump in loss composition shocks the optimizer; λ jumps failed the same way."""
    values = _multipliers(1000, 0.9)
    assert values == sorted(values)
    partial = [value for value in values if 0.0 < value < 1.0]
    assert len(partial) == 19
    assert partial[0] == pytest.approx(0.05)
    assert values.count(0.0) == 900


def test_anchor_start_is_a_fraction_of_max_steps():
    assert _multipliers(200, 0.5, steps=[99, 100, 103]) == [0.0, 0.25, 1.0]
    assert _multipliers(200, 0.75, steps=[149, 150, 153]) == [0.0, 0.25, 1.0]


def test_anchor_ramp_still_completes_on_a_short_run_with_a_late_anchor():
    """`anchor_start` near 1.0 must not leave the KD phase with no full-weight step."""
    assert _multipliers(10, 0.95) == [0.0] * 9 + [1.0]
    assert _multipliers(4, 0.9) == [0.0, 0.0, 0.0, 1.0]
    assert _multipliers(2, 0.9) == [0.0, 1.0]
    # a single-step run is all tail: the schedule still delivers its KD phase
    assert _multipliers(1, 0.9) == [1.0]


def test_anchor_without_a_known_horizon_falls_back_to_the_constant_blend(caplog):
    assert DistillAnchorCallback.multiplier_at(0, 0, 0.9) == 1.0

    spy = _MultiplierSpy()
    callback = DistillAnchorCallback(trainer=spy)
    with caplog.at_level("WARNING"):
        callback.on_train_begin(ARGS, _state(0, max_steps=0), CONTROL)
        callback.on_step_begin(ARGS, _state(1, max_steps=0), CONTROL)

    assert spy.pushed == [1.0]
    assert sum("max_steps" in record.message for record in caplog.records) == 1


def test_anchor_callback_pushes_the_multiplier_onto_the_trainer():
    spy = _MultiplierSpy()
    callback = DistillAnchorCallback(trainer=spy, anchor_start=0.9)

    callback.on_train_begin(ARGS, _state(0), CONTROL)
    assert spy.pushed == [0.0]
    assert callback.multiplier == 0.0

    for step in (50, 90, 91, 92, 99):
        callback.on_step_begin(ARGS, _state(step), CONTROL)

    # only the changes are pushed, and the last one is full weight
    assert spy.pushed == [0.0, 0.5, 1.0]
    assert callback.multiplier == 1.0


def test_anchor_callback_resumes_at_the_checkpoint_step():
    spy = _MultiplierSpy()
    callback = DistillAnchorCallback(trainer=spy, anchor_start=0.9)

    callback.on_train_begin(ARGS, _state(95), CONTROL)

    assert spy.pushed == [1.0]


def test_anchor_callback_tolerates_a_trainer_without_the_setter():
    callback = DistillAnchorCallback(trainer=SimpleNamespace(), anchor_start=0.9)

    callback.on_step_begin(ARGS, _state(95), CONTROL)

    assert callback.multiplier == 1.0


def _two_group_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    latent = [module.weight for _, module in _iter_fakes(model)]
    latent_ids = {id(param) for param in latent}
    other = [param for param in model.parameters() if id(param) not in latent_ids]
    return torch.optim.AdamW(
        [
            {"params": latent, "weight_decay": 0.1},
            {"params": other, "weight_decay": 0.1},
        ]
    )


def _model_with_fp_linear() -> nn.Module:
    model = _model(q_proj=_codes(4, 2, 2))
    model.add_module("lm_head", nn.Linear(4, 4, bias=False))
    return model


def test_wd_anneal_takes_the_model_from_the_trainer_kwargs():
    model = _model_with_fp_linear()
    optimizer = _two_group_optimizer(model)
    callback = WdAnnealCallback(zero_at=0.5)

    callback.on_step_begin(
        ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer, model=model
    )
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.1]


def test_wd_anneal_falls_back_to_every_group_when_latents_are_hidden():
    model = _model_with_fp_linear()
    flat = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.AdamW([{"params": [flat], "weight_decay": 0.1}])
    callback = WdAnnealCallback(zero_at=0.5, model=model)

    callback.on_step_begin(ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer)
    assert optimizer.param_groups[0]["weight_decay"] == 0.0


def test_wd_anneal_zeroes_only_ternary_param_groups():
    model = _model_with_fp_linear()
    optimizer = _two_group_optimizer(model)
    callback = WdAnnealCallback(zero_at=0.5, model=model)

    callback.on_step_begin(ARGS, _state(4, max_steps=10), CONTROL, optimizer=optimizer)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.1]

    callback.on_step_begin(ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.1]


def test_wd_anneal_is_idempotent():
    model = _model_with_fp_linear()
    optimizer = _two_group_optimizer(model)
    callback = WdAnnealCallback(zero_at=0.5, model=model)

    callback.on_step_begin(ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer)
    assert callback.annealed is True

    optimizer.param_groups[0]["weight_decay"] = 0.1
    callback.on_step_begin(ARGS, _state(9, max_steps=10), CONTROL, optimizer=optimizer)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_wd_anneal_without_a_model_zeroes_every_group():
    model = _model_with_fp_linear()
    optimizer = _two_group_optimizer(model)
    callback = WdAnnealCallback(zero_at=0.5)

    callback.on_step_begin(ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer)
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.0]


def test_wd_anneal_skips_without_optimizer_or_max_steps():
    model = _model_with_fp_linear()
    optimizer = _two_group_optimizer(model)
    callback = WdAnnealCallback(zero_at=0.0, model=model)

    callback.on_step_begin(ARGS, _state(0, max_steps=10), CONTROL)
    assert callback.annealed is False

    callback.on_step_begin(ARGS, _state(0, max_steps=0), CONTROL, optimizer=optimizer)
    assert callback.annealed is False
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.1, 0.1]


def test_wd_anneal_is_a_noop_without_ternary_modules():
    model = nn.Sequential(nn.Linear(4, 4, bias=False))
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
    callback = WdAnnealCallback(zero_at=0.5, model=model)

    callback.on_step_begin(ARGS, _state(9, max_steps=10), CONTROL, optimizer=optimizer)
    assert optimizer.param_groups[0]["weight_decay"] == 0.1


def test_monitor_reports_zero_fraction_before_any_flip_data():
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=100)

    callback.on_step_end(ARGS, _state(0), CONTROL)

    assert callback.metrics == {
        "ternary/zero_frac": 0.5,
        "ternary/zero_frac/q_proj": 0.5,
    }


def test_monitor_does_not_log_a_nan_flip_rate_on_the_first_tick(caplog):
    """There is no previous snapshot yet, so there is no flip rate to report."""
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    with caplog.at_level("INFO"):
        callback.on_step_end(ARGS, _state(0), CONTROL)
        first = caplog.text
        callback.on_step_end(ARGS, _state(1), CONTROL)

    assert "nan" not in first.lower()
    assert "flip_rate" not in first
    # the second tick has a baseline, so the metric appears
    assert "flip_rate" in caplog.text.replace(first, "", 1)


def test_monitor_counts_flipped_codes_against_the_previous_snapshot():
    module_codes = _codes(8, 4, 4)
    model = _model(q_proj=module_codes)
    callback = TernaryMonitorCallback(model, every_n_steps=100)
    callback.on_step_end(ARGS, _state(100), CONTROL)

    flipped = module_codes.clone()
    flipped[0] = 1
    flipped[1] = -1
    flipped[8] = 0
    _, module = next(_iter_fakes(model))
    module.codes = flipped

    callback.on_step_end(ARGS, _state(200), CONTROL)

    assert callback.metrics["ternary/flip_rate"] == pytest.approx(3 / 16)
    assert callback.metrics["ternary/flip_rate/q_proj"] == pytest.approx(3 / 16)
    assert callback.metrics["ternary/zero_frac"] == pytest.approx(7 / 16)


def test_monitor_aggregates_per_family_and_overall():
    model = _model(q_proj=_codes(8, 4, 4), down_proj=_codes(0, 4, 4))
    model.block.add_module("q_proj_2", _FakeTernary(_codes(8, 0, 0)))
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    callback.on_step_end(ARGS, _state(1), CONTROL)

    assert callback.metrics["ternary/zero_frac/q_proj"] == pytest.approx(0.5)
    assert callback.metrics["ternary/zero_frac/down_proj"] == pytest.approx(0.0)
    assert callback.metrics["ternary/zero_frac/q_proj_2"] == pytest.approx(1.0)
    assert callback.metrics["ternary/zero_frac"] == pytest.approx(16 / 32)
    assert "ternary/flip_rate" not in callback.metrics


def test_monitor_only_fires_on_the_configured_period():
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=100)

    callback.on_step_end(ARGS, _state(50), CONTROL)
    assert callback.metrics == {}
    assert callback.snapshots == {}

    callback.on_step_end(ARGS, _state(100), CONTROL)
    assert callback.metrics


def test_monitor_disabled_when_period_is_zero():
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=0)

    callback.on_step_end(ARGS, _state(0), CONTROL)

    assert callback.metrics == {}


def test_monitor_snapshots_stay_on_cpu():
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    callback.on_step_end(ARGS, _state(1), CONTROL)

    snapshot = callback.snapshots["block.q_proj"]
    assert snapshot.device.type == "cpu"
    assert snapshot.dtype == torch.uint8
    assert snapshot.numel() == 4


def test_monitor_injects_metrics_into_logs_once():
    model = _model(q_proj=_codes(8, 4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=1)
    callback.on_step_end(ARGS, _state(1), CONTROL)

    logs = {"loss": 1.0}
    callback.on_log(ARGS, _state(1), CONTROL, logs=logs)
    assert logs["ternary/zero_frac"] == 0.5
    assert callback.metrics == {}

    logs = {"loss": 0.9}
    callback.on_log(ARGS, _state(2), CONTROL, logs=logs)
    assert logs == {"loss": 0.9}


def test_monitor_is_a_noop_without_ternary_modules():
    model = nn.Sequential(nn.Linear(4, 4))
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    callback.on_step_end(ARGS, _state(1), CONTROL)
    logs: dict[str, float] = {}
    callback.on_log(ARGS, _state(1), CONTROL, logs=logs)

    assert callback.ternary_modules == []
    assert callback.metrics == {}
    assert logs == {}


@pytest.fixture
def real_ternary_deps(monkeypatch):
    """Undo the stubs so the callbacks drive real `TernaryLinear` modules."""
    from axolotl.integrations.ternary.modules import (
        iter_quantized_modules,
        iter_ternary_modules,
    )
    from axolotl.integrations.ternary.quant import flip_count, zero_fraction
    from axolotl.integrations.ternary.swap import module_family

    for name, real in (
        ("iter_ternary_modules", iter_ternary_modules),
        ("iter_quantized_modules", iter_quantized_modules),
        ("flip_count", flip_count),
        ("zero_fraction", zero_fraction),
        ("module_family", module_family),
    ):
        monkeypatch.setattr(callbacks_mod, name, real)


def _real_model() -> nn.Module:
    """A `block.q_proj` ternary linear whose 16 weights are 8 zeros then 8 ones."""
    from axolotl.integrations.ternary.modules import TernaryLinear

    linear = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        linear.weight.copy_(_codes(8, 8, 0).to(torch.float32).reshape(4, 4))
    block = nn.Module()
    block.add_module("q_proj", TernaryLinear.from_linear(linear))
    model = nn.Module()
    model.add_module("block", block)
    return model


def test_lambda_callback_drives_real_ternary_modules(real_ternary_deps):
    model = _real_model()
    module = model.block.q_proj
    callback = LambdaScheduleCallback(model, warmup_steps=10)

    callback.on_train_begin(ARGS, _state(0), CONTROL)
    assert module.lambda_ == 0.0

    callback.on_step_begin(ARGS, _state(5), CONTROL)
    assert module.lambda_ == 0.5

    callback.on_step_begin(ARGS, _state(10), CONTROL)
    assert module.lambda_ == 1.0


def test_monitor_measures_real_code_flips(real_ternary_deps):
    model = _real_model()
    module = model.block.q_proj
    callback = TernaryMonitorCallback(model, every_n_steps=100)

    callback.on_step_end(ARGS, _state(100), CONTROL)
    assert callback.metrics["ternary/zero_frac"] == pytest.approx(0.5)
    assert callback.snapshots["block.q_proj"].numel() == 4

    with torch.no_grad():
        module.weight.view(-1)[:2] = 1.0
    callback.on_step_end(ARGS, _state(200), CONTROL)

    assert callback.metrics["ternary/flip_rate/q_proj"] == pytest.approx(2 / 16)
    assert callback.metrics["ternary/zero_frac"] == pytest.approx(6 / 16)


def test_monitor_pushes_metrics_through_the_trainer_metric_store(real_ternary_deps):
    """`on_log` fires after the trackers have read `logs`, so the store is the seam."""
    stored: list[dict] = []
    trainer = SimpleNamespace(
        store_metrics=lambda metrics, train_eval="train": stored.append(metrics)
    )
    model = _real_model()
    callback = TernaryMonitorCallback(model, every_n_steps=1, trainer=trainer)

    callback.on_step_end(ARGS, _state(1), CONTROL)

    assert stored and "ternary/zero_frac" in stored[0]
    assert callback.metrics == {}
    # the store already took them, so nothing is smuggled through the log dict
    logs: dict[str, float] = {"loss": 1.0}
    callback.on_log(ARGS, _state(1), CONTROL, logs=logs)
    assert logs == {"loss": 1.0}


def test_monitor_falls_back_to_the_log_dict_without_a_metric_store(real_ternary_deps):
    model = _real_model()
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    callback.on_step_end(ARGS, _state(1), CONTROL)
    logs: dict[str, float] = {"loss": 1.0}
    callback.on_log(ARGS, _state(1), CONTROL, logs=logs)

    assert "ternary/zero_frac" in logs


def test_monitor_counts_codes_per_module_not_per_global_weight(real_ternary_deps):
    """Under FSDP2 the snapshot is a shard, so the count has to come from the module."""
    model = _real_model()
    module = model.block.q_proj
    callback = TernaryMonitorCallback(model, every_n_steps=1)

    callback.on_step_end(ARGS, _state(1), CONTROL)

    assert module.code_count() == module.weight.numel()
    assert callback.metrics["ternary/zero_frac"] == pytest.approx(0.5)


def test_wd_anneal_zeroes_the_real_latent_group(real_ternary_deps):
    model = _real_model()
    model.add_module("lm_head", nn.Linear(4, 4, bias=False))
    optimizer = torch.optim.AdamW(
        [
            {"params": [model.block.q_proj.weight], "weight_decay": 0.1},
            {"params": [model.lm_head.weight], "weight_decay": 0.1},
        ]
    )
    callback = WdAnnealCallback(zero_at=0.5)

    callback.on_step_begin(
        ARGS, _state(5, max_steps=10), CONTROL, optimizer=optimizer, model=model
    )
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.0, 0.1]
