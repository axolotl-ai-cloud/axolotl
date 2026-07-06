"""Tests for the external benchmark API plugin."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from axolotl.integrations.benchmark_api import (
    BenchmarkAPICallback,
    _extract_scalar_metrics,
)
from axolotl.integrations.benchmark_api.args import BenchmarkAPIArgs
from axolotl.integrations.benchmark_api.early_stopping import EarlyStopper
from axolotl.utils.dict import DictDefault


def _make_cfg(**benchmark_api):
    """Validate through the real args model, then dump to a DictDefault (prod path)."""
    validated = BenchmarkAPIArgs(benchmark_api=benchmark_api)
    return DictDefault(validated.model_dump(exclude_none=True))


# --------------------------------------------------------------------------- #
# config schema
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "alias,expected",
    [
        ("lower", "min"),
        ("min", "min"),
        ("smaller", "min"),
        ("decrease", "min"),
        ("higher", "max"),
        ("max", "max"),
        ("larger", "max"),
        ("increase", "max"),
        ("LOWER", "min"),
    ],
)
def test_mode_alias_normalization(alias, expected):
    cfg = _make_cfg(
        endpoint="http://x",
        early_stopping={"enabled": True, "metric": "m", "mode": alias},
    )
    assert cfg.benchmark_api.early_stopping.mode == expected


def test_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        BenchmarkAPIArgs(
            benchmark_api={
                "endpoint": "http://x",
                "early_stopping": {"enabled": True, "metric": "m", "mode": "sideways"},
            }
        )


def test_metric_required_when_enabled():
    with pytest.raises(ValidationError):
        BenchmarkAPIArgs(
            benchmark_api={
                "endpoint": "http://x",
                "early_stopping": {"enabled": True},
            }
        )


def test_defaults():
    cfg = _make_cfg(endpoint="http://x")
    assert cfg.benchmark_api.run_on == ["save"]
    assert cfg.benchmark_api.timeout_sec == 3600
    assert cfg.benchmark_api.fail_training_on_error is False


def test_invalid_run_on_event_rejected():
    with pytest.raises(ValidationError):
        BenchmarkAPIArgs(benchmark_api={"endpoint": "http://x", "run_on": ["bogus"]})


# --------------------------------------------------------------------------- #
# scalar metric filtering
# --------------------------------------------------------------------------- #


def test_extract_scalar_metrics_filters_non_scalar():
    raw = {
        "a": 0.5,
        "b": 3,
        "c": "nan",
        "d": [1, 2],
        "e": {"nested": 1},
        "f": True,  # bool excluded
        "g": None,
    }
    assert _extract_scalar_metrics(raw) == {"a": 0.5, "b": 3}


def test_extract_scalar_metrics_non_dict():
    assert _extract_scalar_metrics(None) == {}
    assert _extract_scalar_metrics([1, 2]) == {}


# --------------------------------------------------------------------------- #
# early stopping
# --------------------------------------------------------------------------- #


def test_threshold_lower():
    stopper = EarlyStopper("cer", mode="min", threshold=0.075)
    assert stopper.update({"cer": 0.09}) == (False, "")
    stop, reason = stopper.update({"cer": 0.07})
    assert stop and "threshold" in reason


def test_threshold_higher():
    stopper = EarlyStopper("em", mode="max", threshold=0.40)
    assert stopper.update({"em": 0.30})[0] is False
    assert stopper.update({"em": 0.41})[0] is True


def test_patience_min_delta_lower():
    stopper = EarlyStopper("cer", mode="min", patience=2, min_delta=0.01)
    assert stopper.update({"cer": 0.50})[0] is False  # first, best set
    assert stopper.update({"cer": 0.495})[0] is False  # <0.01 gain -> bad 1
    assert stopper.update({"cer": 0.493})[0] is True  # bad 2 -> stop


def test_patience_resets_on_real_improvement():
    stopper = EarlyStopper("cer", mode="min", patience=2, min_delta=0.01)
    stopper.update({"cer": 0.50})
    stopper.update({"cer": 0.499})  # bad 1
    assert stopper.update({"cer": 0.40})[0] is False  # big gain resets counter
    assert stopper.num_bad_runs == 0
    assert stopper.best == 0.40


def test_missing_metric_is_noop():
    stopper = EarlyStopper("cer", mode="min", patience=1)
    assert stopper.update({"other": 1.0}) == (False, "")
    assert stopper.best is None


# --------------------------------------------------------------------------- #
# callback behavior
# --------------------------------------------------------------------------- #


def _callback(monkeypatch, response=None, raise_exc=None, **benchmark_api):
    cfg = _make_cfg(**benchmark_api)
    trainer = Mock()
    callback = BenchmarkAPICallback(cfg, trainer)

    posted = {}

    def fake_post(url, json=None, timeout=None):
        posted["url"] = url
        posted["json"] = json
        posted["timeout"] = timeout
        if raise_exc is not None:
            raise raise_exc
        resp = Mock()
        resp.raise_for_status = Mock()
        resp.json = Mock(return_value=response)
        return resp

    monkeypatch.setattr("axolotl.integrations.benchmark_api.requests.post", fake_post)
    return callback, trainer, posted


def _state(step=100, world_zero=True):
    return SimpleNamespace(global_step=step, is_world_process_zero=world_zero)


def _control():
    return SimpleNamespace(should_training_stop=False)


def _args(tmp_path):
    return SimpleNamespace(output_dir=str(tmp_path))


def test_callback_logs_scalar_metrics(monkeypatch, tmp_path):
    callback, trainer, posted = _callback(
        monkeypatch,
        response={
            "status": "completed",
            "metrics": {"eval/cer": 0.08, "eval/note": "x"},
        },
        endpoint="http://bench/eval",
        run_on=["save"],
    )
    control = callback.on_save(_args(tmp_path), _state(), _control())

    trainer.log.assert_called_once_with({"eval/cer": 0.08})
    assert posted["json"]["event"] == "save"
    assert posted["json"]["step"] == 100
    assert posted["json"]["output_dir"] == str(tmp_path)
    assert control.should_training_stop is False


def test_callback_resolves_checkpoint_dir(monkeypatch, tmp_path):
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    callback, _, posted = _callback(
        monkeypatch,
        response={"status": "completed", "metrics": {}},
        endpoint="http://bench/eval",
    )
    callback.on_save(_args(tmp_path), _state(step=100), _control())
    assert posted["json"]["checkpoint_dir"] == str(ckpt)


def test_callback_checkpoint_dir_falls_back_to_output_dir(monkeypatch, tmp_path):
    callback, _, posted = _callback(
        monkeypatch,
        response={"status": "completed", "metrics": {}},
        endpoint="http://bench/eval",
    )
    callback.on_save(_args(tmp_path), _state(step=999), _control())
    assert posted["json"]["checkpoint_dir"] == str(tmp_path)


def test_callback_skips_non_main_process(monkeypatch, tmp_path):
    callback, trainer, posted = _callback(
        monkeypatch,
        response={"status": "completed", "metrics": {"m": 1.0}},
        endpoint="http://bench/eval",
    )
    callback.on_save(_args(tmp_path), _state(world_zero=False), _control())
    trainer.log.assert_not_called()
    assert posted == {}  # no HTTP call made


def test_callback_respects_run_on(monkeypatch, tmp_path):
    callback, trainer, posted = _callback(
        monkeypatch,
        response={"status": "completed", "metrics": {"m": 1.0}},
        endpoint="http://bench/eval",
        run_on=["train_end"],
    )
    callback.on_save(_args(tmp_path), _state(), _control())  # not in run_on
    trainer.log.assert_not_called()
    assert posted == {}

    callback.on_train_end(_args(tmp_path), _state(), _control())
    trainer.log.assert_called_once()


def test_callback_skips_non_completed_status(monkeypatch, tmp_path):
    callback, trainer, _ = _callback(
        monkeypatch,
        response={"status": "queued", "metrics": {"m": 1.0}},
        endpoint="http://bench/eval",
    )
    callback.on_save(_args(tmp_path), _state(), _control())
    trainer.log.assert_not_called()


def test_callback_error_swallowed_by_default(monkeypatch, tmp_path):
    callback, trainer, _ = _callback(
        monkeypatch,
        raise_exc=RuntimeError("connection refused"),
        endpoint="http://bench/eval",
        fail_training_on_error=False,
    )
    # should not raise
    callback.on_save(_args(tmp_path), _state(), _control())
    trainer.log.assert_not_called()


def test_callback_error_raised_when_configured(monkeypatch, tmp_path):
    callback, _, _ = _callback(
        monkeypatch,
        raise_exc=RuntimeError("connection refused"),
        endpoint="http://bench/eval",
        fail_training_on_error=True,
    )
    with pytest.raises(RuntimeError):
        callback.on_save(_args(tmp_path), _state(), _control())


def test_callback_non_dict_response_swallowed_by_default(monkeypatch, tmp_path):
    callback, trainer, _ = _callback(
        monkeypatch,
        response=[1, 2, 3],  # runner returned a JSON array, not an object
        endpoint="http://bench/eval",
        fail_training_on_error=False,
    )
    # should not raise, and should log nothing
    callback.on_save(_args(tmp_path), _state(), _control())
    trainer.log.assert_not_called()


def test_callback_non_dict_response_raised_when_configured(monkeypatch, tmp_path):
    callback, _, _ = _callback(
        monkeypatch,
        response=None,  # runner returned JSON null
        endpoint="http://bench/eval",
        fail_training_on_error=True,
    )
    with pytest.raises(ValueError):
        callback.on_save(_args(tmp_path), _state(), _control())


def test_callback_early_stopping_sets_control(monkeypatch, tmp_path):
    callback, _, _ = _callback(
        monkeypatch,
        response={"status": "completed", "metrics": {"eval/cer": 0.05}},
        endpoint="http://bench/eval",
        early_stopping={
            "enabled": True,
            "metric": "eval/cer",
            "mode": "lower",
            "threshold": 0.075,
        },
    )
    control = callback.on_save(_args(tmp_path), _state(), _control())
    assert control.should_training_stop is True


def test_plugin_registers_callback(monkeypatch):
    from axolotl.integrations.benchmark_api import BenchmarkAPIPlugin

    plugin = BenchmarkAPIPlugin()
    cfg = _make_cfg(endpoint="http://bench/eval")
    callbacks = plugin.add_callbacks_post_trainer(cfg, Mock())
    assert len(callbacks) == 1
    assert isinstance(callbacks[0], BenchmarkAPICallback)


def test_plugin_no_callback_without_config():
    from axolotl.integrations.benchmark_api import BenchmarkAPIPlugin

    plugin = BenchmarkAPIPlugin()
    assert plugin.add_callbacks_post_trainer(DictDefault({}), Mock()) == []
