"""The trainer points users at the stability docs when the logged loss is NaN/inf."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
from transformers import Trainer

from axolotl.core.trainers import base
from axolotl.core.trainers.base import NAN_INF_DOCS_URL, AxolotlTrainer


def _trainer(nan_inf_filter=False):
    trainer = object.__new__(AxolotlTrainer)
    trainer.args = SimpleNamespace(
        include_tkps=False, logging_nan_inf_filter=nan_inf_filter
    )
    trainer._stored_metrics = defaultdict(
        lambda: defaultdict(lambda: {"values": [], "reduction": "mean"})
    )
    return trainer


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(Trainer, "log", lambda self, logs, start_time=None: None)
    monkeypatch.setattr(base.LOG, "warning_once", seen.append)
    return seen


@pytest.mark.parametrize("loss", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_loss_points_to_the_filter_and_docs(warnings_seen, loss):
    _trainer().log({"loss": loss})

    assert len(warnings_seen) == 1
    assert "logging_nan_inf_filter: true" in warnings_seen[0]
    assert NAN_INF_DOCS_URL in warnings_seen[0]


def test_filter_enabled_still_points_to_docs(warnings_seen):
    _trainer(nan_inf_filter=True).log({"loss": float("nan")})

    assert len(warnings_seen) == 1
    assert "even with logging_nan_inf_filter: true" in warnings_seen[0]
    assert NAN_INF_DOCS_URL in warnings_seen[0]


@pytest.mark.parametrize("logs", [{"loss": 1.25}, {"eval_loss": float("nan")}])
def test_finite_train_loss_and_eval_logs_do_not_warn(warnings_seen, logs):
    _trainer().log(logs)

    assert warnings_seen == []
