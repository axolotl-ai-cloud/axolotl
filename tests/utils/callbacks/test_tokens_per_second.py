"""Tests for trainable-token throughput accounting."""

import json
import os
from unittest.mock import MagicMock, patch

from transformers import TrainerState

from axolotl.core.builders import HFCausalTrainerBuilder
from axolotl.core.trainers.constants import TOKENS_STATE_FILE
from axolotl.core.trainers.utils import trainable_tokens_per_sec_per_gpu
from axolotl.utils.callbacks.tokens_per_second import TokensPerSecondCallback
from axolotl.utils.dict import DictDefault
from axolotl.utils.train import determine_last_checkpoint


def _cumulative_after_window(microbatch_token_counts, world_size, start=0.0):
    """Mimic the trainer's cumulative counter: every microbatch is SUM-reduced
    across ranks (balanced => x world_size) then added to the running total."""
    cum = start
    for tok in microbatch_token_counts:
        cum += tok * world_size
    return cum


class TestTrainableTokensPerSec:
    """Throughput from cumulative-counter deltas (regression for GA undercount)."""

    def test_basic_rate(self):
        # 800 trainable tokens over 10s on a single GPU
        assert trainable_tokens_per_sec_per_gpu(1000.0, 1800.0, 1, 10.0) == 80.0

    def test_world_size_divides_to_per_gpu(self):
        # 1600 tokens summed across 2 ranks over 10s => 80 tok/s/gpu
        assert trainable_tokens_per_sec_per_gpu(0.0, 1600.0, 2, 10.0) == 80.0

    def test_first_window_returns_none(self):
        assert trainable_tokens_per_sec_per_gpu(None, 1800.0, 1, 10.0) is None

    def test_resume_first_window_does_not_spike(self):
        # after resume the counter is restored to a large value but there is no
        # prior window yet, so the first post-resume log must not emit a rate
        assert trainable_tokens_per_sec_per_gpu(None, 5_000_000.0, 1, 10.0) is None

    def test_non_positive_elapsed_returns_none(self):
        assert trainable_tokens_per_sec_per_gpu(1000.0, 1800.0, 1, 0.0) is None
        assert trainable_tokens_per_sec_per_gpu(1000.0, 1800.0, 1, -5.0) is None

    def test_independent_of_gradient_accumulation(self):
        # same tokens and wall time, processed as 1 microbatch vs 8 microbatches
        ga1 = _cumulative_after_window([800], world_size=1)
        ga8 = _cumulative_after_window([100] * 8, world_size=1)
        rate1 = trainable_tokens_per_sec_per_gpu(0.0, ga1, 1, 10.0)
        rate8 = trainable_tokens_per_sec_per_gpu(0.0, ga8, 1, 10.0)
        assert rate1 == rate8 == 80.0

    def test_overhead_lowers_rate(self):
        # same tokens, larger wall time (e.g. eval/checkpoint in window) => lower rate
        fast = trainable_tokens_per_sec_per_gpu(0.0, 800.0, 1, 10.0)
        slow = trainable_tokens_per_sec_per_gpu(0.0, 800.0, 1, 20.0)
        assert slow < fast


def _write_checkpoint(output_dir, step, total, trainable):
    checkpoint = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint, exist_ok=True)
    with open(
        os.path.join(checkpoint, TOKENS_STATE_FILE), "w", encoding="utf-8"
    ) as fout:
        json.dump({"total": total, "trainable": trainable}, fout)
    return checkpoint


def _restored_tokens(callback):
    """Run on_train_begin and report the counters the trainer would see.

    The trainer gates on ``hasattr(state, "tokens")``, so an untouched state
    must not grow the attribute at all.
    """
    state = TrainerState()
    callback.on_train_begin(MagicMock(), state, MagicMock())
    return getattr(state, "tokens", None)


def _builder_tkps_callback(cfg):
    """Build the callback the way training actually does.

    Going through ``HFCausalTrainerBuilder.get_callbacks()`` is what pins the
    ordering this fix is about: the callback is constructed here, while
    ``cfg.resume_from_checkpoint`` is still unresolved.
    """
    builder = HFCausalTrainerBuilder.__new__(HFCausalTrainerBuilder)
    builder.cfg = cfg
    builder.model = MagicMock()
    with (
        patch("axolotl.core.builders.base.PluginManager") as pm,
        patch("axolotl.core.builders.base.TelemetryManager") as tm,
    ):
        pm.get_instance.return_value.add_callbacks_pre_trainer.return_value = []
        tm.get_instance.return_value.enabled = False
        callbacks = builder.get_callbacks()
    return next(c for c in callbacks if isinstance(c, TokensPerSecondCallback))


class TestTokensPerSecondResume:
    """Counter restore must survive auto-resume, which resolves the path late."""

    def test_restores_on_auto_resume(self, tmp_path):
        output_dir = str(tmp_path)
        _write_checkpoint(output_dir, 10, total=1234, trainable=567)
        cfg = DictDefault(
            {
                "output_dir": output_dir,
                "auto_resume_from_checkpoints": True,
                "resume_from_checkpoint": None,
                "include_tkps": True,
            }
        )

        # the builder runs first, while the path is still unresolved
        callback = _builder_tkps_callback(cfg)
        determine_last_checkpoint(cfg)

        tokens = _restored_tokens(callback)
        assert tokens is not None
        assert tokens["total"].item() == 1234
        assert tokens["trainable"].item() == 567

    def test_explicit_checkpoint_still_restores(self, tmp_path):
        output_dir = str(tmp_path)
        checkpoint = _write_checkpoint(output_dir, 5, total=42, trainable=7)
        cfg = DictDefault(
            {
                "output_dir": output_dir,
                "resume_from_checkpoint": checkpoint,
                "include_tkps": True,
            }
        )

        callback = _builder_tkps_callback(cfg)
        assert _restored_tokens(callback)["total"].item() == 42

    def test_fresh_run_leaves_counters_alone(self, tmp_path):
        cfg = DictDefault(
            {
                "output_dir": str(tmp_path),
                "auto_resume_from_checkpoints": True,
                "resume_from_checkpoint": None,
                "include_tkps": True,
            }
        )

        callback = _builder_tkps_callback(cfg)
        determine_last_checkpoint(cfg)

        assert _restored_tokens(callback) is None

    def test_no_cfg_falls_back_to_constructor_value(self, tmp_path):
        checkpoint = _write_checkpoint(str(tmp_path), 3, total=99, trainable=9)

        callback = TokensPerSecondCallback(resume_from_checkpoint=checkpoint)
        assert _restored_tokens(callback)["total"].item() == 99

    def test_unreadable_token_state_is_ignored(self, tmp_path):
        """A truncated state file must not take the whole run down."""
        checkpoint = os.path.join(str(tmp_path), "checkpoint-1")
        os.makedirs(checkpoint, exist_ok=True)
        with open(
            os.path.join(checkpoint, TOKENS_STATE_FILE), "w", encoding="utf-8"
        ) as fout:
            fout.write("{not json")

        callback = TokensPerSecondCallback(resume_from_checkpoint=checkpoint)
        assert _restored_tokens(callback) is None
