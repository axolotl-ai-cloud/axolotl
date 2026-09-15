"""Tests for SkipEvalOnResumeCallback."""

from unittest.mock import MagicMock

from transformers import TrainerControl, TrainerState, TrainingArguments

from axolotl.utils.callbacks import SkipEvalOnResumeCallback


class TestSkipEvalOnResumeCallback:
    """Tests for skipping redundant evaluation on checkpoint resume."""

    @staticmethod
    def _make_state(global_step: int) -> TrainerState:
        state = MagicMock(spec=TrainerState)
        state.global_step = global_step
        return state

    def test_suppresses_eval_at_resume_step(self):
        cb = SkipEvalOnResumeCallback()
        args = MagicMock(spec=TrainingArguments)
        state = self._make_state(20)
        control = TrainerControl(should_evaluate=False)

        # Simulate on_train_begin at checkpoint-20
        cb.on_train_begin(args, state, control)

        # Trainer sets should_evaluate = True for step 20
        control.should_evaluate = True
        result = cb.on_step_end(args, state, control)

        assert result.should_evaluate is False

    def test_allows_eval_after_resume_step(self):
        cb = SkipEvalOnResumeCallback()
        args = MagicMock(spec=TrainingArguments)
        state = self._make_state(20)
        control = TrainerControl(should_evaluate=False)

        cb.on_train_begin(args, state, control)

        # Advance past the resume point
        state.global_step = 30
        control.should_evaluate = True
        result = cb.on_step_end(args, state, control)

        assert result.should_evaluate is True

    def test_noop_on_fresh_run(self):
        cb = SkipEvalOnResumeCallback()
        args = MagicMock(spec=TrainingArguments)
        state = self._make_state(0)
        control = TrainerControl(should_evaluate=False)

        # Fresh run: global_step starts at 0
        cb.on_train_begin(args, state, control)

        # Even if eval triggers at step 0 (unlikely but defensive)
        state.global_step = 10
        control.should_evaluate = True
        result = cb.on_step_end(args, state, control)

        assert result.should_evaluate is True
