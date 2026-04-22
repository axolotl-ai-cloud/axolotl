"""Tests for the GCCallback"""

from unittest.mock import MagicMock, patch

from axolotl.utils.callbacks import GCCallback


class TestGCCallback:
    """Tests for GCCallback which handles Python gc.collect() during training."""

    def test_init_with_gc_collect_steps(self):
        cb = GCCallback(gc_collect_steps=10)
        assert cb.gc_collect_steps == 10

    def test_init_with_negative_gc_collect_steps(self):
        cb = GCCallback(gc_collect_steps=-1)
        assert cb.gc_collect_steps == -1

    def test_init_with_legacy_gc_steps(self):
        cb = GCCallback(gc_steps=5)
        assert cb.gc_collect_steps == 5

    def test_init_gc_collect_steps_overrides_gc_steps(self):
        cb = GCCallback(gc_collect_steps=10, gc_steps=5)
        assert cb.gc_collect_steps == 10

    def test_init_default(self):
        cb = GCCallback()
        assert cb.gc_collect_steps == -1

    @patch("axolotl.utils.callbacks.gc.collect")
    @patch("axolotl.utils.callbacks.torch.cuda.empty_cache")
    def test_gc_calls_collect_and_empty_cache(self, mock_empty_cache, mock_gc_collect):
        cb = GCCallback(gc_collect_steps=1)
        cb._gc()
        mock_gc_collect.assert_called_once()
        mock_empty_cache.assert_called_once()

    @patch("axolotl.utils.callbacks.gc.collect")
    @patch("axolotl.utils.callbacks.torch.cuda.empty_cache")
    def test_on_step_end_periodic_gc(self, mock_empty_cache, mock_gc_collect):
        cb = GCCallback(gc_collect_steps=5)
        args = MagicMock()
        args.save_strategy = "no"
        state = MagicMock()
        state.global_step = 10
        state.save_steps = 0
        state.max_steps = 100
        control = MagicMock()
        control.should_evaluate = False

        cb.on_step_end(args, state, control)

        # Step 10 % 5 == 0, so GC should have been called
        mock_gc_collect.assert_called_once()

    @patch("axolotl.utils.callbacks.gc.collect")
    @patch("axolotl.utils.callbacks.torch.cuda.empty_cache")
    def test_on_step_end_no_gc_when_not_on_interval(
        self, mock_empty_cache, mock_gc_collect
    ):
        cb = GCCallback(gc_collect_steps=5)
        args = MagicMock()
        args.save_strategy = "no"
        state = MagicMock()
        state.global_step = 7
        state.save_steps = 0
        state.max_steps = 100
        control = MagicMock()
        control.should_evaluate = False

        cb.on_step_end(args, state, control)

        # Step 7 % 5 != 0, so GC should not have been called
        mock_gc_collect.assert_not_called()

    @patch("axolotl.utils.callbacks.gc.collect")
    @patch("axolotl.utils.callbacks.torch.cuda.empty_cache")
    def test_on_step_end_gc_before_eval(self, mock_empty_cache, mock_gc_collect):
        cb = GCCallback(gc_collect_steps=-1)
        args = MagicMock()
        state = MagicMock()
        state.global_step = 3
        control = MagicMock()
        control.should_evaluate = True

        cb.on_step_end(args, state, control)

        mock_gc_collect.assert_called_once()
        assert cb.next_gc_on_begin_step == 4
