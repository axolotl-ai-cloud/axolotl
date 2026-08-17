"""Tests for trainable-token throughput accounting."""

from axolotl.core.trainers.utils import trainable_tokens_per_sec_per_gpu


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
