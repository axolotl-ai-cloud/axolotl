"""
Single-metric early stopping tracker for the benchmark API plugin.
"""

from typing import Mapping, Optional, Tuple


class EarlyStopper:
    """
    Tracks one benchmark metric and decides when training should stop.

    Two independent stop conditions (either triggers a stop):
      - threshold: the metric reaches the target value.
      - patience: the metric fails to improve by at least ``min_delta`` for
        ``patience`` consecutive runs.
    """

    def __init__(
        self,
        metric: str,
        mode: str = "min",
        patience: int = 3,
        min_delta: float = 0.0,
        threshold: Optional[float] = None,
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.threshold = threshold

        self.best: Optional[float] = None
        self.num_bad_runs = 0

    def _is_better(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def _meets_threshold(self, value: float) -> bool:
        if self.threshold is None:
            return False
        if self.mode == "min":
            return value <= self.threshold
        return value >= self.threshold

    def update(self, metrics: Mapping[str, float]) -> Tuple[bool, str]:
        """
        Feed the latest benchmark metrics.

        Returns ``(should_stop, reason)``. ``reason`` is empty when not stopping.
        """
        if self.metric not in metrics:
            return False, ""

        value = metrics[self.metric]

        if self._meets_threshold(value):
            return True, (
                f"{self.metric}={value:g} reached threshold {self.threshold:g} "
                f"(mode={self.mode})"
            )

        # A run counts as progress only if it clears best by at least min_delta.
        # best moves only on a qualifying improvement (Keras-style, not a moving
        # reference), so a metric that improves in sub-min_delta steps still resets
        # once the cumulative gain clears the threshold.
        qualifies = self.best is None or (
            self._is_better(value) and abs(value - self.best) >= self.min_delta
        )
        if qualifies:
            self.num_bad_runs = 0
            self.best = value
        else:
            self.num_bad_runs += 1

        if self.num_bad_runs >= self.patience:
            return True, (
                f"{self.metric} did not improve by >= {self.min_delta:g} "
                f"for {self.patience} runs (best={self.best:g})"
            )

        return False, ""
