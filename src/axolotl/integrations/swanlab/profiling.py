"""SwanLab profiling utilities for Axolotl trainers.

This module provides decorators and context managers for profiling
trainer methods and logging execution times to SwanLab.
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@contextmanager
def swanlab_profiling_context(trainer: Any, func_name: str):
    """Context manager for profiling trainer methods.

    Measures execution time and logs to SwanLab if enabled.

    Example usage:
        >>> with swanlab_profiling_context(self, "training_step"):
        ...     result = do_expensive_computation()

    Args:
        trainer: Trainer instance (must have cfg attribute with use_swanlab flag)
        func_name: Name of the function being profiled

    Yields:
        None
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time

        # Check if SwanLab is enabled and initialized
        use_swanlab = getattr(getattr(trainer, "cfg", None), "use_swanlab", False)
        if use_swanlab:
            try:
                import swanlab

                if swanlab.get_run() is not None:
                    # Log profiling metric
                    trainer_class = trainer.__class__.__name__
                    metric_name = f"profiling/Time taken: {trainer_class}.{func_name}"

                    swanlab.log({metric_name: duration})

            except ImportError:
                # SwanLab not installed, silently skip
                pass
            except Exception as err:  # pylint: disable=broad-except
                # Log error but don't fail training
                LOG.debug(f"Failed to log profiling metric for {func_name}: {err}")


def swanlab_profile(func: Callable) -> Callable:
    """Decorator to profile and log function execution time to SwanLab.

    Automatically measures execution time of trainer methods and logs
    to SwanLab as profiling metrics.

    Example usage:
        >>> class MyTrainer:
        ...     @swanlab_profile
        ...     def training_step(self, model, inputs):
        ...         return super().training_step(model, inputs)

    Args:
        func: Function to profile (must be a method of a trainer instance)

    Returns:
        Wrapped function with profiling
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with swanlab_profiling_context(self, func.__name__):
            return func(self, *args, **kwargs)

    return wrapper


class ProfilingConfig:
    """Configuration for SwanLab profiling.

    This class provides a centralized way to control profiling behavior.

    Attributes:
        enabled: Whether profiling is enabled globally
        min_duration_ms: Minimum duration (in ms) to log (filters out very fast ops)
        log_interval: Log every N function calls (to reduce overhead)
    """

    def __init__(
        self,
        enabled: bool = True,
        min_duration_ms: float = 0.1,
        log_interval: int = 1,
    ):
        """Initialize profiling configuration.

        Args:
            enabled: Enable profiling. Default: True
            min_duration_ms: Minimum duration to log (ms). Default: 0.1
            log_interval: Log every N calls. Default: 1 (log all)
        """
        self.enabled = enabled
        self.min_duration_ms = min_duration_ms
        self.log_interval = log_interval
        self._call_counts: dict[str, int] = {}

    def should_log(self, func_name: str, duration_seconds: float) -> bool:
        """Check if a profiling measurement should be logged.

        Args:
            func_name: Name of the profiled function
            duration_seconds: Execution duration in seconds

        Returns:
            True if should log, False otherwise
        """
        if not self.enabled:
            return False

        # Check minimum duration threshold
        duration_ms = duration_seconds * 1000
        if duration_ms < self.min_duration_ms:
            return False

        # Check log interval
        self._call_counts.setdefault(func_name, 0)
        self._call_counts[func_name] += 1

        # Always log on first call OR at intervals
        count = self._call_counts[func_name]
        if count == 1 or count % self.log_interval == 0:
            return True

        return False


# Global profiling config (can be modified by users)
DEFAULT_PROFILING_CONFIG = ProfilingConfig()


@contextmanager
def swanlab_profiling_context_advanced(
    trainer: Any,
    func_name: str,
    config: ProfilingConfig | None = None,
):
    """Advanced profiling context with configurable behavior.

    Similar to swanlab_profiling_context but with additional configuration
    options for filtering and throttling profiling logs.

    Example usage:
        >>> config = ProfilingConfig(min_duration_ms=1.0, log_interval=10)
        >>> with swanlab_profiling_context_advanced(self, "forward", config):
        ...     output = model(inputs)

    Args:
        trainer: Trainer instance
        func_name: Function name
        config: Profiling configuration. If None, uses DEFAULT_PROFILING_CONFIG

    Yields:
        None
    """
    if config is None:
        config = DEFAULT_PROFILING_CONFIG

    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time

        # Check if should log based on config
        if config.should_log(func_name, duration):
            # Check if SwanLab is enabled
            use_swanlab = getattr(getattr(trainer, "cfg", None), "use_swanlab", False)
            if use_swanlab:
                try:
                    import swanlab

                    if swanlab.get_run() is not None:
                        trainer_class = trainer.__class__.__name__
                        metric_name = (
                            f"profiling/Time taken: {trainer_class}.{func_name}"
                        )

                        swanlab.log({metric_name: duration})

                except ImportError:
                    pass
                except Exception as err:  # pylint: disable=broad-except
                    LOG.debug(f"Failed to log profiling metric for {func_name}: {err}")
