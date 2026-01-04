"""SwanLab completion logger for RLHF/DPO/KTO/ORPO/GRPO training.

This module provides utilities for logging model completions during
preference training to SwanLab for qualitative analysis.
"""

from collections import deque
from collections.abc import Mapping
from typing import Any

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CompletionLogger:
    """Memory-bounded logger for RLHF completions.

    Stores prompts, completions, and rewards in fixed-size deques to prevent
    memory leaks during long training runs. Logs completion tables to SwanLab
    for qualitative analysis of model outputs.

    Example usage:
        >>> logger = CompletionLogger(maxlen=128)
        >>> logger.add_dpo_completion(
        ...     step=0,
        ...     prompt="What is AI?",
        ...     chosen="Artificial Intelligence is...",
        ...     rejected="AI means...",
        ...     reward_diff=0.5
        ... )
        >>> logger.log_to_swanlab()

    Attributes:
        maxlen: Maximum number of completions to store (older ones are dropped)
        data: Deque storing completion dictionaries
    """

    def __init__(self, maxlen: int = 128):
        """Initialize completion logger with bounded buffer.

        Args:
            maxlen: Maximum number of completions to store. When the buffer
                is full, oldest completions are automatically discarded.
                Default: 128 (sufficient for most RLHF runs without memory issues)
        """
        self.maxlen = maxlen
        self.data: deque[Mapping[str, Any]] = deque(maxlen=maxlen)

    def add_dpo_completion(
        self,
        step: int,
        prompt: str,
        chosen: str,
        rejected: str,
        reward_diff: float | None = None,
    ) -> None:
        """Add a DPO completion to the buffer.

        Args:
            step: Training step number
            prompt: Input prompt
            chosen: Chosen (preferred) completion
            rejected: Rejected (non-preferred) completion
            reward_diff: Reward difference (chosen - rejected), if available
        """
        entry = {
            "step": step,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        if reward_diff is not None:
            entry["reward_diff"] = reward_diff

        self.data.append(entry)

    def add_kto_completion(
        self,
        step: int,
        prompt: str,
        completion: str,
        label: bool,
        reward: float | None = None,
    ) -> None:
        """Add a KTO completion to the buffer.

        Args:
            step: Training step number
            prompt: Input prompt
            completion: Model-generated completion
            label: True if desirable, False if undesirable
            reward: Reward score, if available
        """
        entry = {
            "step": step,
            "prompt": prompt,
            "completion": completion,
            "label": "desirable" if label else "undesirable",
        }
        if reward is not None:
            entry["reward"] = reward

        self.data.append(entry)

    def add_orpo_completion(
        self,
        step: int,
        prompt: str,
        chosen: str,
        rejected: str,
        log_odds_ratio: float | None = None,
    ) -> None:
        """Add an ORPO completion to the buffer.

        Args:
            step: Training step number
            prompt: Input prompt
            chosen: Chosen (preferred) completion
            rejected: Rejected (non-preferred) completion
            log_odds_ratio: Log odds ratio between chosen and rejected
        """
        entry = {
            "step": step,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }
        if log_odds_ratio is not None:
            entry["log_odds_ratio"] = log_odds_ratio

        self.data.append(entry)

    def add_grpo_completion(
        self,
        step: int,
        prompt: str,
        completion: str,
        reward: float | None = None,
        advantage: float | None = None,
    ) -> None:
        """Add a GRPO completion to the buffer.

        Args:
            step: Training step number
            prompt: Input prompt
            completion: Model-generated completion
            reward: Reward score from reward model
            advantage: Advantage estimate (reward - baseline)
        """
        entry = {
            "step": step,
            "prompt": prompt,
            "completion": completion,
        }
        if reward is not None:
            entry["reward"] = reward
        if advantage is not None:
            entry["advantage"] = advantage

        self.data.append(entry)

    def log_to_swanlab(self, table_name: str = "completions") -> bool:
        """Log buffered completions to SwanLab as a table.

        Creates a SwanLab echarts Table with all buffered completions.
        Only logs if SwanLab is initialized and data is available.

        Args:
            table_name: Name of the table in SwanLab dashboard.
                Default: "completions"

        Returns:
            True if logging succeeded, False otherwise
        """
        if not self.data:
            LOG.debug("No completions to log to SwanLab")
            return False

        try:
            import swanlab

            if swanlab.get_run() is None:
                LOG.debug("SwanLab not initialized, skipping completion logging")
                return False

            # Convert deque to list of dicts
            completions = list(self.data)

            # Extract headers from first entry (all entries should have same structure)
            headers = list(completions[0].keys())

            # Build rows: each completion becomes one row
            rows = []
            for completion in completions:
                row = [completion.get(header, "") for header in headers]
                rows.append(row)

            # Log to SwanLab as echarts Table
            swanlab.log({table_name: swanlab.echarts.Table().add(headers, rows)})

            LOG.info(f"Logged {len(rows)} completions to SwanLab table '{table_name}'")
            return True

        except ImportError:
            LOG.warning(
                "SwanLab not installed, cannot log completions. "
                "Install with: pip install swanlab"
            )
            return False
        except Exception as err:  # pylint: disable=broad-except
            LOG.exception("Failed to log completions to SwanLab: %s", err)
            return False

    def clear(self) -> None:
        """Clear all buffered completions."""
        self.data.clear()

    def __len__(self) -> int:
        """Return number of buffered completions."""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation showing buffer status."""
        return (
            f"CompletionLogger(maxlen={self.maxlen}, "
            f"buffered={len(self.data)}/{self.maxlen})"
        )
