"""SwanLab callbacks for Axolotl trainers.

This module provides HuggingFace Trainer callbacks for logging
RLHF completions to SwanLab.
"""

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.integrations.swanlab.completion_logger import CompletionLogger
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class SwanLabRLHFCompletionCallback(TrainerCallback):
    """Callback for logging RLHF completions to SwanLab.

    This callback periodically logs model completions (prompts, chosen/rejected
    responses, rewards) to SwanLab during RLHF training for qualitative analysis.

    Supports DPO, KTO, ORPO, and GRPO trainers.

    Example usage:
        >>> callback = SwanLabRLHFCompletionCallback(
        ...     log_interval=100,  # Log every 100 steps
        ...     max_completions=128,  # Keep last 128 completions
        ... )
        >>> trainer.add_callback(callback)

    Attributes:
        logger: CompletionLogger instance
        log_interval: Number of steps between SwanLab logging
        trainer_type: Auto-detected trainer type (dpo/kto/orpo/grpo)
    """

    def __init__(
        self,
        log_interval: int = 100,
        max_completions: int = 128,
        table_name: str = "rlhf_completions",
    ):
        """Initialize SwanLab RLHF completion callback.

        Args:
            log_interval: Log to SwanLab every N steps. Default: 100
            max_completions: Maximum completions to buffer. Default: 128
            table_name: SwanLab table name. Default: "rlhf_completions"
        """
        super().__init__()
        self.logger = CompletionLogger(maxlen=max_completions)
        self.log_interval = log_interval
        self.table_name = table_name
        self.trainer_type: str | None = None  # Auto-detected
        self._last_logged_step = 0

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Detect trainer type on initialization."""
        trainer = kwargs.get("trainer")
        if trainer is not None:
            trainer_name = trainer.__class__.__name__
            if "DPO" in trainer_name:
                self.trainer_type = "dpo"
            elif "KTO" in trainer_name:
                self.trainer_type = "kto"
            elif "ORPO" in trainer_name:
                self.trainer_type = "orpo"
            elif "GRPO" in trainer_name:
                self.trainer_type = "grpo"
            else:
                self.trainer_type = "unknown"

            LOG.info(
                f"SwanLab RLHF completion logging enabled for {trainer_name} "
                f"(type: {self.trainer_type})"
            )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ):
        """Capture completions from logs and buffer them.

        Different trainers log completions in different formats:
        - DPO: logs['dpo/chosen'], logs['dpo/rejected'], logs['dpo/reward_diff']
        - KTO: logs['kto/completion'], logs['kto/label'], logs['kto/reward']
        - ORPO: logs['orpo/chosen'], logs['orpo/rejected']
        - GRPO: logs['grpo/completion'], logs['grpo/reward']

        Note: This is a placeholder implementation. Actual log keys depend
        on the TRL trainer implementation. You may need to patch the trainers
        to expose completion data in logs.
        """
        if logs is None or self.trainer_type is None:
            return

        step = state.global_step

        # DPO completions
        if self.trainer_type == "dpo":
            if all(key in logs for key in ["dpo/prompt", "dpo/chosen", "dpo/rejected"]):
                self.logger.add_dpo_completion(
                    step=step,
                    prompt=logs.get("dpo/prompt", ""),
                    chosen=logs.get("dpo/chosen", ""),
                    rejected=logs.get("dpo/rejected", ""),
                    reward_diff=logs.get("dpo/reward_diff"),
                )

        # KTO completions
        elif self.trainer_type == "kto":
            if all(key in logs for key in ["kto/prompt", "kto/completion"]):
                self.logger.add_kto_completion(
                    step=step,
                    prompt=logs.get("kto/prompt", ""),
                    completion=logs.get("kto/completion", ""),
                    label=logs.get("kto/label", False),
                    reward=logs.get("kto/reward"),
                )

        # ORPO completions
        elif self.trainer_type == "orpo":
            if all(
                key in logs for key in ["orpo/prompt", "orpo/chosen", "orpo/rejected"]
            ):
                self.logger.add_orpo_completion(
                    step=step,
                    prompt=logs.get("orpo/prompt", ""),
                    chosen=logs.get("orpo/chosen", ""),
                    rejected=logs.get("orpo/rejected", ""),
                    log_odds_ratio=logs.get("orpo/log_odds_ratio"),
                )

        # GRPO completions
        elif self.trainer_type == "grpo":
            if all(key in logs for key in ["grpo/prompt", "grpo/completion"]):
                self.logger.add_grpo_completion(
                    step=step,
                    prompt=logs.get("grpo/prompt", ""),
                    completion=logs.get("grpo/completion", ""),
                    reward=logs.get("grpo/reward"),
                    advantage=logs.get("grpo/advantage"),
                )

        # Periodically log to SwanLab
        if step - self._last_logged_step >= self.log_interval:
            if len(self.logger) > 0:
                self.logger.log_to_swanlab(table_name=self.table_name)
                self.logger.clear()
                self._last_logged_step = step

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Log remaining completions at end of training."""
        if len(self.logger) > 0:
            LOG.info(
                f"Training complete, logging final {len(self.logger)} completions to SwanLab"
            )
            self.logger.log_to_swanlab(table_name=self.table_name)
            self._last_logged_step = state.global_step
