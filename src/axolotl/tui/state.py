"""TUI shared data model — dataclasses for the dashboard state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class GPUStats:
    id: int
    name: str
    util_pct: float
    vram_used_gb: float
    vram_total_gb: float
    temp_c: int
    power_w: float | None


@dataclass
class LogLine:
    timestamp: datetime
    level: str  # "info" | "debug" | "warning" | "error"
    message: str


@dataclass
class CompletionSample:
    step: int
    prompt: str
    completion: str
    reward: float | None
    advantage: float | None


@dataclass
class TUIState:
    # Run metadata
    run_name: str = ""
    model_name: str = ""
    training_mode: str = "sft"
    world_size: int = 1
    start_time: datetime = field(default_factory=datetime.now)

    # Progress
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0.0
    total_epochs: int = 1
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None

    # Training metrics (rolling window + current)
    loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    tokens_per_second: float | None = None
    samples_per_second: float | None = None
    mfu: float | None = None

    # RL-specific (None for non-RL modes)
    rewards_mean: float | None = None
    rewards_std: float | None = None
    kl_divergence: float | None = None
    clip_ratio: float | None = None
    queue_size: int | None = None

    # Per-GPU hardware (list indexed by local rank)
    gpus: list[GPUStats] = field(default_factory=list)

    # Recent log lines
    log_lines: deque[LogLine] = field(default_factory=lambda: deque(maxlen=200))

    # Recent completions (GRPO/SFT with log_completions)
    completions: deque[CompletionSample] = field(
        default_factory=lambda: deque(maxlen=20)
    )

    # Loss history for sparkline
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=50))

    # Arbitrary plugin state
    extra: dict[str, Any] = field(default_factory=dict)
