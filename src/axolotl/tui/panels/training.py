"""TrainingPanel — live scalar metrics table with loss sparkline."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState

# Braille sparkline characters (8 levels)
_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(values: list[float] | None, width: int = 20) -> str:
    if not values or len(values) < 2:
        return ""
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    rng = hi - lo if hi != lo else 1.0
    return "".join(_SPARK_CHARS[min(int((v - lo) / rng * 7), 7)] for v in vals)


# Known key ordering and formatting
_KNOWN_KEYS: list[tuple[str, str, str]] = [
    ("loss", "loss", ".4f"),
    ("grad_norm", "grad norm", ".3f"),
    ("learning_rate", "lr", ".2e"),
    ("tokens_per_second", "tok/s", ".1f"),
    ("samples_per_second", "samples/s", ".1f"),
    ("mfu", "MFU", ".1f"),
    # RL-specific
    ("rewards_mean", "rewards/mean", ".4f"),
    ("rewards_std", "rewards/std", ".4f"),
    ("kl_divergence", "KL", ".4f"),
    ("clip_ratio", "clip ratio", ".3f"),
    ("queue_size", "queue", "d"),
]


@register_panel(position="left", weight=10)
class TrainingPanel(BasePanel):
    name = "training"
    min_height = 8

    def render(self, state: TUIState) -> RenderableType:
        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=None,
            pad_edge=False,
        )
        table.add_column("metric", style="cyan", no_wrap=True)
        table.add_column("value", justify="right")
        table.add_column("trend", justify="left", no_wrap=True)

        for attr, label, fmt in _KNOWN_KEYS:
            val = getattr(state, attr, None)
            if val is None:
                # Also check extra dict
                val = state.extra.get(attr)
            if val is None:
                continue

            try:
                formatted = f"{val:{fmt}}"
            except (ValueError, TypeError):
                formatted = str(val)

            trend = ""
            if attr == "loss":
                trend = _sparkline(list(state.loss_history))

            table.add_row(label, formatted, trend)

        # Any extra keys not in _KNOWN_KEYS
        known_attrs = {k for k, _, _ in _KNOWN_KEYS}
        for key, val in sorted(state.extra.items()):
            if key in known_attrs or val is None:
                continue
            try:
                formatted = f"{val:.4f}"
            except (ValueError, TypeError):
                formatted = str(val)
            table.add_row(key, formatted, "")

        if table.row_count == 0:
            return Panel(
                Text("Waiting for first log step...", style="dim"),
                title="Training",
                border_style="blue",
            )

        return Panel(table, title="Training", border_style="blue")
