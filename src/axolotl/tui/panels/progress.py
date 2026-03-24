"""ProgressPanel — top-bar progress display with step count, elapsed, ETA."""

from __future__ import annotations

from rich.console import RenderableType
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState


def _fmt_time(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "--:--:--"
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "eta --"
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    if h > 0:
        return f"eta {h}h{m:02d}m"
    return f"eta {m}m{int(seconds) % 60:02d}s"


@register_panel(position="top", weight=10)
class ProgressPanel(BasePanel):
    name = "progress"
    min_height = 3
    max_height = 3

    def render(self, state: TUIState) -> RenderableType:
        pct = (
            (state.current_step / state.total_steps * 100)
            if state.total_steps > 0
            else 0
        )

        # Header line
        mode_upper = state.training_mode.upper() if state.training_mode else "SFT"
        model_short = state.model_name.split("/")[-1] if state.model_name else "model"
        header = Text.assemble(
            ("● ", "bold green"),
            ("AXOLOTL", "bold cyan"),
            f"  {mode_upper} · {model_short}   ",
            (
                f"{state.current_step} / {state.total_steps}",
                "bold",
            ),
            f"  ·  {_fmt_time(state.elapsed_seconds)} elapsed  ·  {_fmt_eta(state.eta_seconds)}  ·  {pct:.1f}%",
        )

        # Progress bar
        progress = Progress(
            TextColumn(""),
            BarColumn(bar_width=None),
            TextColumn("{task.percentage:>3.0f}%"),
            expand=True,
        )
        task = progress.add_task("", total=state.total_steps or 1)
        progress.update(task, completed=state.current_step)

        table = Table.grid(expand=True)
        table.add_row(header)
        table.add_row(progress)
        return table
