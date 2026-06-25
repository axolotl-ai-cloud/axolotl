"""CompletionsPanel — shows recent RL/log_completions samples."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState


def _truncate(s: str, maxlen: int = 60) -> str:
    return s[:maxlen] + "…" if len(s) > maxlen else s


@register_panel(position="bottom", weight=20)
class CompletionsPanel(BasePanel):
    name = "completions"
    min_height = 6
    modes = ["grpo", "dpo"]

    def render(self, state: TUIState) -> RenderableType:
        if "*" not in self.modes and state.training_mode not in self.modes:
            return Text("")

        if not state.completions:
            return Panel(
                Text("No completions yet...", style="dim"),
                title="Completions",
                border_style="magenta",
            )

        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=None,
            pad_edge=False,
        )
        table.add_column("step", justify="right", width=6)
        table.add_column("prompt", no_wrap=False, max_width=40)
        table.add_column("completion", no_wrap=False, max_width=40)
        table.add_column("reward", justify="right", width=8)
        table.add_column("adv", justify="right", width=8)

        for sample in list(state.completions)[-5:]:
            reward_str = f"{sample.reward:.2f}" if sample.reward is not None else "--"
            adv_str = (
                f"{sample.advantage:+.2f}" if sample.advantage is not None else "--"
            )
            table.add_row(
                str(sample.step),
                _truncate(sample.prompt),
                _truncate(sample.completion),
                reward_str,
                adv_str,
            )

        return Panel(table, title="Completions", border_style="magenta")
