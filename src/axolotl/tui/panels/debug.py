"""DebugPanel — scrolling log of debug-level messages, separate from main events."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState


@register_panel(position="bottom", weight=30)
class DebugPanel(BasePanel):
    name = "debug"
    min_height = 6
    max_height = 10

    def render(self, state: TUIState) -> RenderableType:
        lines = Text()
        # Show last 8 debug-level log lines
        debug_lines = [
            log_entry for log_entry in state.log_lines if log_entry.level == "debug"
        ][-8:]
        for log_line in debug_lines:
            ts = log_line.timestamp.strftime("%H:%M:%S")
            lines.append(f"[{ts}] ", style="dim")
            lines.append(log_line.message[:200], style="dim")
            lines.append("\n")

        if not debug_lines:
            lines = Text("No debug messages yet...", style="dim")

        return Panel(lines, title="Debug", border_style="dim")
