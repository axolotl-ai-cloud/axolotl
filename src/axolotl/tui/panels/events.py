"""EventsPanel — scrolling log of recent events, color-coded by level."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState

_LEVEL_STYLES = {
    "debug": "dim",
    "info": "",
    "warning": "yellow",
    "error": "red bold",
    "critical": "red bold",
}


@register_panel(position="bottom", weight=10)
class EventsPanel(BasePanel):
    name = "events"
    min_height = 8
    max_height = 20

    def render(self, state: TUIState) -> RenderableType:
        lines = Text()
        # Show last 15 non-debug log lines (debug goes to DebugPanel)
        recent = [l for l in state.log_lines if l.level != "debug"][-15:]
        for log_line in recent:
            ts = log_line.timestamp.strftime("%H:%M:%S")
            level = log_line.level.upper()
            style = _LEVEL_STYLES.get(log_line.level, "")
            lines.append(f"[{ts}] ", style="dim")
            lines.append(f"[{level}] ", style=style or "")
            lines.append(log_line.message[:200], style=style or "")
            lines.append("\n")

        if not recent:
            lines = Text("No events yet...", style="dim")

        return Panel(lines, title="Events", border_style="yellow")
