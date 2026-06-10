"""HardwarePanel — per-GPU stats via pynvml."""

from __future__ import annotations

from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from axolotl.tui.panels import BasePanel, register_panel
from axolotl.tui.state import TUIState

_BAR_FULL = "█"
_BAR_EMPTY = "░"


def _util_bar(pct: float, width: int = 6) -> Text:
    filled = int(pct / 100 * width)
    bar = _BAR_FULL * filled + _BAR_EMPTY * (width - filled)
    color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
    return Text.assemble((bar, color), f" {pct:3.0f}%")


@register_panel(position="right", weight=10)
class HardwarePanel(BasePanel):
    name = "hardware"
    min_height = 6

    def render(self, state: TUIState) -> RenderableType:
        if not state.gpus:
            return Panel(
                Text("GPU stats unavailable", style="dim"),
                title="Hardware",
                border_style="green",
            )

        table = Table(
            show_header=True,
            header_style="bold",
            expand=True,
            box=None,
            pad_edge=False,
        )
        table.add_column("id", justify="right", width=3)
        table.add_column("util", no_wrap=True)
        table.add_column("vram", no_wrap=True)
        table.add_column("°C", justify="right", width=4)
        table.add_column("W", justify="right", width=5)

        total_vram_used = 0.0
        total_vram_total = 0.0
        total_util = 0.0

        for gpu in state.gpus:
            total_vram_used += gpu.vram_used_gb
            total_vram_total += gpu.vram_total_gb
            total_util += gpu.util_pct

            power_str = f"{gpu.power_w:.0f}" if gpu.power_w is not None else "--"
            table.add_row(
                str(gpu.id),
                _util_bar(gpu.util_pct),
                f"{gpu.vram_used_gb:.1f}/{gpu.vram_total_gb:.1f} GB",
                str(gpu.temp_c),
                power_str,
            )

        # Footer with aggregates
        n = len(state.gpus)
        if n > 1:
            avg_util = total_util / n
            table.add_row(
                "Σ",
                Text(f"avg {avg_util:.0f}%", style="dim"),
                Text(f"{total_vram_used:.1f}/{total_vram_total:.1f} GB", style="dim"),
                "",
                "",
            )

        return Panel(table, title="Hardware", border_style="green")
