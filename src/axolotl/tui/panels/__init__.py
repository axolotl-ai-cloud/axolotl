"""Panel registry and base class for TUI panels."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rich.console import RenderableType

from axolotl.tui.state import TUIState

# ---------------------------------------------------------------------------
# Panel registry
# ---------------------------------------------------------------------------

_panel_registry: dict[str, type[BasePanel]] = {}


def register_panel(position: str = "bottom", weight: int = 50):
    """Decorator to register a panel class with position and weight."""

    def decorator(cls: type[BasePanel]) -> type[BasePanel]:
        cls.position = position
        cls.weight = weight
        _panel_registry[cls.name] = cls
        return cls

    return decorator


def get_registered_panels() -> dict[str, type[BasePanel]]:
    return dict(_panel_registry)


# ---------------------------------------------------------------------------
# BasePanel
# ---------------------------------------------------------------------------


class BasePanel(ABC):
    name: str = ""
    position: str = "bottom"
    weight: int = 50
    min_height: int = 4
    max_height: int | None = None
    modes: list[str] = ["*"]

    @abstractmethod
    def render(self, state: TUIState) -> RenderableType:
        """Return a rich renderable. Called every tick."""
        ...

    def on_event(self, event: dict) -> None:  # noqa: B027
        """Optional: react to raw metric events before state is merged."""
        pass


# Auto-import built-in panels to trigger registration
from axolotl.tui.panels.completions import CompletionsPanel  # noqa: E402, F401
from axolotl.tui.panels.debug import DebugPanel  # noqa: E402, F401
from axolotl.tui.panels.events import EventsPanel  # noqa: E402, F401
from axolotl.tui.panels.hardware import HardwarePanel  # noqa: E402, F401
from axolotl.tui.panels.progress import ProgressPanel  # noqa: E402, F401
from axolotl.tui.panels.training import TrainingPanel  # noqa: E402, F401
