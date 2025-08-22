"""Base screen class for Axolotl TUI screens."""

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Footer, Header, Static


class BaseScreen(Screen):
    """Base class for all Axolotl TUI screens."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, title: str = "Axolotl", subtitle: str = ""):
        """Initialize the base screen.

        Args:
            title: The screen title
            subtitle: Optional subtitle for the screen
        """
        super().__init__()
        self.screen_title = title
        self.screen_subtitle = subtitle

    def compose(self) -> ComposeResult:
        """Compose the base screen layout."""
        yield Header()
        yield Container(
            Static(f"🦾 {self.screen_title}", classes="screen-title"),
            (
                Static(self.screen_subtitle, classes="screen-subtitle")
                if self.screen_subtitle
                else Static("")
            ),
            Container(id="content"),
            id="main-container",
        )
        yield Footer()

    def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()
