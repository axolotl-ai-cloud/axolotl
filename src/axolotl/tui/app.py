"""Main TUI application for Axolotl."""

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Static

from axolotl.tui.screens.config import ConfigScreen
from axolotl.tui.screens.datasets import DatasetScreen
from axolotl.tui.screens.inference import InferenceScreen
from axolotl.tui.screens.models import ModelScreen
from axolotl.tui.screens.monitor import MonitorScreen
from axolotl.tui.screens.training import TrainingScreen


class WelcomeScreen(Screen):
    """Welcome screen with main menu."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "config", "Configuration"),
        Binding("t", "training", "Training"),
        Binding("d", "datasets", "Datasets"),
        Binding("m", "models", "Models"),
        Binding("i", "inference", "Inference"),
        Binding("s", "monitor", "System Monitor"),
    ]

    def compose(self) -> ComposeResult:
        """Compose the welcome screen."""
        yield Header()
        yield Container(
            Static("🦾 Axolotl TUI", classes="title"),
            Static(
                "A Terminal User Interface for fine-tuning LLMs", classes="subtitle"
            ),
            Container(
                Button("Configuration Management [C]", id="config", variant="primary"),
                Button("Training Management [T]", id="training", variant="primary"),
                Button("Dataset Management [D]", id="datasets", variant="primary"),
                Button("Model Management [M]", id="models", variant="primary"),
                Button("Inference & Testing [I]", id="inference", variant="primary"),
                Button("System Monitor [S]", id="monitor", variant="primary"),
                classes="menu-container",
            ),
            classes="welcome-container",
        )
        yield Footer()

    def action_quit(self) -> None:
        """Quit the application."""
        self.app.exit()

    def action_config(self) -> None:
        """Navigate to config screen."""
        self.app.push_screen(ConfigScreen())

    def action_training(self) -> None:
        """Navigate to training screen."""
        self.app.push_screen(TrainingScreen())

    def action_datasets(self) -> None:
        """Navigate to datasets screen."""
        self.app.push_screen(DatasetScreen())

    def action_models(self) -> None:
        """Navigate to models screen."""
        self.app.push_screen(ModelScreen())

    def action_inference(self) -> None:
        """Navigate to inference screen."""
        self.app.push_screen(InferenceScreen())

    def action_monitor(self) -> None:
        """Navigate to monitor screen."""
        self.app.push_screen(MonitorScreen())

    @on(Button.Pressed, "#config")
    def on_config_pressed(self) -> None:
        """Handle config button press."""
        self.action_config()

    @on(Button.Pressed, "#training")
    def on_training_pressed(self) -> None:
        """Handle training button press."""
        self.action_training()

    @on(Button.Pressed, "#datasets")
    def on_datasets_pressed(self) -> None:
        """Handle datasets button press."""
        self.action_datasets()

    @on(Button.Pressed, "#models")
    def on_models_pressed(self) -> None:
        """Handle models button press."""
        self.action_models()

    @on(Button.Pressed, "#inference")
    def on_inference_pressed(self) -> None:
        """Handle inference button press."""
        self.action_inference()

    @on(Button.Pressed, "#monitor")
    def on_monitor_pressed(self) -> None:
        """Handle monitor button press."""
        self.action_monitor()


class AxolotlTUI(App):
    """Main Axolotl TUI Application."""

    CSS = """
    .title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    .subtitle {
        text-align: center;
        padding: 1;
        color: $text-muted;
    }

    .welcome-container {
        align: center middle;
        height: 100%;
        width: 100%;
    }

    .menu-container {
        layout: vertical;
        align: center middle;
        padding: 2;
        width: auto;
        height: auto;
    }

    .menu-container Button {
        width: 35;
        margin: 1;
    }

    WelcomeScreen {
        align: center middle;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("escape", "back", "Back", priority=True),
    ]

    def on_mount(self) -> None:
        """Called when the app is mounted."""
        self.title = "Axolotl TUI"
        self.sub_title = "Fine-tuning LLMs made easy"
        self.push_screen(WelcomeScreen())

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_back(self) -> None:
        """Go back to previous screen."""
        if len(self.screen_stack) > 1:
            self.pop_screen()


def run():
    """Run the Axolotl TUI application."""
    app = AxolotlTUI()
    app.run()


if __name__ == "__main__":
    run()
