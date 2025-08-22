"""Training dialogs for Axolotl TUI."""

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select, Static


class NewTrainingDialog(ModalScreen):
    """Dialog for starting a new training job."""

    CSS = """
    NewTrainingDialog {
        align: center middle;
    }

    .dialog-container {
        background: $surface;
        border: thick $primary;
        padding: 2;
        width: 60;
        height: auto;
    }

    .dialog-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    .form-field {
        margin: 1 0;
    }

    .form-label {
        margin: 0 0 1 0;
        color: $text-muted;
    }

    .button-container {
        layout: horizontal;
        align: center middle;
        margin: 2 0 0 0;
    }

    .button-container Button {
        margin: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the dialog."""
        yield Container(
            Static("Start New Training Job", classes="dialog-title"),
            Container(
                Label("Configuration File:", classes="form-label"),
                Input(
                    placeholder="Path to config YAML file",
                    id="config-path",
                    value="/workspace/configs/",
                ),
                classes="form-field",
            ),
            Container(
                Label("Launcher:", classes="form-label"),
                Select(
                    [
                        ("accelerate", "Accelerate (Recommended)"),
                        ("torchrun", "TorchRun"),
                        ("deepspeed", "DeepSpeed"),
                    ],
                    id="launcher",
                    value="accelerate",
                ),
                classes="form-field",
            ),
            Container(
                Button("Start Training", variant="primary", id="start"),
                Button("Cancel", variant="default", id="cancel"),
                classes="button-container",
            ),
            classes="dialog-container",
        )

    @on(Button.Pressed, "#start")
    def handle_start(self) -> None:
        """Handle start button press."""
        config_input = self.query_one("#config-path", Input)
        launcher_select = self.query_one("#launcher", Select)

        config_path = config_input.value.strip()
        if not config_path:
            return

        if not Path(config_path).exists():
            return

        result = {
            "config_path": config_path,
            "launcher": launcher_select.value,
        }

        self.dismiss(result)

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)
