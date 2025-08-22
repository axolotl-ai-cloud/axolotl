"""Model management screen for Axolotl TUI."""

from pathlib import Path
from typing import Dict, List, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    Log,
    ProgressBar,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from axolotl.tui.screens.base import BaseScreen


class ModelScreen(BaseScreen):
    """Model management screen."""

    BINDINGS = [
        Binding("ctrl+m", "merge_lora", "Merge LoRA"),
        Binding("ctrl+q", "quantize", "Quantize"),
        Binding("ctrl+e", "evaluate", "Evaluate"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    .model-container {
        layout: horizontal;
        height: 100%;
    }

    .model-list {
        width: 50%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .model-operations {
        width: 50%;
        border: solid $secondary;
        padding: 1;
        margin: 1;
    }

    .model-actions {
        layout: horizontal;
        height: 4;
        align: center middle;
        padding: 1;
    }

    .model-actions Button {
        margin: 0 1;
    }

    DataTable {
        height: 80%;
    }

    .screen-title {
        text-align: center;
        text-style: bold;
        padding: 1;
        color: $primary;
    }

    .screen-subtitle {
        text-align: center;
        padding: 0 0 1 0;
        color: $text-muted;
    }
    """

    def __init__(self):
        """Initialize the model screen."""
        super().__init__(
            title="Model Management",
            subtitle="Manage trained models, merge LoRA adapters, and quantize models",
        )
        self.models: Dict[str, Dict] = {}
        self.selected_model: Optional[str] = None

    def compose(self) -> ComposeResult:
        """Compose the model screen layout."""
        yield Header()
        yield Container(
            Static("🦾 Model Management", classes="screen-title"),
            Static(
                "Manage trained models, merge LoRA adapters, and quantize models",
                classes="screen-subtitle",
            ),
            Container(
                Container(
                    Label("Available Models"),
                    DataTable(id="model-table"),
                    Container(
                        Button("Merge LoRA", id="merge-lora", variant="primary"),
                        Button("Quantize", id="quantize", variant="success"),
                        Button("Evaluate", id="evaluate", variant="warning"),
                        Button("Refresh", id="refresh", variant="default"),
                        classes="model-actions",
                    ),
                    classes="model-list",
                ),
                Container(
                    TabbedContent(
                        TabPane(
                            "Operations",
                            Container(
                                Log(id="operations-log", wrap=True, highlight=True),
                                Container(
                                    Label("Operation Progress:"),
                                    ProgressBar(
                                        total=100,
                                        id="operation-progress",
                                    ),
                                ),
                            ),
                        ),
                        TabPane(
                            "Model Info",
                            ScrollableContainer(
                                Static(
                                    "Model information will appear here",
                                    id="model-info",
                                ),
                            ),
                        ),
                    ),
                    classes="model-operations",
                ),
                classes="model-container",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.setup_model_table()
        self.load_models()

        log = self.query_one("#operations-log", Log)
        log.write_line("Model manager ready.")

    def setup_model_table(self) -> None:
        """Setup the model table."""
        table = self.query_one("#model-table", DataTable)
        table.add_columns("Name", "Type", "Size", "Status")
        table.cursor_type = "row"
        table.zebra_stripes = True

    @work(thread=True)
    async def load_models(self) -> None:
        """Load available models."""
        # Check outputs directory for trained models
        outputs_dir = Path("./outputs")
        if outputs_dir.exists():
            for model_dir in outputs_dir.glob("*"):
                if model_dir.is_dir():
                    self.models[model_dir.name] = {
                        "name": model_dir.name,
                        "path": str(model_dir),
                        "type": "checkpoint",
                        "size": self.get_dir_size(model_dir),
                        "status": "available",
                    }

        self.refresh_model_table()

    def get_dir_size(self, path: Path) -> str:
        """Get human-readable directory size."""
        try:
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

            for unit in ["B", "KB", "MB", "GB"]:
                if total_size < 1024.0:
                    return f"{total_size:.2f} {unit}"
                total_size /= 1024.0
            return f"{total_size:.2f} TB"
        except Exception:
            return "Unknown"

    def refresh_model_table(self) -> None:
        """Refresh the model table."""
        table = self.query_one("#model-table", DataTable)
        table.clear()

        for name, info in self.models.items():
            table.add_row(
                name[:30],
                info["type"],
                info["size"],
                info["status"],
            )

    @on(DataTable.RowSelected)
    def handle_model_selected(self, event: DataTable.RowSelected) -> None:
        """Handle model selection from table."""
        if event.row_index >= 0:
            model_names = list(self.models.keys())
            if event.row_index < len(model_names):
                self.selected_model = model_names[event.row_index]
                self.update_model_info()

    def update_model_info(self) -> None:
        """Update model information display."""
        if not self.selected_model:
            return

        info = self.models[self.selected_model]
        info_text = f"""
Model Name: {info['name']}
Path: {info['path']}
Type: {info['type']}
Size: {info['size']}
Status: {info['status']}
        """

        self.query_one("#model-info", Static).update(info_text)

    @on(Button.Pressed, "#merge-lora")
    @work(thread=True)
    async def handle_merge_lora(self) -> None:
        """Merge LoRA adapters with base model."""
        if not self.selected_model:
            log = self.query_one("#operations-log", Log)
            log.write_line("⚠️ No model selected")
            return

        model_info = self.models[self.selected_model]
        log = self.query_one("#operations-log", Log)
        log.clear()
        log.write_line(f"🔄 Merging LoRA adapters for {self.selected_model}...")

        progress = self.query_one("#operation-progress", ProgressBar)
        progress.update(progress=0)

        try:
            import subprocess

            cmd = ["python", "-m", "axolotl.cli.merge_lora", model_info["path"]]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                log.write_line(line.strip())
                progress.advance(10)

            process.wait()

            if process.returncode == 0:
                log.write_line("✅ LoRA merge completed successfully!")
                progress.update(progress=100)
            else:
                log.write_line(f"❌ LoRA merge failed with code {process.returncode}")

        except Exception as e:
            log.write_line(f"❌ Error during LoRA merge: {str(e)}")

    @on(Button.Pressed, "#quantize")
    @work(thread=True)
    async def handle_quantize(self) -> None:
        """Quantize selected model."""
        if not self.selected_model:
            log = self.query_one("#operations-log", Log)
            log.write_line("⚠️ No model selected")
            return

        model_info = self.models[self.selected_model]
        log = self.query_one("#operations-log", Log)
        log.clear()
        log.write_line(f"🔄 Quantizing {self.selected_model}...")

        progress = self.query_one("#operation-progress", ProgressBar)
        progress.update(progress=0)

        try:
            import subprocess

            cmd = [
                "python",
                "-m",
                "axolotl.cli.quantize",
                model_info["path"],
                "--output-dir",
                f"{model_info['path']}_quantized",
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                log.write_line(line.strip())
                progress.advance(5)

            process.wait()

            if process.returncode == 0:
                log.write_line("✅ Quantization completed successfully!")
                progress.update(progress=100)
            else:
                log.write_line(f"❌ Quantization failed with code {process.returncode}")

        except Exception as e:
            log.write_line(f"❌ Error during quantization: {str(e)}")

    @on(Button.Pressed, "#evaluate")
    @work(thread=True)
    async def handle_evaluate(self) -> None:
        """Evaluate selected model."""
        if not self.selected_model:
            log = self.query_one("#operations-log", Log)
            log.write_line("⚠️ No model selected")
            return

        model_info = self.models[self.selected_model]
        log = self.query_one("#operations-log", Log)
        log.clear()
        log.write_line(f"🔄 Evaluating {self.selected_model}...")

        progress = self.query_one("#operation-progress", ProgressBar)
        progress.update(progress=0)

        try:
            import subprocess

            cmd = ["python", "-m", "axolotl.cli.evaluate", model_info["path"]]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            for line in process.stdout:
                log.write_line(line.strip())
                progress.advance(10)

            process.wait()

            if process.returncode == 0:
                log.write_line("✅ Evaluation completed successfully!")
                progress.update(progress=100)
            else:
                log.write_line(f"❌ Evaluation failed with code {process.returncode}")

        except Exception as e:
            log.write_line(f"❌ Error during evaluation: {str(e)}")

    @on(Button.Pressed, "#refresh")
    def handle_refresh(self) -> None:
        """Refresh model list."""
        self.load_models()

    def action_merge_lora(self) -> None:
        """Merge LoRA adapters."""
        self.handle_merge_lora()

    def action_quantize(self) -> None:
        """Quantize model."""
        self.handle_quantize()

    def action_evaluate(self) -> None:
        """Evaluate model."""
        self.handle_evaluate()

    def action_refresh(self) -> None:
        """Refresh model list."""
        self.handle_refresh()
