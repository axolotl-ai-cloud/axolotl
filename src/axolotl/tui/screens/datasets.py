"""Dataset management screen for Axolotl TUI."""

import json
from pathlib import Path
from typing import Dict, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Log,
    ProgressBar,
    Static,
    TextArea,
)

from axolotl.tui.screens.base import BaseScreen


class DatasetScreen(BaseScreen):
    """Dataset management screen."""

    BINDINGS = [
        Binding("ctrl+p", "preprocess", "Preprocess"),
        Binding("ctrl+v", "preview", "Preview"),
        Binding("ctrl+i", "info", "Info"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    .dataset-container {
        layout: horizontal;
        height: 100%;
    }

    .dataset-list {
        width: 40%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .dataset-details {
        width: 60%;
        border: solid $secondary;
        padding: 1;
        margin: 1;
    }

    .dataset-actions {
        layout: horizontal;
        height: 4;
        align: center middle;
        padding: 1;
    }

    .dataset-actions Button {
        margin: 0 1;
    }

    DataTable {
        height: 100%;
    }

    .preview-container {
        height: 100%;
        border: solid $primary;
        padding: 1;
    }

    TextArea {
        height: 100%;
    }

    .stats-container {
        layout: vertical;
        padding: 1;
    }

    .stat-row {
        layout: horizontal;
        padding: 0 0 1 0;
    }

    .stat-label {
        width: 50%;
        color: $text-muted;
    }

    .stat-value {
        width: 50%;
        text-align: right;
        text-style: bold;
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

    .progress-container {
        padding: 1;
        border: solid $warning;
        margin: 1;
    }
    """

    def __init__(self):
        """Initialize the dataset screen."""
        super().__init__(
            title="Dataset Management",
            subtitle="Browse, preview, and preprocess datasets",
        )
        self.datasets: Dict[str, Dict] = {}
        self.selected_dataset: Optional[str] = None
        self.preprocessing_active = False

    def compose(self) -> ComposeResult:
        """Compose the dataset screen layout."""
        yield Header()
        yield Container(
            Static("🦾 Dataset Management", classes="screen-title"),
            Static(
                "Browse, preview, and preprocess datasets", classes="screen-subtitle"
            ),
            Container(
                Container(
                    Label("Available Datasets"),
                    DataTable(id="dataset-table"),
                    Container(
                        Button("Load Dataset", id="load-dataset", variant="primary"),
                        Button("Preprocess", id="preprocess", variant="success"),
                        Button("Download", id="download", variant="default"),
                        Button("Refresh", id="refresh", variant="default"),
                        classes="dataset-actions",
                    ),
                    classes="dataset-list",
                ),
                Container(
                    TextArea("", id="dataset-preview", read_only=True),
                    Container(
                        Static("Dataset Name:", classes="stat-label"),
                        Static("-", id="stat-name", classes="stat-value"),
                        Static("Type:", classes="stat-label"),
                        Static("-", id="stat-type", classes="stat-value"),
                        Static("Size:", classes="stat-label"),
                        Static("-", id="stat-size", classes="stat-value"),
                        Static("Samples:", classes="stat-label"),
                        Static("-", id="stat-samples", classes="stat-value"),
                        Static("Features:", classes="stat-label"),
                        Static("-", id="stat-features", classes="stat-value"),
                        Static("Format:", classes="stat-label"),
                        Static("-", id="stat-format", classes="stat-value"),
                        Static("Preprocessed:", classes="stat-label"),
                        Static("-", id="stat-preprocessed", classes="stat-value"),
                    ),
                    Log(id="processing-log"),
                    ProgressBar(total=100, id="preprocessing-progress"),
                    classes="dataset-details",
                ),
                classes="dataset-container",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.setup_dataset_table()
        self.load_datasets()

        log = self.query_one("#processing-log", Log)
        log.write_line("Dataset manager ready.")

    def setup_dataset_table(self) -> None:
        """Setup the dataset table."""
        table = self.query_one("#dataset-table", DataTable)
        table.add_columns("Name", "Type", "Size", "Status")
        table.cursor_type = "row"
        table.zebra_stripes = True

    @work(thread=True)
    async def load_datasets(self) -> None:
        """Load available datasets."""
        # Check for local datasets
        datasets_dir = Path("/workspace/datasets")
        if datasets_dir.exists():
            for dataset_path in datasets_dir.glob("*"):
                if dataset_path.is_dir():
                    self.datasets[dataset_path.name] = {
                        "name": dataset_path.name,
                        "path": str(dataset_path),
                        "type": "local",
                        "size": self.get_dir_size(dataset_path),
                        "status": "available",
                    }

        # Check for HuggingFace datasets in configs
        configs_dir = Path("/workspace/configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yaml"):
                try:
                    import yaml

                    with open(config_file) as f:
                        config = yaml.safe_load(f)
                        if "datasets" in config:
                            for ds in config.get("datasets", []):
                                if "path" in ds:
                                    ds_name = ds["path"].split("/")[-1]
                                    self.datasets[ds_name] = {
                                        "name": ds_name,
                                        "path": ds["path"],
                                        "type": ds.get("type", "huggingface"),
                                        "size": "Unknown",
                                        "status": "remote",
                                    }
                except Exception:
                    pass

        self.refresh_dataset_table()

    def get_dir_size(self, path: Path) -> str:
        """Get human-readable directory size."""
        total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        for unit in ["B", "KB", "MB", "GB"]:
            if total_size < 1024.0:
                return f"{total_size:.2f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.2f} TB"

    def refresh_dataset_table(self) -> None:
        """Refresh the dataset table."""
        table = self.query_one("#dataset-table", DataTable)
        table.clear()

        for name, info in self.datasets.items():
            table.add_row(
                name[:30],
                info["type"],
                info["size"],
                info["status"],
            )

    @on(DataTable.RowSelected)
    def handle_dataset_selected(self, event: DataTable.RowSelected) -> None:
        """Handle dataset selection from table."""
        if event.cursor_row >= 0:
            dataset_names = list(self.datasets.keys())
            if event.cursor_row < len(dataset_names):
                self.selected_dataset = dataset_names[event.cursor_row]
                self.load_dataset_preview()
                self.update_dataset_stats()

    @work(thread=True)
    async def load_dataset_preview(self) -> None:
        """Load preview of selected dataset."""
        if not self.selected_dataset:
            return

        dataset_info = self.datasets[self.selected_dataset]
        preview_text = ""

        try:
            if dataset_info["type"] == "local" and Path(dataset_info["path"]).exists():
                # Load first few samples from local dataset
                sample_files = list(Path(dataset_info["path"]).glob("*.json"))[:3]
                samples = []
                for sample_file in sample_files:
                    with open(sample_file) as f:
                        samples.append(json.load(f))

                preview_text = json.dumps(samples, indent=2)
            else:
                # Show dataset info for remote datasets
                preview_text = json.dumps(dataset_info, indent=2)

        except Exception as e:
            preview_text = f"Error loading preview: {str(e)}"

        preview = self.query_one("#dataset-preview", TextArea)
        preview.load_text(preview_text)

    def update_dataset_stats(self) -> None:
        """Update dataset statistics display."""
        if not self.selected_dataset:
            return

        info = self.datasets[self.selected_dataset]

        self.query_one("#stat-name", Static).update(info["name"])
        self.query_one("#stat-type", Static).update(info["type"])
        self.query_one("#stat-size", Static).update(info["size"])
        self.query_one("#stat-samples", Static).update("N/A")
        self.query_one("#stat-features", Static).update("N/A")
        self.query_one("#stat-format", Static).update("JSON")
        self.query_one("#stat-preprocessed", Static).update("No")

    @on(Button.Pressed, "#preprocess")
    @work(thread=True)
    async def handle_preprocess(self) -> None:
        """Preprocess selected dataset."""
        if not self.selected_dataset or self.preprocessing_active:
            return

        self.preprocessing_active = True
        dataset_info = self.datasets[self.selected_dataset]

        log = self.query_one("#processing-log", Log)
        log.clear()
        log.write_line(f"🔄 Starting preprocessing for {self.selected_dataset}...")

        progress = self.query_one("#preprocessing-progress", ProgressBar)
        progress.update(progress=0)

        try:
            import subprocess
            import tempfile

            # Create a temporary config for preprocessing
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                config = {
                    "datasets": [
                        {
                            "path": dataset_info["path"],
                            "type": dataset_info.get("type", "alpaca"),
                        }
                    ],
                    "output_dir": f"/tmp/preprocessed_{self.selected_dataset}",
                }
                import yaml

                yaml.dump(config, f)
                temp_config = f.name

            # Run preprocessing
            cmd = ["python", "-m", "axolotl.cli.preprocess", temp_config]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Monitor progress
            for line in process.stdout:
                log.write_line(line.strip())
                # Update progress bar based on output
                if "Processing" in line:
                    progress.advance(10)

            process.wait()

            if process.returncode == 0:
                log.write_line("✅ Preprocessing completed successfully!")
                dataset_info["status"] = "preprocessed"
                progress.update(progress=100)
            else:
                log.write_line(
                    f"❌ Preprocessing failed with code {process.returncode}"
                )

            import os

            os.unlink(temp_config)

        except Exception as e:
            log.write_line(f"❌ Error during preprocessing: {str(e)}")
        finally:
            self.preprocessing_active = False
            self.refresh_dataset_table()

    @on(Button.Pressed, "#load-dataset")
    async def handle_load_dataset(self) -> None:
        """Load a new dataset."""
        log = self.query_one("#processing-log", Log)
        log.write_line("📦 Load dataset functionality coming soon...")

    @on(Button.Pressed, "#download")
    @work(thread=True)
    async def handle_download(self) -> None:
        """Download a remote dataset."""
        if not self.selected_dataset:
            return

        dataset_info = self.datasets[self.selected_dataset]
        if dataset_info["type"] != "huggingface":
            return

        log = self.query_one("#processing-log", Log)
        log.clear()
        log.write_line(f"📥 Downloading {self.selected_dataset} from HuggingFace...")

        try:
            from datasets import load_dataset

            dataset = load_dataset(dataset_info["path"])
            save_path = Path(f"/workspace/datasets/{self.selected_dataset}")
            save_path.mkdir(parents=True, exist_ok=True)

            dataset.save_to_disk(str(save_path))

            log.write_line(f"✅ Downloaded to {save_path}")
            dataset_info["type"] = "local"
            dataset_info["status"] = "available"
            dataset_info["path"] = str(save_path)
            self.refresh_dataset_table()

        except Exception as e:
            log.write_line(f"❌ Download failed: {str(e)}")

    @on(Button.Pressed, "#refresh")
    def handle_refresh(self) -> None:
        """Refresh dataset list."""
        self.load_datasets()

    def action_preprocess(self) -> None:
        """Preprocess selected dataset."""
        self.handle_preprocess()

    def action_refresh(self) -> None:
        """Refresh dataset list."""
        self.handle_refresh()
