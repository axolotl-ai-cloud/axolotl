"""Training management screen for Axolotl TUI."""

import asyncio
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    Log,
    ProgressBar,
    Select,
    Sparkline,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from axolotl.tui.screens.base import BaseScreen


@dataclass
class TrainingJob:
    """Represents a training job."""

    id: str
    config_path: str
    status: str  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    process: Optional[subprocess.Popen] = None
    log_file: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float = 0.0
    losses: List[float] = None

    def __post_init__(self):
        if self.losses is None:
            self.losses = []


class TrainingScreen(BaseScreen):
    """Training management screen."""

    BINDINGS = [
        Binding("ctrl+t", "new_training", "New Training"),
        Binding("ctrl+r", "resume_training", "Resume"),
        Binding("ctrl+x", "stop_training", "Stop"),
        Binding("ctrl+l", "view_logs", "View Logs"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = """
    .training-container {
        layout: vertical;
        height: 100%;
    }

    .job-list-container {
        height: 40%;
        border: solid $primary;
        padding: 1;
        margin: 1;
    }

    .job-details-container {
        height: 60%;
        padding: 1;
    }

    .control-panel {
        layout: horizontal;
        height: 4;
        align: center middle;
        padding: 1;
        border: solid $secondary;
        margin: 1;
    }

    .control-panel Button {
        margin: 0 1;
    }

    .metrics-panel {
        layout: horizontal;
        height: 10;
        border: solid $info;
        padding: 1;
        margin: 1;
    }

    .metric-card {
        width: 25%;
        border: tall $surface;
        padding: 1;
        margin: 0 1;
    }

    .metric-label {
        text-style: bold;
        color: $text-muted;
    }

    .metric-value {
        text-style: bold;
        text-align: center;
        padding: 1;
    }

    .log-viewer {
        border: solid $warning;
        padding: 1;
        margin: 1;
    }

    #training-logs {
        height: 100%;
    }

    DataTable {
        height: 100%;
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

    .sparkline-container {
        height: 5;
        border: solid $success;
        padding: 1;
        margin: 1;
    }
    """

    def __init__(self):
        """Initialize the training screen."""
        super().__init__(
            title="Training Management",
            subtitle="Launch, monitor, and manage training jobs",
        )
        self.jobs: Dict[str, TrainingJob] = {}
        self.selected_job_id: Optional[str] = None
        self.update_timer = None

    def compose(self) -> ComposeResult:
        """Compose the training screen layout."""
        yield Header()
        yield Container(
            Static("🦾 Training Management", classes="screen-title"),
            Static(
                "Launch, monitor, and manage training jobs", classes="screen-subtitle"
            ),
            Container(
                Container(
                    Label("Active Training Jobs"),
                    DataTable(id="job-table"),
                    classes="job-list-container",
                ),
                Container(
                    Button("New Training", id="new-training", variant="primary"),
                    Button("Resume", id="resume-training", variant="success"),
                    Button("Stop", id="stop-training", variant="error"),
                    Button("View Logs", id="view-logs", variant="default"),
                    Button("Clear Completed", id="clear-completed", variant="warning"),
                    Button("Refresh", id="refresh", variant="default"),
                    classes="control-panel",
                ),
                Container(
                    Container(
                        Static("Current Epoch", classes="metric-label"),
                        Static("0 / 0", id="epoch-metric", classes="metric-value"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("Loss", classes="metric-label"),
                        Static("0.000", id="loss-metric", classes="metric-value"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("Status", classes="metric-label"),
                        Static("Idle", id="status-metric", classes="metric-value"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("Duration", classes="metric-label"),
                        Static(
                            "00:00:00", id="duration-metric", classes="metric-value"
                        ),
                        classes="metric-card",
                    ),
                    classes="metrics-panel",
                ),
                Container(
                    Label("Loss History"),
                    Sparkline(
                        [],
                        id="loss-sparkline",
                        summary_function=min,
                    ),
                    classes="sparkline-container",
                ),
                Container(
                    TabbedContent(
                        TabPane(
                            "Training Logs",
                            Log(id="training-logs", wrap=True, highlight=True),
                        ),
                        TabPane(
                            "System Logs",
                            Log(id="system-logs", wrap=True, highlight=True),
                        ),
                        TabPane(
                            "Validation",
                            Log(id="validation-logs", wrap=True, highlight=True),
                        ),
                    ),
                    classes="log-viewer",
                ),
                classes="job-details-container",
            ),
            classes="training-container",
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.setup_job_table()
        self.start_update_timer()

        log = self.query_one("#training-logs", Log)
        log.write_line(
            "Training manager ready. Select a configuration to start training."
        )

    def setup_job_table(self) -> None:
        """Setup the job table."""
        table = self.query_one("#job-table", DataTable)
        table.add_columns("ID", "Config", "Status", "Epoch", "Loss", "Duration")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def start_update_timer(self) -> None:
        """Start the periodic update timer."""
        self.set_interval(2.0, self.update_job_status)

    @work(thread=True)
    async def update_job_status(self) -> None:
        """Update job status periodically."""
        for job_id, job in self.jobs.items():
            if job.status == "running" and job.process:
                poll = job.process.poll()
                if poll is not None:
                    if poll == 0:
                        job.status = "completed"
                    else:
                        job.status = "failed"
                    job.end_time = datetime.now()

        self.refresh_job_table()
        self.update_selected_job_metrics()

    def refresh_job_table(self) -> None:
        """Refresh the job table."""
        table = self.query_one("#job-table", DataTable)
        table.clear()

        for job_id, job in self.jobs.items():
            duration = self.calculate_duration(job)
            table.add_row(
                job_id[:8],
                Path(job.config_path).name,
                job.status,
                f"{job.current_epoch}/{job.total_epochs}",
                f"{job.current_loss:.4f}" if job.current_loss else "N/A",
                duration,
            )

    def calculate_duration(self, job: TrainingJob) -> str:
        """Calculate job duration."""
        if not job.start_time:
            return "00:00:00"

        end_time = job.end_time or datetime.now()
        duration = end_time - job.start_time
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        seconds = int(duration.total_seconds() % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def update_selected_job_metrics(self) -> None:
        """Update metrics for selected job."""
        if not self.selected_job_id or self.selected_job_id not in self.jobs:
            return

        job = self.jobs[self.selected_job_id]

        self.query_one("#epoch-metric", Static).update(
            f"{job.current_epoch} / {job.total_epochs}"
        )
        self.query_one("#loss-metric", Static).update(
            f"{job.current_loss:.4f}" if job.current_loss else "N/A"
        )
        self.query_one("#status-metric", Static).update(job.status.upper())
        self.query_one("#duration-metric", Static).update(self.calculate_duration(job))

        if job.losses:
            sparkline = self.query_one("#loss-sparkline", Sparkline)
            sparkline.data = job.losses[-50:]  # Show last 50 loss values

    @on(DataTable.RowSelected)
    def handle_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle job selection from table."""
        if event.row_index >= 0:
            job_ids = list(self.jobs.keys())
            if event.row_index < len(job_ids):
                self.selected_job_id = job_ids[event.row_index]
                self.update_selected_job_metrics()
                self.load_job_logs()

    def load_job_logs(self) -> None:
        """Load logs for selected job."""
        if not self.selected_job_id or self.selected_job_id not in self.jobs:
            return

        job = self.jobs[self.selected_job_id]
        if job.log_file and Path(job.log_file).exists():
            try:
                with open(job.log_file, "r") as f:
                    content = f.read()
                    log = self.query_one("#training-logs", Log)
                    log.clear()
                    for line in content.split("\n")[-100:]:  # Show last 100 lines
                        if line.strip():
                            log.write_line(line)
            except Exception as e:
                log = self.query_one("#training-logs", Log)
                log.write_line(f"Error loading logs: {str(e)}")

    @on(Button.Pressed, "#new-training")
    async def handle_new_training(self) -> None:
        """Start a new training job."""
        from axolotl.tui.dialogs.training import NewTrainingDialog

        dialog = NewTrainingDialog()
        result = await self.app.push_screen_wait(dialog)

        if result and "config_path" in result:
            await self.start_training_job(
                result["config_path"], result.get("launcher", "accelerate")
            )

    @work(thread=True)
    async def start_training_job(
        self, config_path: str, launcher: str = "accelerate"
    ) -> None:
        """Start a training job."""
        import uuid
        from datetime import datetime

        job_id = str(uuid.uuid4())
        log_file = f"/tmp/axolotl_training_{job_id}.log"

        job = TrainingJob(
            id=job_id,
            config_path=config_path,
            status="pending",
            start_time=datetime.now(),
            log_file=log_file,
            total_epochs=3,  # Default, should parse from config
        )

        self.jobs[job_id] = job
        self.selected_job_id = job_id

        log = self.query_one("#training-logs", Log)
        log.clear()
        log.write_line(f"🚀 Starting training job {job_id[:8]}...")
        log.write_line(f"Config: {config_path}")
        log.write_line(f"Launcher: {launcher}")

        try:
            if launcher == "accelerate":
                cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config_path]
            else:
                cmd = [
                    "torchrun",
                    "--nproc_per_node=1",
                    "-m",
                    "axolotl.cli.train",
                    config_path,
                ]

            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )

            job.process = process
            job.status = "running"

            log.write_line("✅ Training started successfully!")
            self.refresh_job_table()

            self.monitor_training_output(job_id)

        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            log.write_line(f"❌ Failed to start training: {str(e)}")
            self.refresh_job_table()

    def monitor_training_output(self, job_id: str) -> None:
        """Monitor training output and extract metrics."""
        if job_id not in self.jobs:
            return

        job = self.jobs[job_id]
        if not job.log_file:
            return

        def tail_log():
            import re
            import time

            with open(job.log_file, "r") as f:
                f.seek(0, 2)  # Go to end of file
                while job.status == "running":
                    line = f.readline()
                    if line:
                        # Parse training metrics from log
                        epoch_match = re.search(r"Epoch (\d+)/(\d+)", line)
                        if epoch_match:
                            job.current_epoch = int(epoch_match.group(1))
                            job.total_epochs = int(epoch_match.group(2))

                        loss_match = re.search(
                            r"loss['\"]?\s*:\s*([\d.]+)", line, re.IGNORECASE
                        )
                        if loss_match:
                            job.current_loss = float(loss_match.group(1))
                            job.losses.append(job.current_loss)

                        # Update log viewer
                        self.call_from_thread(self.append_training_log, line.strip())
                    else:
                        time.sleep(0.5)

        thread = threading.Thread(target=tail_log, daemon=True)
        thread.start()

    def append_training_log(self, line: str) -> None:
        """Append line to training log."""
        log = self.query_one("#training-logs", Log)
        log.write_line(line)

    @on(Button.Pressed, "#stop-training")
    def handle_stop_training(self) -> None:
        """Stop selected training job."""
        if not self.selected_job_id or self.selected_job_id not in self.jobs:
            log = self.query_one("#training-logs", Log)
            log.write_line("⚠️ No job selected")
            return

        job = self.jobs[self.selected_job_id]
        if job.status == "running" and job.process:
            job.process.terminate()
            job.status = "stopped"
            job.end_time = datetime.now()

            log = self.query_one("#training-logs", Log)
            log.write_line(f"🛑 Training job {job.id[:8]} stopped")
            self.refresh_job_table()

    @on(Button.Pressed, "#resume-training")
    async def handle_resume_training(self) -> None:
        """Resume a stopped training job."""
        if not self.selected_job_id or self.selected_job_id not in self.jobs:
            log = self.query_one("#training-logs", Log)
            log.write_line("⚠️ No job selected")
            return

        job = self.jobs[self.selected_job_id]
        if job.status in ["stopped", "failed"]:
            await self.start_training_job(job.config_path)

    @on(Button.Pressed, "#clear-completed")
    def handle_clear_completed(self) -> None:
        """Clear completed jobs from the list."""
        completed_jobs = [
            job_id
            for job_id, job in self.jobs.items()
            if job.status in ["completed", "failed", "stopped"]
        ]

        for job_id in completed_jobs:
            del self.jobs[job_id]

        self.refresh_job_table()
        log = self.query_one("#training-logs", Log)
        log.write_line(f"🧹 Cleared {len(completed_jobs)} completed jobs")

    @on(Button.Pressed, "#refresh")
    def handle_refresh(self) -> None:
        """Refresh the job list and metrics."""
        self.refresh_job_table()
        self.update_selected_job_metrics()
        if self.selected_job_id:
            self.load_job_logs()

    @on(Button.Pressed, "#view-logs")
    def handle_view_logs(self) -> None:
        """View full logs for selected job."""
        if not self.selected_job_id or self.selected_job_id not in self.jobs:
            return

        job = self.jobs[self.selected_job_id]
        if job.log_file and Path(job.log_file).exists():
            import subprocess

            subprocess.run(["less", job.log_file])

    def action_new_training(self) -> None:
        """Start a new training job."""
        self.handle_new_training()

    def action_stop_training(self) -> None:
        """Stop selected training job."""
        self.handle_stop_training()

    def action_resume_training(self) -> None:
        """Resume selected training job."""
        self.handle_resume_training()

    def action_refresh(self) -> None:
        """Refresh the display."""
        self.handle_refresh()
