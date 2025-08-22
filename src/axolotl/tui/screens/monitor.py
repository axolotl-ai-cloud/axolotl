"""System monitoring screen for Axolotl TUI."""

import psutil
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    Log,
    ProgressBar,
    Sparkline,
    Static,
    TabbedContent,
    TabPane,
)

from axolotl.tui.screens.base import BaseScreen


class MonitorScreen(BaseScreen):
    """System monitoring screen."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("ctrl+k", "kill_process", "Kill Process"),
    ]

    CSS = """
    .monitor-container {
        layout: vertical;
        height: 100%;
    }

    .metrics-grid {
        layout: horizontal;
        height: 20%;
        padding: 1;
    }

    .metric-card {
        width: 25%;
        border: solid $surface;
        padding: 1;
        margin: 0 1;
    }

    .metric-label {
        text-style: bold;
        color: $text-muted;
        text-align: center;
    }

    .metric-value {
        text-style: bold;
        text-align: center;
        padding: 1;
        font-size: 2;
    }

    .charts-container {
        height: 40%;
        layout: horizontal;
        padding: 1;
    }

    .chart-panel {
        width: 50%;
        border: solid $info;
        padding: 1;
        margin: 0 1;
    }

    .processes-container {
        height: 40%;
        border: solid $warning;
        padding: 1;
        margin: 1;
    }

    DataTable {
        height: 90%;
    }

    .process-controls {
        layout: horizontal;
        height: 4;
        align: center middle;
        padding: 1;
    }

    .process-controls Button {
        margin: 0 1;
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

    Sparkline {
        height: 8;
    }

    ProgressBar {
        margin: 1 0;
    }
    """

    def __init__(self):
        """Initialize the monitor screen."""
        super().__init__(
            title="System Monitor",
            subtitle="Monitor system resources and running processes",
        )
        self.cpu_history = []
        self.memory_history = []
        self.gpu_history = []

    def compose(self) -> ComposeResult:
        """Compose the monitor screen layout."""
        yield Header()
        yield Container(
            Static("🦾 System Monitor", classes="screen-title"),
            Static(
                "Monitor system resources and running processes",
                classes="screen-subtitle",
            ),
            Container(
                Container(
                    Container(
                        Static("CPU Usage", classes="metric-label"),
                        Static("0%", id="cpu-usage", classes="metric-value"),
                        ProgressBar(total=100, id="cpu-progress"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("Memory", classes="metric-label"),
                        Static("0%", id="memory-usage", classes="metric-value"),
                        ProgressBar(total=100, id="memory-progress"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("GPU Usage", classes="metric-label"),
                        Static("0%", id="gpu-usage", classes="metric-value"),
                        ProgressBar(total=100, id="gpu-progress"),
                        classes="metric-card",
                    ),
                    Container(
                        Static("Temperature", classes="metric-label"),
                        Static("0°C", id="temperature", classes="metric-value"),
                        classes="metric-card",
                    ),
                    classes="metrics-grid",
                ),
                Container(
                    Container(
                        Label("CPU History"),
                        Sparkline([], id="cpu-sparkline"),
                        classes="chart-panel",
                    ),
                    Container(
                        Label("Memory History"),
                        Sparkline([], id="memory-sparkline"),
                        classes="chart-panel",
                    ),
                    classes="charts-container",
                ),
                Container(
                    TabbedContent(
                        TabPane(
                            "Processes",
                            Container(
                                DataTable(id="process-table"),
                                Container(
                                    Button(
                                        "Kill Process",
                                        id="kill-process",
                                        variant="error",
                                    ),
                                    Button("Refresh", id="refresh", variant="default"),
                                    Button(
                                        "Auto Refresh",
                                        id="auto-refresh",
                                        variant="primary",
                                    ),
                                    classes="process-controls",
                                ),
                            ),
                        ),
                        TabPane(
                            "GPU Info",
                            Log(id="gpu-info", wrap=True, highlight=True),
                        ),
                        TabPane(
                            "System Logs",
                            Log(id="system-logs", wrap=True, highlight=True),
                        ),
                    ),
                    classes="processes-container",
                ),
                classes="monitor-container",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self.setup_process_table()
        self.start_monitoring()

        # Initial system info
        self.update_system_info()
        self.update_gpu_info()

    def setup_process_table(self) -> None:
        """Setup the process table."""
        table = self.query_one("#process-table", DataTable)
        table.add_columns("PID", "Name", "CPU%", "Memory%", "Status")
        table.cursor_type = "row"
        table.zebra_stripes = True

    def start_monitoring(self) -> None:
        """Start the monitoring timer."""
        self.set_interval(2.0, self.update_system_metrics)

    @work(thread=True)
    async def update_system_metrics(self) -> None:
        """Update system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_history.append(cpu_percent)
            if len(self.cpu_history) > 50:
                self.cpu_history.pop(0)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_history.append(memory_percent)
            if len(self.memory_history) > 50:
                self.memory_history.pop(0)

            # GPU usage (if available)
            gpu_percent = self.get_gpu_usage()
            self.gpu_history.append(gpu_percent)
            if len(self.gpu_history) > 50:
                self.gpu_history.pop(0)

            # Temperature
            temperature = self.get_temperature()

            # Update UI
            self.update_metrics_display(
                cpu_percent, memory_percent, gpu_percent, temperature
            )
            self.update_sparklines()
            self.update_process_table()

        except Exception as e:
            log = self.query_one("#system-logs", Log)
            log.write_line(f"Error updating metrics: {str(e)}")

    def get_gpu_usage(self) -> float:
        """Get GPU usage percentage."""
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu
        except Exception:
            return 0.0

    def get_temperature(self) -> str:
        """Get system temperature."""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                for name, entries in temps.items():
                    if entries:
                        return f"{entries[0].current:.1f}°C"
            return "N/A"
        except Exception:
            return "N/A"

    def update_metrics_display(
        self, cpu: float, memory: float, gpu: float, temp: str
    ) -> None:
        """Update metrics display."""
        self.query_one("#cpu-usage", Static).update(f"{cpu:.1f}%")
        self.query_one("#memory-usage", Static).update(f"{memory:.1f}%")
        self.query_one("#gpu-usage", Static).update(f"{gpu:.1f}%")
        self.query_one("#temperature", Static).update(temp)

        self.query_one("#cpu-progress", ProgressBar).update(progress=cpu)
        self.query_one("#memory-progress", ProgressBar).update(progress=memory)
        self.query_one("#gpu-progress", ProgressBar).update(progress=gpu)

    def update_sparklines(self) -> None:
        """Update sparkline charts."""
        if self.cpu_history:
            cpu_sparkline = self.query_one("#cpu-sparkline", Sparkline)
            cpu_sparkline.data = self.cpu_history

        if self.memory_history:
            memory_sparkline = self.query_one("#memory-sparkline", Sparkline)
            memory_sparkline.data = self.memory_history

    def update_process_table(self) -> None:
        """Update the process table."""
        table = self.query_one("#process-table", DataTable)
        table.clear()

        try:
            # Get top processes by CPU usage
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "status"]
            ):
                try:
                    pinfo = proc.info
                    if pinfo["cpu_percent"] > 0.1:  # Only show processes using CPU
                        processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

            # Sort by CPU usage
            processes.sort(key=lambda x: x["cpu_percent"], reverse=True)

            # Add top 20 processes
            for proc in processes[:20]:
                table.add_row(
                    str(proc["pid"]),
                    proc["name"][:20],
                    f"{proc['cpu_percent']:.1f}%",
                    f"{proc['memory_percent']:.1f}%",
                    proc["status"],
                )

        except Exception as e:
            log = self.query_one("#system-logs", Log)
            log.write_line(f"Error updating process table: {str(e)}")

    def update_system_info(self) -> None:
        """Update system information."""
        try:
            # System info
            boot_time = psutil.boot_time()
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()

            log = self.query_one("#system-logs", Log)
            log.write_line(f"System started. CPU cores: {cpu_count}")
            log.write_line(f"Total memory: {memory.total / (1024**3):.1f} GB")
            log.write_line(f"Available memory: {memory.available / (1024**3):.1f} GB")

        except Exception as e:
            log = self.query_one("#system-logs", Log)
            log.write_line(f"Error getting system info: {str(e)}")

    def update_gpu_info(self) -> None:
        """Update GPU information."""
        try:
            import pynvml

            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            log = self.query_one("#gpu-info", Log)
            log.clear()
            log.write_line(f"Found {device_count} GPU(s)")

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode()
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

                log.write_line(f"\nGPU {i}: {name}")
                log.write_line(
                    f"Memory: {memory_info.used / (1024**3):.1f} / {memory_info.total / (1024**3):.1f} GB"
                )
                log.write_line(f"Free: {memory_info.free / (1024**3):.1f} GB")

        except Exception as e:
            log = self.query_one("#gpu-info", Log)
            log.clear()
            log.write_line(f"GPU info unavailable: {str(e)}")

    @on(Button.Pressed, "#kill-process")
    def handle_kill_process(self) -> None:
        """Kill selected process."""
        table = self.query_one("#process-table", DataTable)
        if table.cursor_row >= 0:
            try:
                row = table.get_row_at(table.cursor_row)
                pid = int(row[0])

                process = psutil.Process(pid)
                process.terminate()

                log = self.query_one("#system-logs", Log)
                log.write_line(f"Terminated process {pid}")

            except Exception as e:
                log = self.query_one("#system-logs", Log)
                log.write_line(f"Error killing process: {str(e)}")

    @on(Button.Pressed, "#refresh")
    def handle_refresh(self) -> None:
        """Refresh all metrics."""
        self.update_system_info()
        self.update_gpu_info()

        log = self.query_one("#system-logs", Log)
        log.write_line("Metrics refreshed")

    @on(Button.Pressed, "#auto-refresh")
    def handle_auto_refresh(self) -> None:
        """Toggle auto refresh."""
        log = self.query_one("#system-logs", Log)
        log.write_line("Auto refresh is always enabled (every 2 seconds)")

    def action_refresh(self) -> None:
        """Refresh action."""
        self.handle_refresh()

    def action_kill_process(self) -> None:
        """Kill process action."""
        self.handle_kill_process()
