"""TUIRenderer — background daemon thread that drives the rich.live.Live display."""

from __future__ import annotations

import importlib
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

from axolotl.tui.config import TUIConfig
from axolotl.tui.gpu import GPUPoller
from axolotl.tui.io_capture import (
    IOCapture,
    ParserChain,
    get_registered_parsers,
)
from axolotl.tui.panels import BasePanel, get_registered_panels
from axolotl.tui.state import CompletionSample, LogLine, TUIState

LOG = logging.getLogger(__name__)


class TUIRenderer:
    """Background thread that renders the TUI dashboard using rich.live.Live."""

    def __init__(self, config: TUIConfig, metric_queue: queue.Queue):
        self._config = config
        self._queue = metric_queue
        self._state = TUIState()
        self._gpu_poller = GPUPoller()
        self._panels: list[BasePanel] = []
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._io_capture: IOCapture | None = None
        self._parser_chain: ParserChain | None = None

    def _init_panels(self) -> None:
        registry = get_registered_panels()
        for panel_name in self._config.panels:
            if panel_name in registry:
                self._panels.append(registry[panel_name]())

    def _init_parser_chain(self) -> None:
        self._parser_chain = ParserChain()
        # Register all built-in parsers
        for parser_cls in get_registered_parsers():
            self._parser_chain.register(parser_cls())

        # Load plugin parsers
        for plugin_spec in self._config.parser_plugins:
            try:
                if "::" in plugin_spec:
                    # file path :: class name
                    file_path, class_name = plugin_spec.split("::", 1)
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "custom_parser", file_path
                    )
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    parser_cls = getattr(mod, class_name)
                else:
                    # dotted module path
                    module_path, class_name = plugin_spec.rsplit(".", 1)
                    mod = importlib.import_module(module_path)
                    parser_cls = getattr(mod, class_name)
                self._parser_chain.register(parser_cls())
            except Exception as exc:
                LOG.warning(f"Failed to load parser plugin {plugin_spec}: {exc}")

    def _build_layout(self) -> Layout:
        layout = Layout()

        top_panels = [p for p in self._panels if p.position == "top"]
        left_panels = [p for p in self._panels if p.position == "left"]
        right_panels = [p for p in self._panels if p.position == "right"]
        bottom_panels = [p for p in self._panels if p.position == "bottom"]

        sections = []

        if top_panels:
            layout_top = Layout(name="top", size=3)
            sections.append(layout_top)

        if left_panels or right_panels:
            layout_middle = Layout(name="middle", ratio=3)
            middle_parts = []
            if left_panels:
                middle_parts.append(Layout(name="left", ratio=1))
            if right_panels:
                middle_parts.append(Layout(name="right", ratio=1))
            if middle_parts:
                layout_middle.split_row(*middle_parts)
            sections.append(layout_middle)

        if bottom_panels:
            layout_bottom = Layout(name="bottom", ratio=2)
            if len(bottom_panels) > 1:
                layout_bottom.split_row(
                    *[
                        Layout(name=f"bottom_{i}", ratio=1)
                        for i in range(len(bottom_panels))
                    ]
                )
            sections.append(layout_bottom)

        if sections:
            layout.split_column(*sections)

        return layout

    def _update_layout(self, layout: Layout) -> None:
        top_panels = [p for p in self._panels if p.position == "top"]
        left_panels = [p for p in self._panels if p.position == "left"]
        right_panels = [p for p in self._panels if p.position == "right"]
        bottom_panels = [p for p in self._panels if p.position == "bottom"]

        if top_panels:
            layout["top"].update(top_panels[0].render(self._state))

        if left_panels:
            layout["left"].update(left_panels[0].render(self._state))

        if right_panels:
            layout["right"].update(right_panels[0].render(self._state))

        if bottom_panels:
            if len(bottom_panels) == 1:
                layout["bottom"].update(bottom_panels[0].render(self._state))
            else:
                for i, panel in enumerate(bottom_panels):
                    layout[f"bottom_{i}"].update(panel.render(self._state))

    def _drain_queue(self) -> None:
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break

            # Dispatch event to panels first
            for panel in self._panels:
                panel.on_event(event)

            event_type = event.get("type")

            if event_type == "metrics":
                logs = event.get("logs", {})
                self._apply_metrics(logs)

            elif event_type == "step":
                self._state.current_step = event.get("step", self._state.current_step)
                self._state.total_steps = event.get(
                    "total_steps", self._state.total_steps
                )
                self._state.current_epoch = event.get(
                    "epoch", self._state.current_epoch
                )
                now = time.time()
                self._state.elapsed_seconds = now - self._state.start_time.timestamp()
                if (
                    self._state.current_step > 0
                    and self._state.total_steps > 0
                ):
                    rate = self._state.elapsed_seconds / self._state.current_step
                    remaining = self._state.total_steps - self._state.current_step
                    self._state.eta_seconds = rate * remaining

            elif event_type == "log_line":
                level = event.get("level", "info")
                message = event.get("message", "")
                self._state.log_lines.append(
                    LogLine(
                        timestamp=datetime.now(),
                        level=level,
                        message=message,
                    )
                )

            elif event_type == "completion":
                self._state.completions.append(
                    CompletionSample(
                        step=event.get("step", 0),
                        prompt=event.get("prompt", ""),
                        completion=event.get("completion", ""),
                        reward=event.get("reward"),
                        advantage=event.get("advantage"),
                    )
                )

            elif event_type == "run_info":
                if "run_name" in event:
                    self._state.run_name = event["run_name"]
                if "model_name" in event:
                    self._state.model_name = event["model_name"]
                if "training_mode" in event:
                    self._state.training_mode = event["training_mode"]
                if "world_size" in event:
                    self._state.world_size = event["world_size"]
                if "total_steps" in event:
                    self._state.total_steps = event["total_steps"]
                if "total_epochs" in event:
                    self._state.total_epochs = event["total_epochs"]

            elif event_type == "done":
                self._stop_event.set()

    def _apply_metrics(self, logs: dict[str, Any]) -> None:
        metric_map = {
            "loss": "loss",
            "grad_norm": "grad_norm",
            "learning_rate": "learning_rate",
            "tokens_per_second": "tokens_per_second",
            "samples_per_second": "samples_per_second",
            "mfu": "mfu",
            "rewards/mean": "rewards_mean",
            "rewards_mean": "rewards_mean",
            "rewards/std": "rewards_std",
            "rewards_std": "rewards_std",
            "kl": "kl_divergence",
            "kl_divergence": "kl_divergence",
            "clip_ratio": "clip_ratio",
            "queue_size": "queue_size",
        }

        for key, value in logs.items():
            if key in metric_map:
                setattr(self._state, metric_map[key], value)
            else:
                self._state.extra[key] = value

        if "loss" in logs and logs["loss"] is not None:
            self._state.loss_history.append(logs["loss"])

    def start(self) -> None:
        self._init_panels()
        self._init_parser_chain()

        # Set up I/O capture
        self._io_capture = IOCapture(
            log_path=self._config.stdout_log_path,
            parser_chain=self._parser_chain,
            metric_queue=self._queue,
        )

        # Monkeypatch tqdm to suppress terminal output and route through our queue.
        # This prevents tqdm progress bars from flickering through the TUI and
        # ensures all progress events appear in the Events panel.
        self._install_tqdm_hook()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _install_tqdm_hook(self) -> None:
        """Replace tqdm's display method to route updates through TUI queue."""
        try:
            import tqdm
            import tqdm.auto
            import io

            q = self._queue
            parser = self._tqdm_parser = None
            # Find our tqdm parser in the chain
            for p in (self._parser_chain._parsers if self._parser_chain else []):
                if p.name == "tqdm":
                    self._tqdm_parser = p
                    break

            # Save originals for restore
            self._orig_tqdm_init = tqdm.tqdm.__init__
            self._orig_tqdm_class = tqdm.auto.tqdm

            renderer_self = self

            class TUITqdm(tqdm.tqdm):
                """tqdm subclass that sends progress to TUI instead of terminal."""

                def __init__(self, *args, **kwargs):
                    # Force output to devnull so nothing reaches the terminal
                    kwargs["file"] = io.StringIO()
                    kwargs["dynamic_ncols"] = False
                    kwargs["ncols"] = 80
                    super().__init__(*args, **kwargs)

                def display(self, msg=None, pos=None):
                    # Build a progress string and push to queue
                    if self.total and self.total > 0:
                        pct = self.n / self.total * 100
                        desc = self.desc.rstrip(": ") if self.desc else ""
                        elapsed = self.format_interval(self.elapsed) if hasattr(self, 'elapsed') and self.elapsed else "?"
                        rate = self.format_sizeof(1.0 / self.avg_time) if hasattr(self, 'avg_time') and self.avg_time else "?"

                        # Emit events at milestones or at low frequency
                        is_milestone = (
                            self.n == 0 or
                            self.n >= self.total or
                            int(pct) % 25 == 0
                        )
                        if is_milestone:
                            try:
                                q.put_nowait({
                                    "type": "log_line",
                                    "level": "info",
                                    "message": f"[{desc}] {pct:.0f}% ({self.n}/{self.total})" if desc else f"{pct:.0f}% ({self.n}/{self.total})",
                                })
                            except Exception:
                                pass

                        try:
                            metric_key = f"progress/{desc.lower().replace(' ', '_')}" if desc else "progress/unknown"
                            q.put_nowait({
                                "type": "metrics",
                                "logs": {metric_key: pct / 100.0},
                            })
                        except Exception:
                            pass

                def close(self):
                    # Emit final completion event
                    if self.total and self.total > 0 and self.n > 0:
                        desc = self.desc.rstrip(": ") if self.desc else ""
                        try:
                            q.put_nowait({
                                "type": "log_line",
                                "level": "info",
                                "message": f"[{desc}] 100% ({self.total}/{self.total}) done" if desc else f"100% ({self.total}/{self.total}) done",
                            })
                        except Exception:
                            pass
                    super().close()

            # Replace tqdm globally
            tqdm.auto.tqdm = TUITqdm
            tqdm.tqdm = TUITqdm
            # Also patch tqdm.std which some libraries use directly
            tqdm.std.tqdm = TUITqdm
            self._tui_tqdm_cls = TUITqdm

        except Exception as exc:
            LOG.debug(f"Failed to install tqdm hook: {exc}")

    def _uninstall_tqdm_hook(self) -> None:
        """Restore original tqdm."""
        try:
            import tqdm
            import tqdm.auto

            if hasattr(self, "_orig_tqdm_class"):
                tqdm.auto.tqdm = self._orig_tqdm_class
            if hasattr(self, "_orig_tqdm_init"):
                tqdm.tqdm.__init__ = self._orig_tqdm_init
                tqdm.std.tqdm = tqdm.tqdm
        except Exception:
            pass

    def stop(self) -> None:
        self._stop_event.set()
        self._uninstall_tqdm_hook()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        import os

        # Save a handle to the REAL terminal BEFORE IO capture redirects fds.
        # This ensures rich.live.Live writes to the terminal, not the pipe.
        saved_tty_fd = os.dup(1)
        tty_file = os.fdopen(saved_tty_fd, "w", buffering=1, closefd=True)
        console = Console(file=tty_file)

        layout = self._build_layout()
        tick_interval = 1.0 / max(self._config.refresh_rate, 1)
        gpu_poll_counter = 0
        gpu_poll_ticks = max(
            1, int(self._config.hardware_poll_interval / tick_interval)
        )

        # Start I/O capture — redirects fd 1/2 to pipe AFTER we saved the tty fd
        if self._io_capture:
            self._io_capture.start()

        try:
            with Live(
                layout,
                console=console,
                refresh_per_second=self._config.refresh_rate,
                screen=True,
                redirect_stdout=False,
                redirect_stderr=False,
            ) as live:
                while not self._stop_event.is_set():
                    self._drain_queue()

                    # Poll GPU stats periodically
                    gpu_poll_counter += 1
                    if gpu_poll_counter >= gpu_poll_ticks:
                        gpu_poll_counter = 0
                        if self._gpu_poller.available:
                            self._state.gpus = self._gpu_poller.poll()

                    # Update elapsed time
                    self._state.elapsed_seconds = (
                        time.time() - self._state.start_time.timestamp()
                    )

                    self._update_layout(layout)
                    live.update(layout)

                    time.sleep(tick_interval)

                # Final drain
                self._drain_queue()
                self._update_layout(layout)
                live.update(layout)
        finally:
            if self._io_capture:
                self._io_capture.stop()
            try:
                tty_file.close()
            except Exception:
                pass
