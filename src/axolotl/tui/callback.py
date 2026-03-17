"""AxolotlTUICallback — HF TrainerCallback that feeds metrics to the TUI."""

from __future__ import annotations

import logging
import queue
from typing import Any

from transformers.trainer_callback import TrainerCallback

from axolotl.tui.config import TUIConfig
from axolotl.tui.renderer import TUIRenderer


class _TUILogHandler(logging.Handler):
    """Logging handler that pushes log records into the TUI metric queue."""

    _LEVEL_MAP = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warning",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, metric_queue: queue.Queue, min_level: str = "info"):
        super().__init__()
        level_name = min_level.upper()
        self.setLevel(getattr(logging, level_name, logging.INFO))
        self._queue = metric_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = self._LEVEL_MAP.get(record.levelno, "info")
            msg = self.format(record)
            self._queue.put_nowait({
                "type": "log_line",
                "level": level,
                "message": msg,
            })
        except queue.Full:
            pass
        except Exception:
            self.handleError(record)


class AxolotlTUICallback(TrainerCallback):
    """Pushes training metrics into a queue for the TUI renderer.

    The callback never blocks on the render thread. The queue is bounded
    (maxsize=512) with put_nowait; overflow is silently dropped.
    """

    def __init__(self, config: TUIConfig):
        self._config = config
        self._queue: queue.Queue = queue.Queue(maxsize=4096)
        self._renderer = TUIRenderer(config=config, metric_queue=self._queue)
        self._log_handler: _TUILogHandler | None = None
        self._renderer_started_early: bool = False

    def _put(self, event: dict) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            pass

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Send run info
        run_name = getattr(args, "run_name", "") or ""
        self._put(
            {
                "type": "run_info",
                "run_name": run_name,
                "total_steps": state.max_steps,
                "total_epochs": int(args.num_train_epochs) if args.num_train_epochs else 1,
            }
        )

        if not self._renderer_started_early:
            # Attach a logging handler to feed log messages into the events panel
            self._log_handler = _TUILogHandler(
                self._queue, min_level=self._config.log_level
            )
            self._log_handler.setFormatter(
                logging.Formatter("[%(name)s] %(message)s")
            )
            # Attach to both root and axolotl loggers (axolotl has propagate=False)
            logging.getLogger().addHandler(self._log_handler)
            logging.getLogger("axolotl").addHandler(self._log_handler)

            # Start the renderer background thread
            self._renderer.start()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Filter out non-numeric keys and internal keys
        filtered = {}
        for key, value in logs.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float)):
                filtered[key] = value
            elif isinstance(value, str):
                # HF Trainer sometimes passes string-encoded numbers
                try:
                    filtered[key] = float(value)
                except (ValueError, TypeError):
                    pass

        if filtered:
            self._put({"type": "metrics", "logs": filtered})

    def on_step_end(self, args, state, control, **kwargs):
        self._put(
            {
                "type": "step",
                "step": state.global_step,
                "total_steps": state.max_steps,
                "epoch": state.epoch if state.epoch else 0,
            }
        )

    def on_prediction_step(self, args, state, control, **kwargs):
        pass

    def on_train_end(self, args, state, control, **kwargs):
        self._put({"type": "done"})
        # If renderer was started early, do_train's finally block handles stop
        if not self._renderer_started_early:
            self._renderer.stop()

        # Remove the logging handler (only if we added it)
        if self._log_handler:
            logging.getLogger().removeHandler(self._log_handler)
            logging.getLogger("axolotl").removeHandler(self._log_handler)
            self._log_handler = None
