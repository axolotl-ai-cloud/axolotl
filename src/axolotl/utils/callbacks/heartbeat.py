"""Heartbeat callback to monitor training status.

This module provides a callback that sets up a simple web server on localhost
to report the training status. If the training process crashes or completes,
the heartbeat endpoint will stop returning 200 status codes.
"""

import atexit
import json
import logging
import signal
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class HeartbeatHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the heartbeat server."""

    status = "running"
    last_update = datetime.now().isoformat()
    training_info = {
        "step": 0,
        "epoch": 0,
        "total_steps": 0,
        "loss": None,
    }

    def do_GET(self):  # pylint: disable=invalid-name
        """Handle GET requests to the heartbeat server."""
        if self.path == "/heartbeat":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            response = {
                "status": self.status,
                "last_update": self.last_update,
                **self.training_info,
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Not Found")

    def log_message(self, format, *args):  # pylint: disable=redefined-builtin
        """Override to prevent noisy HTTP logs."""
        return


class HeartbeatServer:
    """Simple HTTP server to provide heartbeat status."""

    def __init__(self, port=224209):
        """
        Initialize the heartbeat server.

        Args:
            port: Port number to run the server on (default: 224209)
        """
        self.port = port
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start the heartbeat server in a separate thread."""
        if self.running:
            return

        def run_server():
            logger.info(f"Starting heartbeat server on port {self.port}")
            self.server = HTTPServer(("localhost", self.port), HeartbeatHTTPHandler)
            self.running = True
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        atexit.register(self.stop)

        original_sigterm = signal.getsignal(signal.SIGTERM)
        original_sigint = signal.getsignal(signal.SIGINT)

        def handle_signal(sig, frame):
            HeartbeatHTTPHandler.status = "stopped"
            HeartbeatHTTPHandler.last_update = datetime.now().isoformat()

            if sig == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(sig, frame)
            elif sig == signal.SIGINT and callable(original_sigint):
                original_sigint(sig, frame)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    def stop(self):
        """Stop the heartbeat server."""
        if self.server and self.running:
            logger.info("Stopping heartbeat server")
            self.running = False
            self.server.shutdown()
            self.server.server_close()

    def update_status(self, status, **training_info):
        """
        Update the server status.

        Args:
            status: Status string ("running", "completed", "crashed", etc.)
            training_info: Additional training information to expose
        """
        HeartbeatHTTPHandler.status = status
        HeartbeatHTTPHandler.last_update = datetime.now().isoformat()

        if training_info:
            HeartbeatHTTPHandler.training_info.update(training_info)


class HeartbeatCallback(TrainerCallback):
    """
    Callback to monitor training status and expose a heartbeat endpoint.
    
    This callback sets up a simple HTTP server that reports the training status.
    If the training process crashes or completes, the heartbeat endpoint will
    stop returning 200 status codes.
    """

    def __init__(self, port=224209, update_frequency=10):
        """
        Initialize the heartbeat callback.

        Args:
            port: Port number to run the server on (default: 224209)
            update_frequency: How often to update the heartbeat status in seconds
        """
        self.server = HeartbeatServer(port=port)
        self.update_frequency = update_frequency
        self.last_update_time = 0
        self.total_steps = 0

    def on_init_end(self, args, **kwargs):
        """Start the heartbeat server when training is initialized."""
        self.server.start()
        self.total_steps = args.max_steps if args.max_steps > 0 else "unknown"
        self.server.update_status("initialized", total_steps=self.total_steps)

    def on_train_begin(self, **kwargs):
        """Update status when training begins."""
        self.server.update_status("training", total_steps=self.total_steps)

    def on_step_end(self, state, control, **kwargs):
        """Update status periodically during training."""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_frequency:
            self.last_update_time = current_time
            self.server.update_status(
                "training",
                step=state.global_step,
                epoch=state.epoch,
                total_steps=self.total_steps,
                loss=state.log_history[-1].get("loss") if state.log_history else None,
            )
        return control

    def on_train_end(self, state, **kwargs):
        """Update status when training ends."""
        self.server.update_status(
            "completed",
            step=state.global_step,
            epoch=state.epoch,
            total_steps=self.total_steps,
        )

    def on_exception(self, state, **kwargs):
        """Update status when an exception occurs during training."""
        exception = kwargs.get("exception", "Unknown exception")
        self.server.update_status(
            "crashed",
            step=state.global_step,
            epoch=state.epoch,
            total_steps=self.total_steps,
            error=str(exception),
        )


def heartbeat_callback_factory(port=224209, update_frequency=10):
    """
    Factory function to create a heartbeat callback.

    Args:
        port: Port number to run the server on (default: 224209)
        update_frequency: How often to update the heartbeat status in seconds

    Returns:
        A HeartbeatCallback instance
    """
    return HeartbeatCallback(port=port, update_frequency=update_frequency)
