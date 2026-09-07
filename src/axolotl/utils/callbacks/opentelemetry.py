"""OpenTelemetry metrics callback for Axolotl training"""

import threading
from typing import Dict, Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

try:
    from opentelemetry import metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.metrics import set_meter_provider
    from opentelemetry.sdk.metrics import MeterProvider as SDKMeterProvider
    from prometheus_client import start_http_server

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    LOG.warning("OpenTelemetry not available. pip install [opentelemetry]")
    OPENTELEMETRY_AVAILABLE = False


class OpenTelemetryMetricsCallback(TrainerCallback):
    """
    TrainerCallback that exports training metrics to OpenTelemetry/Prometheus.

    This callback automatically tracks key training metrics including:
    - Training loss
    - Evaluation loss
    - Learning rate
    - Epoch progress
    - Global step count
    - Gradient norm

    Metrics are exposed via HTTP endpoint for Prometheus scraping.
    """

    def __init__(self, cfg):
        if not OPENTELEMETRY_AVAILABLE:
            LOG.warning("OpenTelemetry not available, metrics will not be collected")
            self.metrics_enabled = False
            return

        self.cfg = cfg
        self.metrics_host = getattr(cfg, "otel_metrics_host", "localhost")
        self.metrics_port = getattr(cfg, "otel_metrics_port", 8000)
        self.metrics_enabled = True
        self.server_started = False
        self.metrics_lock = threading.Lock()

        try:
            # Create Prometheus metrics reader
            prometheus_reader = PrometheusMetricReader()

            # Create meter provider with Prometheus exporter
            provider = SDKMeterProvider(metric_readers=[prometheus_reader])
            set_meter_provider(provider)

            # Get meter for creating metrics
            self.meter = metrics.get_meter("axolotl.training")

            # Create metrics
            self._create_metrics()

        except Exception as e:
            LOG.warning(f"Failed to initialize OpenTelemetry metrics: {e}")
            self.metrics_enabled = False

    def _create_metrics(self):
        """Create all metrics that will be tracked"""
        self.train_loss_gauge = self.meter.create_gauge(
            name="axolotl_train_loss",
            description="Current training loss",
            unit="1",
        )

        self.eval_loss_gauge = self.meter.create_gauge(
            name="axolotl_eval_loss",
            description="Current evaluation loss",
            unit="1",
        )

        self.learning_rate_gauge = self.meter.create_gauge(
            name="axolotl_learning_rate",
            description="Current learning rate",
            unit="1",
        )

        self.epoch_gauge = self.meter.create_gauge(
            name="axolotl_epoch",
            description="Current training epoch",
            unit="1",
        )

        self.global_step_counter = self.meter.create_counter(
            name="axolotl_global_steps",
            description="Total training steps completed",
            unit="1",
        )

        self.grad_norm_gauge = self.meter.create_gauge(
            name="axolotl_gradient_norm",
            description="Gradient norm",
            unit="1",
        )

        self.memory_usage_gauge = self.meter.create_gauge(
            name="axolotl_memory_usage",
            description="Current memory usage in MB",
            unit="MB",
        )

    def _start_metrics_server(self):
        """Start the HTTP server for metrics exposure"""
        if self.server_started:
            return

        try:
            start_http_server(self.metrics_port, addr=self.metrics_host)
            self.server_started = True
            LOG.info(
                f"OpenTelemetry metrics server started on http://{self.metrics_host}:{self.metrics_port}/metrics"
            )

        except Exception as e:
            LOG.error(f"Failed to start OpenTelemetry metrics server: {e}")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the beginning of training"""
        if not self.metrics_enabled:
            return

        self._start_metrics_server()
        LOG.info("OpenTelemetry metrics collection started")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called when logging occurs"""
        if not self.metrics_enabled or not logs:
            return

        if "loss" in logs:
            self.train_loss_gauge.set(logs["loss"])

        if "eval_loss" in logs:
            self.eval_loss_gauge.set(logs["eval_loss"])

        if "learning_rate" in logs:
            self.learning_rate_gauge.set(logs["learning_rate"])

        if "epoch" in logs:
            self.epoch_gauge.set(logs["epoch"])

        if "grad_norm" in logs:
            self.grad_norm_gauge.set(logs["grad_norm"])
        if "memory_usage" in logs:
            self.memory_usage_gauge.set(logs["memory_usage"])

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of each training step"""
        if not self.metrics_enabled:
            return

        # Update step counter and epoch
        self.global_step_counter.add(1)
        if state.epoch is not None:
            self.epoch_gauge.set(state.epoch)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Called after evaluation"""
        if not self.metrics_enabled or not metrics:
            return

        if "eval_loss" in metrics:
            self.eval_loss_gauge.set(metrics["eval_loss"])

        # Record any other eval metrics as gauges
        for key, value in metrics.items():
            if key.startswith("eval_") and isinstance(value, (int, float)):
                # Create gauge for this metric if it doesn't exist
                gauge_name = f"axolotl_{key}"
                try:
                    gauge = self.meter.create_gauge(
                        name=gauge_name,
                        description=f"Evaluation metric: {key}",
                        unit="1",
                    )
                    gauge.set(value)
                except Exception as e:
                    LOG.warning(f"Failed to create/update metric {gauge_name}: {e}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called at the end of training"""
        if not self.metrics_enabled:
            return

        LOG.info("Training completed. OpenTelemetry metrics collection finished.")
        LOG.info(
            f"Metrics are still available at http://{self.metrics_host}:{self.metrics_port}/metrics"
        )
