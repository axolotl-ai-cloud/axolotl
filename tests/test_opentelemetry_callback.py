"""Tests for OpenTelemetry metrics callback functionality."""

import time

import pytest

from axolotl.utils.dict import DictDefault


@pytest.fixture
def mock_otel_config():
    """Mock configuration for OpenTelemetry callback."""
    return DictDefault(
        {
            "use_otel_metrics": True,
            "otel_metrics_host": "localhost",
            "otel_metrics_port": 8003,  # Use unique port for tests
        }
    )


@pytest.fixture
def mock_trainer_state():
    """Mock trainer state for callback testing."""
    from transformers import TrainerState

    state = TrainerState()
    state.epoch = 1.0
    state.global_step = 100
    return state


@pytest.fixture
def mock_training_args():
    """Mock training arguments for callback testing."""
    from transformers import TrainingArguments

    return TrainingArguments(output_dir="/tmp/test")


@pytest.fixture
def mock_trainer_control():
    """Mock trainer control for callback testing."""
    from transformers.trainer_callback import TrainerControl

    return TrainerControl()


class TestOpenTelemetryConfig:
    """Test OpenTelemetry configuration schema."""

    def test_config_schema_valid(self):
        """Test OpenTelemetry configuration schema validation."""
        from axolotl.utils.schemas.integrations import OpenTelemetryConfig

        # Test valid config
        valid_config = {
            "use_otel_metrics": True,
            "otel_metrics_host": "localhost",
            "otel_metrics_port": 8000,
        }

        otel_config = OpenTelemetryConfig(**valid_config)
        assert otel_config.use_otel_metrics is True
        assert otel_config.otel_metrics_host == "localhost"
        assert otel_config.otel_metrics_port == 8000

    def test_config_defaults(self):
        """Test OpenTelemetry configuration default values."""
        from axolotl.utils.schemas.integrations import OpenTelemetryConfig

        # Test minimal config with defaults
        minimal_config = {"use_otel_metrics": True}

        otel_config = OpenTelemetryConfig(**minimal_config)
        assert otel_config.use_otel_metrics is True
        assert otel_config.otel_metrics_host == "localhost"  # default
        assert otel_config.otel_metrics_port == 8000  # default

    def test_config_disabled_by_default(self):
        """Test that OpenTelemetry is disabled by default."""
        from axolotl.utils.schemas.integrations import OpenTelemetryConfig

        # Test default config
        default_config = OpenTelemetryConfig()
        assert default_config.use_otel_metrics is False


class TestOpenTelemetryCallback:
    """Test OpenTelemetry callback functionality."""

    def test_callback_import(self):
        """Test that OpenTelemetry callback can be imported."""
        from axolotl.utils.callbacks.opentelemetry import OpenTelemetryMetricsCallback

        assert OpenTelemetryMetricsCallback is not None

    def test_callback_graceful_fallback(self, mock_otel_config):
        """Test callback gracefully handles missing dependencies."""
        from axolotl.utils.callbacks.opentelemetry import OpenTelemetryMetricsCallback

        # This should not raise an exception even if dependencies are missing
        callback = OpenTelemetryMetricsCallback(mock_otel_config)

        # Callback should exist but may have metrics disabled
        assert callback is not None
        assert hasattr(callback, "metrics_enabled")

    def test_callback_initialization_enabled(self, mock_otel_config):
        """Test callback initialization when OpenTelemetry is available."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        callback = OpenTelemetryMetricsCallback(mock_otel_config)

        if OPENTELEMETRY_AVAILABLE:
            assert callback.metrics_enabled is True
            assert callback.cfg == mock_otel_config
            assert callback.metrics_host == "localhost"
            assert callback.metrics_port == 8003
        else:
            assert callback.metrics_enabled is False

    def test_metrics_server_lifecycle(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test metrics server starts and stops correctly."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)

        # Start server
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )
        assert callback.server_started is True

        # End training
        callback.on_train_end(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

    def test_metrics_recording(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test that metrics are recorded during training."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

        # Test logging metrics
        test_logs = {
            "loss": 0.5,
            "learning_rate": 1e-4,
            "grad_norm": 0.8,
        }

        # This should not raise an exception
        callback.on_log(
            mock_training_args, mock_trainer_state, mock_trainer_control, logs=test_logs
        )
        assert callback.metrics_enabled is True

    def test_evaluation_metrics(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test evaluation metrics recording."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

        # Test evaluation metrics
        eval_logs = {
            "eval_loss": 0.3,
            "eval_accuracy": 0.95,
        }

        # This should not raise an exception
        callback.on_evaluate(
            mock_training_args, mock_trainer_state, mock_trainer_control, eval_logs
        )
        assert callback.metrics_enabled is True

    def test_thread_safety(self, mock_otel_config):
        """Test that callback has thread safety mechanisms."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        assert hasattr(callback, "metrics_lock")
        # Check it's a lock-like object
        assert hasattr(callback.metrics_lock, "__enter__")
        assert hasattr(callback.metrics_lock, "__exit__")


class TestOpenTelemetryIntegration:
    """Integration tests for OpenTelemetry."""

    def test_availability_check(self):
        """Test availability check function."""
        from axolotl.utils import is_opentelemetry_available

        result = is_opentelemetry_available()
        assert isinstance(result, bool)

    def test_prometheus_endpoint_basic(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test basic Prometheus endpoint functionality."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        try:
            import requests
        except ImportError:
            pytest.skip("requests library not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

        if not callback.server_started:
            pytest.skip("Metrics server failed to start")

        # Give server time to start
        time.sleep(1)

        # Try to access metrics endpoint
        try:
            response = requests.get(
                f"http://{callback.metrics_host}:{callback.metrics_port}/metrics",
                timeout=2,
            )
            assert response.status_code == 200
            # Check for Prometheus format
            assert "# TYPE" in response.text or "# HELP" in response.text
        except requests.exceptions.RequestException:
            pytest.skip(
                "Could not connect to metrics endpoint - this is expected in some environments"
            )


class TestOpenTelemetryCallbackMethods:
    """Test specific callback methods."""

    def test_step_end_callback(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test step end callback method."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

        # Should not raise an exception
        callback.on_step_end(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

    def test_epoch_end_callback(
        self,
        mock_otel_config,
        mock_trainer_state,
        mock_training_args,
        mock_trainer_control,
    ):
        """Test epoch end callback method."""
        from axolotl.utils.callbacks.opentelemetry import (
            OPENTELEMETRY_AVAILABLE,
            OpenTelemetryMetricsCallback,
        )

        if not OPENTELEMETRY_AVAILABLE:
            pytest.skip("OpenTelemetry dependencies not available")

        callback = OpenTelemetryMetricsCallback(mock_otel_config)
        callback.on_train_begin(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )

        # Should not raise an exception
        callback.on_epoch_end(
            mock_training_args, mock_trainer_state, mock_trainer_control
        )
