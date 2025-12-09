"""Tests for telemetry callback module."""

# pylint: disable=redefined-outer-name

import time
from unittest.mock import MagicMock, patch

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from axolotl.telemetry.callbacks import TIME_SINCE_LAST, TelemetryCallback


def calc_expected_metrics(step, last_step, current_time, last_time, start_time=900.0):
    """Calculate expected metrics values for tests"""
    time_diff = current_time - last_time
    step_diff = step - last_step
    return {
        "steps_per_second": (
            step_diff / time_diff if time_diff > 0 and step_diff > 0 else 0
        ),
        "time_since_last_report": time_diff,
        "elapsed_time": current_time - start_time,
    }


@pytest.fixture
def mock_time():
    """Mock time.time() to have predictable values in tests"""
    with patch("axolotl.telemetry.callbacks.time") as mock_time:
        mock_time.time.return_value = 1000.0
        yield mock_time


@pytest.fixture
def mock_telemetry_manager():
    """Create a mock TelemetryManager"""
    with patch("axolotl.telemetry.callbacks.TelemetryManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager_class.get_instance.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def mock_runtime_metrics_tracker():
    """Create a mock RuntimeMetricsTracker"""
    with patch(
        "axolotl.telemetry.callbacks.RuntimeMetricsTracker"
    ) as mock_tracker_class:
        mock_tracker = MagicMock()
        # Set up metrics property on the tracker
        mock_metrics = MagicMock()
        mock_metrics.to_dict.return_value = {
            "total_steps": 100,
            "peak_cpu_memory_bytes": 1024,
        }
        mock_tracker.metrics = mock_metrics

        # Make the constructor return our mock
        mock_tracker_class.return_value = mock_tracker
        yield mock_tracker


@pytest.fixture
def training_args():
    """Create a minimal TrainingArguments instance"""
    return TrainingArguments(output_dir="./output")


@pytest.fixture
def trainer_state():
    """Create a mock TrainerState"""
    state = MagicMock(spec=TrainerState)
    state.global_step = 10
    state.epoch = 0.5  # halfway through first epoch
    state.log_history = [{"loss": 2.5, "learning_rate": 5e-5}]
    return state


@pytest.fixture
def trainer_control():
    """Create a mock TrainerControl"""
    return MagicMock(spec=TrainerControl)


# pylint: disable=unused-argument
@pytest.fixture
def callback(mock_telemetry_manager, mock_runtime_metrics_tracker):
    """Create a TelemetryCallback instance with mocked dependencies"""
    return TelemetryCallback()


class TestTelemetryCallback:
    """Tests for the TelemetryCallback class."""

    def test_initialization(self, callback, mock_runtime_metrics_tracker):
        """Test callback initialization."""
        assert callback.current_epoch == -1
        assert callback.tracker == mock_runtime_metrics_tracker
        assert callback.last_report_step == 0
        assert hasattr(callback, "start_time")
        assert hasattr(callback, "last_report_time")
        assert callback.report_interval_steps == 100

    def test_on_train_begin(
        self,
        callback,
        mock_telemetry_manager,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_train_begin sends expected event."""
        callback.on_train_begin(training_args, trainer_state, trainer_control)

        mock_telemetry_manager.send_event.assert_called_once_with(
            event_type="train-start"
        )

    def test_on_train_end(
        self,
        callback,
        mock_telemetry_manager,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_train_end sends expected event with metrics."""
        callback.on_train_end(training_args, trainer_state, trainer_control)

        mock_telemetry_manager.send_event.assert_called_once()
        call_args = mock_telemetry_manager.send_event.call_args[1]

        assert call_args["event_type"] == "train-end"
        assert "loss" in call_args["properties"]
        assert call_args["properties"]["loss"] == 2.5
        assert "learning_rate" in call_args["properties"]
        assert call_args["properties"]["learning_rate"] == 5e-5

        # Check that metrics from RuntimeMetricsTracker are included
        assert "total_steps" in call_args["properties"]
        assert call_args["properties"]["total_steps"] == 100
        assert "peak_cpu_memory_bytes" in call_args["properties"]
        assert call_args["properties"]["peak_cpu_memory_bytes"] == 1024

    def test_on_epoch_begin(
        self,
        callback,
        mock_runtime_metrics_tracker,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_epoch_begin updates epoch counter and calls tracker."""
        initial_epoch = callback.current_epoch

        callback.on_epoch_begin(training_args, trainer_state, trainer_control)

        assert callback.current_epoch == initial_epoch + 1
        mock_runtime_metrics_tracker.start_epoch.assert_called_once_with(
            initial_epoch + 1
        )

    def test_on_epoch_end(
        self,
        callback,
        mock_runtime_metrics_tracker,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_epoch_end calls tracker."""
        # Set current epoch
        callback.current_epoch = 2

        callback.on_epoch_end(training_args, trainer_state, trainer_control)

        mock_runtime_metrics_tracker.end_epoch.assert_called_once_with(2)

    def test_on_step_end_no_report(
        self,
        callback,
        mock_telemetry_manager,
        mock_runtime_metrics_tracker,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_step_end updates tracker but doesn't report if criteria not met."""
        # Set up state to avoid reporting
        trainer_state.global_step = 42  # Not divisible by report_interval_steps
        callback.last_report_step = 41  # Just 1 step since last report
        callback.last_report_time = time.time()  # Just now

        callback.on_step_end(training_args, trainer_state, trainer_control)

        # Should update tracker
        mock_runtime_metrics_tracker.update_step.assert_called_once_with(42)

        # Should not send telemetry
        mock_telemetry_manager.send_event.assert_not_called()

        # Should not update last report time/step
        assert callback.last_report_step == 41

    def test_on_step_end_report_interval_steps(
        self,
        callback,
        mock_telemetry_manager,
        mock_runtime_metrics_tracker,
        mock_time,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_step_end reports when step interval is reached."""
        # Set up state with clear values
        current_step = 100  # Exactly matches report_interval_steps
        last_step = 0
        start_time = 900.0
        current_time = 1000.0
        time_diff = current_time - start_time  # 100 seconds

        # Configure state and callback
        trainer_state.global_step = current_step
        callback.report_interval_steps = 100
        callback.last_report_step = last_step
        callback.start_time = start_time
        callback.last_report_time = start_time

        # Mock time.time() to return consistent values
        mock_time.time.return_value = current_time

        callback.on_step_end(training_args, trainer_state, trainer_control)

        # Should update tracker
        mock_runtime_metrics_tracker.update_step.assert_called_once_with(current_step)
        mock_runtime_metrics_tracker.update_memory_metrics.assert_called_once()

        # Should send telemetry
        mock_telemetry_manager.send_event.assert_called_once()
        call_args = mock_telemetry_manager.send_event.call_args[1]
        assert call_args["event_type"] == "train-progress"

        # Properties should include expected values
        props = call_args["properties"]
        assert props["step"] == current_step
        assert props["elapsed_time"] == time_diff  # 1000 - 900 = 100
        assert props["time_since_last_report"] == time_diff  # 1000 - 900 = 100
        assert props["steps_per_second"] == 1.0  # 100 steps / 100 seconds

        # Should update last report time/step
        assert callback.last_report_step == current_step
        assert callback.last_report_time == current_time

    def test_on_step_end_report_time_elapsed(
        self,
        callback,
        mock_telemetry_manager,
        mock_runtime_metrics_tracker,  # pylint: disable=unused-argument
        mock_time,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_step_end reports when enough time has elapsed."""
        # Set up state with clear values
        current_step = 120
        last_step = 10
        start_time = 900.0
        current_time = 1000.0
        time_diff = TIME_SINCE_LAST + 1  # Just over the threshold

        # Configure state and callback
        trainer_state.global_step = current_step
        callback.report_interval_steps = 100
        callback.last_report_step = last_step
        callback.start_time = start_time
        callback.last_report_time = current_time - time_diff

        # Mock time.time() to return consistent values
        mock_time.time.return_value = current_time

        callback.on_step_end(training_args, trainer_state, trainer_control)

        # Should send telemetry
        mock_telemetry_manager.send_event.assert_called_once()

        # Properties should include expected values
        props = mock_telemetry_manager.send_event.call_args[1]["properties"]
        expected_metrics = calc_expected_metrics(
            current_step, last_step, current_time, current_time - time_diff, start_time
        )
        assert props["steps_per_second"] == expected_metrics["steps_per_second"]
        assert (
            props["time_since_last_report"]
            == expected_metrics["time_since_last_report"]
        )

    def test_on_step_end_first_step(
        self,
        callback,
        mock_telemetry_manager,
        mock_runtime_metrics_tracker,  # pylint: disable=unused-argument
        mock_time,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test on_step_end always reports on first step."""
        # Set up state with clear values
        current_step = 1  # First step
        last_step = 0
        start_time = 900.0
        current_time = 1000.0
        last_report_time = 999.0  # Just 1 second ago

        # Configure state and callback
        trainer_state.global_step = current_step
        callback.report_interval_steps = 100
        callback.last_report_step = last_step
        callback.start_time = start_time
        callback.last_report_time = last_report_time

        # Mock time.time() to return consistent values
        mock_time.time.return_value = current_time

        callback.on_step_end(training_args, trainer_state, trainer_control)

        # Should send telemetry even though not much time has passed
        mock_telemetry_manager.send_event.assert_called_once()

        # Properties should include expected values for first step
        props = mock_telemetry_manager.send_event.call_args[1]["properties"]
        assert props["step"] == current_step
        expected_metrics = calc_expected_metrics(
            current_step, last_step, current_time, last_report_time, start_time
        )
        assert props["steps_per_second"] == expected_metrics["steps_per_second"]

    def test_log_history_empty(
        self,
        callback,
        mock_telemetry_manager,
        mock_runtime_metrics_tracker,  # pylint: disable=unused-argument
        mock_time,
        training_args,
        trainer_state,
        trainer_control,
    ):
        """Test handling of empty log history."""
        # Set up state with clear values
        current_step = 1
        start_time = 900.0
        current_time = 1000.0

        # Configure state and callback
        trainer_state.global_step = current_step
        trainer_state.log_history = []
        callback.start_time = start_time

        # Mock time.time() to return consistent values
        mock_time.time.return_value = current_time

        callback.on_step_end(training_args, trainer_state, trainer_control)

        # Should still send telemetry
        mock_telemetry_manager.send_event.assert_called_once()

        # Properties should have default values for missing log data
        props = mock_telemetry_manager.send_event.call_args[1]["properties"]
        assert props["loss"] == 0
        assert props["learning_rate"] == 0
