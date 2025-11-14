"""Tests for runtime metrics telemetry module"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest

from axolotl.telemetry.runtime_metrics import RuntimeMetrics, RuntimeMetricsTracker


@pytest.fixture
def mock_time():
    """Mock time.time() to have predictable values in tests"""
    with patch("time.time") as mock_time:
        # Start with time 1000.0 and increment by 10 seconds on each call
        times = [1000.0 + i * 10 for i in range(10)]
        mock_time.side_effect = times
        yield mock_time


@pytest.fixture
def mock_telemetry_manager():
    """Create a mock TelemetryManager"""
    with patch(
        "axolotl.telemetry.runtime_metrics.TelemetryManager"
    ) as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.enabled = True
        mock_manager_class.get_instance.return_value = mock_manager
        yield mock_manager


@pytest.fixture
def mock_psutil():
    """Mock psutil for memory information"""
    with patch("axolotl.telemetry.runtime_metrics.psutil") as mock_psutil:
        mock_process = MagicMock()
        mock_memory_info = MagicMock()
        # Set initial memory to 1GB
        mock_memory_info.rss = 1024 * 1024 * 1024
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        yield mock_psutil


@pytest.fixture
def mock_torch():
    """Mock torch.cuda functions"""
    with patch("axolotl.telemetry.runtime_metrics.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2

        # Mock memory allocated per device (1GB for device 0, 2GB for device 1)
        mock_torch.cuda.memory_allocated.side_effect = (
            lambda device: (device + 1) * 1024 * 1024 * 1024
        )

        yield mock_torch


class TestRuntimeMetrics:
    """Tests for RuntimeMetrics class."""

    def test_initialization(self):
        """Test RuntimeMetrics initialization."""
        metrics = RuntimeMetrics(start_time=1000.0)

        assert metrics.start_time == 1000.0
        assert metrics.epoch_start_times == {}
        assert metrics.epoch_end_times == {}
        assert metrics.peak_gpu_memory == {}
        assert metrics.total_steps == 0
        assert metrics.current_epoch == 0
        assert metrics.current_step == 0
        assert metrics.peak_cpu_memory == 0

    def test_elapsed_time(self, mock_time):
        """Test elapsed_time property."""
        metrics = RuntimeMetrics(start_time=1000.0)

        # Mock time.time() to return 1050.0
        mock_time.side_effect = [1050.0]

        assert metrics.elapsed_time == 50.0

    def test_epoch_time(self):
        """Test epoch_time method."""
        metrics = RuntimeMetrics(start_time=1000.0)

        # No epoch data
        assert metrics.epoch_time(0) is None

        # Add epoch start but no end
        metrics.epoch_start_times[0] = 1000.0
        assert metrics.epoch_time(0) is None

        # Add epoch end
        metrics.epoch_end_times[0] = 1060.0
        assert metrics.epoch_time(0) == 60.0

    def test_average_epoch_time(self):
        """Test average_epoch_time method."""
        metrics = RuntimeMetrics(start_time=1000.0)

        # No completed epochs
        assert metrics.average_epoch_time() is None

        # Add one completed epoch
        metrics.epoch_start_times[0] = 1000.0
        metrics.epoch_end_times[0] = 1060.0
        assert metrics.average_epoch_time() == 60.0

        # Add second completed epoch
        metrics.epoch_start_times[1] = 1060.0
        metrics.epoch_end_times[1] = 1140.0  # 80 seconds
        assert metrics.average_epoch_time() == 70.0  # Average of 60 and 80

        # Add incomplete epoch (should not affect average)
        metrics.epoch_start_times[2] = 1140.0
        assert metrics.average_epoch_time() == 70.0

    def test_steps_per_second(self, mock_time):
        """Test steps_per_second method."""
        metrics = RuntimeMetrics(start_time=1000.0)

        # No steps - first call to time.time()
        mock_time.side_effect = None
        mock_time.return_value = 1050.0
        assert metrics.steps_per_second() is None

        # Add steps - second call to time.time()
        metrics.total_steps = 100
        mock_time.return_value = 1050.0  # Keep same time for consistent result
        assert metrics.steps_per_second() == 2.0  # 100 steps / 50 seconds

    def test_to_dict_basic(self, mock_time):
        """Test to_dict method with basic metrics."""
        metrics = RuntimeMetrics(start_time=1000.0)
        metrics.total_steps = 100
        metrics.peak_cpu_memory = 2 * 1024 * 1024 * 1024  # 2GB

        # Mock elapsed_time
        mock_time.side_effect = None
        mock_time.return_value = 1050.0

        result = metrics.to_dict()

        assert result["total_time_seconds"] == 50.0
        assert result["total_steps"] == 100
        assert result["steps_per_second"] == 2.0
        assert result["epochs_completed"] == 0
        assert result["peak_cpu_memory_bytes"] == 2 * 1024 * 1024 * 1024
        assert "epoch_times" not in result
        assert "gpu_memory" not in result

    def test_to_dict_with_epochs(self, mock_time):
        """Test to_dict method with epoch data."""
        metrics = RuntimeMetrics(start_time=1000.0)
        metrics.total_steps = 100

        # Add epoch data
        metrics.epoch_start_times[0] = 1000.0
        metrics.epoch_end_times[0] = 1060.0
        metrics.epoch_start_times[1] = 1060.0
        metrics.epoch_end_times[1] = 1140.0

        # Mock elapsed_time
        mock_time.side_effect = None
        mock_time.return_value = 1150.0

        result = metrics.to_dict()

        assert "epoch_times" in result
        assert result["epoch_times"]["epoch_0_seconds"] == 60.0
        assert result["epoch_times"]["epoch_1_seconds"] == 80.0
        assert result["average_epoch_time_seconds"] == 70.0

    def test_to_dict_with_gpu_memory(self, mock_time):
        """Test to_dict method with GPU memory data."""
        metrics = RuntimeMetrics(start_time=1000.0)
        metrics.peak_gpu_memory = {
            0: 1 * 1024 * 1024 * 1024,  # 1GB
            1: 2 * 1024 * 1024 * 1024,  # 2GB
        }

        # Mock elapsed_time
        mock_time.side_effect = [1050.0]

        result = metrics.to_dict()

        assert "gpu_memory" in result
        assert result["gpu_memory"]["gpu_0_peak_memory_bytes"] == 1 * 1024 * 1024 * 1024
        assert result["gpu_memory"]["gpu_1_peak_memory_bytes"] == 2 * 1024 * 1024 * 1024


class TestRuntimeMetricsTracker:
    """Tests for RuntimeMetricsTracker class."""

    # pylint: disable=unused-argument
    def test_initialization(self, mock_time, mock_telemetry_manager):
        """Test RuntimeMetricsTracker initialization."""
        tracker = RuntimeMetricsTracker()

        assert isinstance(tracker.metrics, RuntimeMetrics)
        assert tracker.metrics.start_time == 1000.0  # First value from mock_time

    # pylint: disable=unused-argument
    def test_start_epoch(
        self, mock_time, mock_psutil, mock_torch, mock_telemetry_manager
    ):
        """Test start_epoch method."""
        tracker = RuntimeMetricsTracker()

        # Reset mock_time to control next value
        mock_time.side_effect = [1010.0]

        tracker.start_epoch(0)

        assert tracker.metrics.current_epoch == 0
        assert tracker.metrics.epoch_start_times[0] == 1010.0

        # Verify memory metrics were updated
        assert tracker.metrics.peak_cpu_memory == 1 * 1024 * 1024 * 1024
        assert 0 in tracker.metrics.peak_gpu_memory
        assert 1 in tracker.metrics.peak_gpu_memory

    # pylint: disable=unused-argument
    def test_end_epoch(self, mock_time, mock_telemetry_manager):
        """Test end_epoch method."""
        tracker = RuntimeMetricsTracker()

        # Start epoch 0
        mock_time.side_effect = [1010.0]
        tracker.start_epoch(0)

        # End epoch 0
        mock_time.side_effect = [1060.0]
        tracker.end_epoch(0)

        assert 0 in tracker.metrics.epoch_end_times
        assert tracker.metrics.epoch_end_times[0] == 1060.0

    # pylint: disable=unused-argument
    def test_update_step(
        self, mock_time, mock_psutil, mock_torch, mock_telemetry_manager
    ):
        """Test update_step method."""
        tracker = RuntimeMetricsTracker()

        # Update step to a non-multiple of 100
        tracker.update_step(42)

        assert tracker.metrics.current_step == 42
        assert tracker.metrics.total_steps == 1

        # Memory metrics should not be updated for non-multiple of 100
        assert tracker.metrics.peak_cpu_memory == 0

        # Update step to a multiple of 100
        tracker.update_step(100)

        assert tracker.metrics.current_step == 100
        assert tracker.metrics.total_steps == 2

        # Memory metrics should be updated for multiple of 100
        assert tracker.metrics.peak_cpu_memory == 1 * 1024 * 1024 * 1024

    # pylint: disable=unused-argument
    def test_update_memory_metrics(
        self, mock_psutil, mock_torch, mock_telemetry_manager
    ):
        """Test update_memory_metrics method."""
        tracker = RuntimeMetricsTracker()

        # Initial memory state
        assert tracker.metrics.peak_cpu_memory == 0
        assert tracker.metrics.peak_gpu_memory == {}

        # Update memory metrics
        tracker.update_memory_metrics()

        # Verify CPU memory
        assert tracker.metrics.peak_cpu_memory == 1 * 1024 * 1024 * 1024

        # Verify GPU memory
        assert tracker.metrics.peak_gpu_memory[0] == 1 * 1024 * 1024 * 1024
        assert tracker.metrics.peak_gpu_memory[1] == 2 * 1024 * 1024 * 1024

        # Change mocked memory values to be lower
        mock_process = mock_psutil.Process.return_value
        mock_memory_info = mock_process.memory_info.return_value
        mock_memory_info.rss = 0.5 * 1024 * 1024 * 1024  # 0.5GB

        mock_torch.cuda.memory_allocated.side_effect = (
            lambda device: (device + 0.5) * 1024 * 1024 * 1024
        )

        # Update memory metrics again
        tracker.update_memory_metrics()

        # Peak values should not decrease
        assert tracker.metrics.peak_cpu_memory == 1 * 1024 * 1024 * 1024
        assert tracker.metrics.peak_gpu_memory[0] == 1 * 1024 * 1024 * 1024
        assert tracker.metrics.peak_gpu_memory[1] == 2 * 1024 * 1024 * 1024

        # Change mocked memory values to be higher
        mock_memory_info.rss = 2 * 1024 * 1024 * 1024  # 2GB

        mock_torch.cuda.memory_allocated.side_effect = (
            lambda device: (device + 2) * 1024 * 1024 * 1024
        )

        # Update memory metrics again
        tracker.update_memory_metrics()

        # Peak values should increase
        assert tracker.metrics.peak_cpu_memory == 2 * 1024 * 1024 * 1024
        assert tracker.metrics.peak_gpu_memory[0] == 2 * 1024 * 1024 * 1024
        assert tracker.metrics.peak_gpu_memory[1] == 3 * 1024 * 1024 * 1024

    # pylint: disable=unused-argument
    def test_get_memory_metrics(self, mock_psutil, mock_torch, mock_telemetry_manager):
        """Test get_memory_metrics method."""
        tracker = RuntimeMetricsTracker()

        # Set peak memory values
        tracker.metrics.peak_cpu_memory = 2 * 1024 * 1024 * 1024
        tracker.metrics.peak_gpu_memory = {
            0: 3 * 1024 * 1024 * 1024,
            1: 4 * 1024 * 1024 * 1024,
        }

        # Get memory metrics
        memory_metrics = tracker.get_memory_metrics()

        # Verify CPU memory
        assert (
            memory_metrics["cpu_memory_bytes"] == 1 * 1024 * 1024 * 1024
        )  # Current value from mock
        assert (
            memory_metrics["peak_cpu_memory_bytes"] == 2 * 1024 * 1024 * 1024
        )  # Peak value we set

        # Verify GPU memory
        assert (
            memory_metrics["gpu_0_memory_bytes"] == 1 * 1024 * 1024 * 1024
        )  # Current value from mock
        assert (
            memory_metrics["gpu_0_peak_memory_bytes"] == 3 * 1024 * 1024 * 1024
        )  # Peak value we set
        assert (
            memory_metrics["gpu_1_memory_bytes"] == 2 * 1024 * 1024 * 1024
        )  # Current value from mock
        assert (
            memory_metrics["gpu_1_peak_memory_bytes"] == 4 * 1024 * 1024 * 1024
        )  # Peak value we set
