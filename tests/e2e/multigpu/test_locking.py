"""Tests for FileLockLoader class."""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from axolotl.utils.data.lock import FileLockLoader
from axolotl.utils.dict import DictDefault


class TestFileLockLoader:
    """Class with tests for FileLockLoader."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    @pytest.fixture
    def cfg(self, temp_dir):
        """Create a test configuration."""
        return DictDefault({"dataset_prepared_path": str(temp_dir)})

    @pytest.fixture
    def loader(self, cfg):
        """Create a FileLockLoader instance for testing."""
        return FileLockLoader(cfg)

    def test_load_first_process(self, loader):
        """Test load() when no ready flag exists (first process)."""
        mock_load_fn = Mock(return_value="test_data")

        result = loader.load(mock_load_fn)

        # Should call the load function
        mock_load_fn.assert_called_once()
        assert result == "test_data"

        # Should create the ready flag
        assert loader.ready_flag_path.exists()

    def test_load_subsequent_process(self, loader):
        """Test load() when ready flag already exists (subsequent process)."""
        # Create ready flag first
        loader.ready_flag_path.touch()

        mock_load_fn = Mock(return_value="loaded_data")

        result = loader.load(mock_load_fn)

        # Should still call load function (to load the prepared data)
        mock_load_fn.assert_called_once()
        assert result == "loaded_data"

    def test_load_concurrent_processes(self, cfg):
        """Test that concurrent processes coordinate correctly."""
        results = []
        call_count = 0

        def slow_load_fn():
            nonlocal call_count
            call_count += 1
            time.sleep(0.1)  # Simulate slow loading
            return f"data_{call_count}"

        def worker():
            loader = FileLockLoader(cfg)
            result = loader.load(slow_load_fn)
            results.append(result)

        # Start multiple threads simultaneously
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Only one thread should have done the initial loading
        # All should return data, but the load function should be called
        # once by the first process and once by each subsequent process
        assert len(results) == 3
        assert all(result.startswith("data_") for result in results)

    @patch("time.sleep")
    def test_load_waiting_for_ready_flag(self, mock_sleep, loader):
        """Test that processes wait for the ready flag to appear."""
        mock_load_fn = Mock(return_value="waiting_data")
        mock_ready_flag_path = Mock()
        exists_call_count = 0

        def mock_exists():
            nonlocal exists_call_count
            exists_call_count += 1

            if exists_call_count == 1:
                # First check: ready flag exists (not first process)
                return True
            if exists_call_count <= 3:
                # While loop checks: flag doesn't exist yet
                return False
            return True

        mock_ready_flag_path.exists.side_effect = mock_exists

        # Replace the ready_flag_path with our mock
        original_path = loader.ready_flag_path
        loader.ready_flag_path = mock_ready_flag_path

        try:
            result = loader.load(mock_load_fn)
        finally:
            # Restore original path
            loader.ready_flag_path = original_path

        # Should have slept twice while waiting
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1)

        # Should eventually call load function
        mock_load_fn.assert_called_once()
        assert result == "waiting_data"

    def test_complete_workflow_with_cleanup(self, loader):
        """Test the complete load -> cleanup workflow."""
        mock_load_fn = Mock(return_value="test_data")

        # First process calls load (this should set up counter)
        result = loader.load(mock_load_fn)
        assert result == "test_data"
        assert loader.ready_flag_path.exists()
        assert loader.counter_path.exists()

        # Cleanup should remove everything since there's only one process
        loader.cleanup()
        assert not loader.ready_flag_path.exists()
        assert not loader.counter_path.exists()

    def test_multiple_processes_workflow(self, loader):
        """Test workflow with multiple processes."""
        # Simulate multiple processes by manually setting up counter
        loader.ready_flag_path.touch()
        loader.counter_path.write_text("3")  # 3 processes

        # First process cleanup
        loader.cleanup()
        assert loader.ready_flag_path.exists()
        assert loader.counter_path.read_text().strip() == "2"

        # Second process cleanup
        loader.cleanup()
        assert loader.ready_flag_path.exists()
        assert loader.counter_path.read_text().strip() == "1"

        # Last process cleanup
        loader.cleanup()
        assert not loader.ready_flag_path.exists()
        assert not loader.counter_path.exists()

    def test_load_exception_handling(self, loader):
        """Test behavior when load_fn raises an exception."""

        def failing_load_fn():
            raise ValueError("Load failed")

        with pytest.raises(ValueError, match="Load failed"):
            loader.load(failing_load_fn)

        # Ready flag should not be created on failure
        assert not loader.ready_flag_path.exists()

    def test_file_lock_called(self, loader):
        """Test that FileLock is properly used."""
        mock_load_fn = Mock(return_value="locked_data")

        with patch("axolotl.utils.data.lock.FileLock") as mock_filelock:
            mock_context = MagicMock()
            mock_filelock.return_value.__enter__ = Mock(return_value=mock_context)
            mock_filelock.return_value.__exit__ = Mock(return_value=None)

            loader.load(mock_load_fn)

            # Verify FileLock was called with correct path
            mock_filelock.assert_called_once_with(str(loader.lock_file_path))

            # Verify context manager was used
            mock_filelock.return_value.__enter__.assert_called_once()
            mock_filelock.return_value.__exit__.assert_called_once()
