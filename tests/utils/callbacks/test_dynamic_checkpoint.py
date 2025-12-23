"""Unit tests for dynamic checkpoint callback"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from axolotl.utils.callbacks.dynamic_checkpoint import (
    DEFAULT_TRIGGER_FILENAME,
    DynamicCheckpointCallback,
)
from axolotl.utils.dict import DictDefault


class TestDynamicCheckpointCallbackInit:
    """Test callback initialization"""

    def test_callback_disabled_by_default(self):
        """Test that callback is disabled when config.enabled=False"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": False},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)
            assert callback.enabled is False

    def test_callback_disabled_when_none(self):
        """Test that callback is disabled when dynamic_checkpoint is None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": None,
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)
            assert callback.enabled is False

    def test_callback_enabled_when_configured(self):
        """Test that callback is enabled when config.enabled=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 10},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)
            assert callback.enabled is True
            assert callback.check_interval == 10

    def test_default_trigger_filename(self):
        """Test that default trigger filename is used"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 10},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)
            assert callback.trigger_filename == DEFAULT_TRIGGER_FILENAME

    def test_check_interval_default(self):
        """Test default check interval"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)
            assert callback.check_interval == 100  # Default from schema


class TestDynamicCheckpointFileDetection:
    """Test file-based checkpoint triggering"""

    def test_trigger_file_detected_and_deleted(self):
        """Test that trigger file is detected and deleted"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 1},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()
            assert trigger_file.exists()

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=False,
                ):
                    result = callback.on_step_end(args, state, control)

            assert not trigger_file.exists()
            assert result.should_save is True

    def test_check_interval_honored(self):
        """Test that file is only checked at check_interval steps"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 10},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            args = Mock(output_dir=tmpdir)
            control = Mock(should_save=False)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()

            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=False,
                ):
                    # Step 5 - shouldn't check (not divisible by 10)
                    state = Mock(global_step=5)
                    result = callback.on_step_end(args, state, control)
                    assert trigger_file.exists()  # Still there
                    assert result.should_save is False

                    # Step 10 - should check
                    state = Mock(global_step=10)
                    result = callback.on_step_end(args, state, control)
                    assert not trigger_file.exists()  # Deleted
                    assert result.should_save is True

    def test_no_file_no_trigger(self):
        """Test that no trigger occurs when file doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 1},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=False,
                ):
                    result = callback.on_step_end(args, state, control)

            assert result.should_save is False

    def test_file_deletion_error_handling(self):
        """Test that file deletion errors are handled gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 1},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=False,
                ):
                    with patch.object(
                        Path, "unlink", side_effect=OSError("Permission denied")
                    ):
                        result = callback.on_step_end(args, state, control)

            assert result.should_save is True


class TestDynamicCheckpointMultiGPU:
    """Test multi-GPU synchronization"""

    def test_only_rank_0_checks_file(self):
        """Test that only rank 0 checks filesystem in multi-GPU setup"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 1},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            # Rank 1 (not main process) - shouldn't check file
            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=False,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=True,
                ):
                    with patch("torch.distributed.broadcast") as mock_broadcast:
                        with patch(
                            "axolotl.utils.callbacks.dynamic_checkpoint.barrier"
                        ):
                            mock_tensor = MagicMock()
                            mock_tensor.item.return_value = 0
                            with patch("torch.tensor", return_value=mock_tensor):
                                callback.on_step_end(args, state, control)

            assert trigger_file.exists()
            # Broadcast should have been called
            assert mock_broadcast.called

    def test_broadcast_synchronization(self):
        """Test that trigger decision is broadcasted to all ranks"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": True, "check_interval": 1},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            # Rank 0 detects file
            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=True,
                ):
                    with patch("torch.distributed.broadcast") as mock_broadcast:
                        with patch(
                            "axolotl.utils.callbacks.dynamic_checkpoint.barrier"
                        ) as mock_barrier:
                            mock_tensor = MagicMock()
                            mock_tensor.item.return_value = 1
                            with patch("torch.tensor", return_value=mock_tensor):
                                with patch("torch.cuda.current_device", return_value=0):
                                    result = callback.on_step_end(args, state, control)

            assert mock_broadcast.called
            assert mock_barrier.called
            # All ranks should trigger
            assert result.should_save is True


class TestDynamicCheckpointSignalHandling:
    """Test signal-based checkpoint triggering"""

    def test_signal_trigger_via_callback(self):
        """Test that signal flag triggers checkpoint save"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {
                        "enabled": True,
                        "check_interval": 1,
                        "enable_signal": True,
                    },
                    "output_dir": tmpdir,
                }
            )

            with patch("signal.signal"):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                    return_value=True,
                ):
                    with patch(
                        "axolotl.utils.callbacks.dynamic_checkpoint.hasattr",
                        return_value=True,
                    ):
                        callback = DynamicCheckpointCallback(cfg)

            callback.should_save_checkpoint = True

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            with patch(
                "axolotl.utils.callbacks.dynamic_checkpoint.is_main_process",
                return_value=True,
            ):
                with patch(
                    "axolotl.utils.callbacks.dynamic_checkpoint.is_distributed",
                    return_value=False,
                ):
                    result = callback.on_step_end(args, state, control)

            assert result.should_save is True
            assert callback.should_save_checkpoint is False

    def test_signal_not_registered_when_disabled(self):
        """Test that signal handler is not registered when disabled"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {
                        "enabled": True,
                        "check_interval": 10,
                        "enable_signal": False,
                    },
                    "output_dir": tmpdir,
                }
            )

            with patch("signal.signal") as mock_signal_register:
                _ = DynamicCheckpointCallback(cfg)

            assert not mock_signal_register.called


class TestDynamicCheckpointDisabled:
    """Test behavior when callback is disabled"""

    def test_disabled_callback_does_nothing(self):
        """Test that disabled callback doesn't check or trigger"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DictDefault(
                {
                    "dynamic_checkpoint": {"enabled": False},
                    "output_dir": tmpdir,
                }
            )
            callback = DynamicCheckpointCallback(cfg)

            trigger_file = Path(tmpdir) / DEFAULT_TRIGGER_FILENAME
            trigger_file.touch()

            args = Mock(output_dir=tmpdir)
            state = Mock(global_step=1)
            control = Mock(should_save=False)

            result = callback.on_step_end(args, state, control)

            assert trigger_file.exists()
            assert result.should_save is False
