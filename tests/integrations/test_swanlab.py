# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for SwanLab Integration Plugin.

Tests conflict detection, configuration validation, and multi-logger warnings.
"""

import logging
import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from axolotl.integrations.swanlab.args import SwanLabConfig
from axolotl.integrations.swanlab.plugins import SwanLabPlugin


class TestSwanLabConfigValidators:
    """Tests for Pydantic field validators in SwanLabConfig."""

    def test_valid_swanlab_mode_cloud(self):
        """Test that 'cloud' mode is valid."""
        config = SwanLabConfig(swanlab_mode="cloud")
        assert config.swanlab_mode == "cloud"

    def test_valid_swanlab_mode_local(self):
        """Test that 'local' mode is valid."""
        config = SwanLabConfig(swanlab_mode="local")
        assert config.swanlab_mode == "local"

    def test_valid_swanlab_mode_offline(self):
        """Test that 'offline' mode is valid."""
        config = SwanLabConfig(swanlab_mode="offline")
        assert config.swanlab_mode == "offline"

    def test_valid_swanlab_mode_disabled(self):
        """Test that 'disabled' mode is valid."""
        config = SwanLabConfig(swanlab_mode="disabled")
        assert config.swanlab_mode == "disabled"

    def test_invalid_swanlab_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValidationError) as exc_info:
            SwanLabConfig(swanlab_mode="invalid")

        error_msg = str(exc_info.value)
        assert "Invalid swanlab_mode" in error_msg
        assert "cloud" in error_msg
        assert "local" in error_msg
        assert "offline" in error_msg
        assert "disabled" in error_msg

    def test_swanlab_mode_none_allowed(self):
        """Test that None mode is allowed (will use default)."""
        config = SwanLabConfig(swanlab_mode=None)
        assert config.swanlab_mode is None

    def test_valid_swanlab_project(self):
        """Test that valid project name is accepted."""
        config = SwanLabConfig(swanlab_project="my-project")
        assert config.swanlab_project == "my-project"

    def test_swanlab_project_none_allowed(self):
        """Test that None project is allowed."""
        config = SwanLabConfig(swanlab_project=None)
        assert config.swanlab_project is None

    def test_empty_swanlab_project_rejected(self):
        """Test that empty string project name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SwanLabConfig(swanlab_project="")

        error_msg = str(exc_info.value)
        assert "cannot be an empty string" in error_msg

    def test_whitespace_only_project_rejected(self):
        """Test that whitespace-only project name is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SwanLabConfig(swanlab_project="   ")

        error_msg = str(exc_info.value)
        assert "cannot be an empty string" in error_msg

    def test_use_swanlab_true_requires_project(self):
        """Test that use_swanlab=True requires swanlab_project."""
        with pytest.raises(ValidationError) as exc_info:
            SwanLabConfig(use_swanlab=True, swanlab_project=None)

        error_msg = str(exc_info.value)
        assert "swanlab_project" in error_msg.lower()
        assert "not set" in error_msg.lower()

    def test_use_swanlab_true_with_project_valid(self):
        """Test that use_swanlab=True with project is valid."""
        config = SwanLabConfig(use_swanlab=True, swanlab_project="my-project")
        assert config.use_swanlab is True
        assert config.swanlab_project == "my-project"

    def test_use_swanlab_false_no_project_valid(self):
        """Test that use_swanlab=False without project is valid."""
        config = SwanLabConfig(use_swanlab=False, swanlab_project=None)
        assert config.use_swanlab is False
        assert config.swanlab_project is None

    def test_use_swanlab_none_no_project_valid(self):
        """Test that use_swanlab=None without project is valid."""
        config = SwanLabConfig(use_swanlab=None, swanlab_project=None)
        assert config.use_swanlab is None
        assert config.swanlab_project is None


class TestSwanLabPluginRegister:
    """Tests for SwanLabPlugin.register() conflict detection."""

    def test_register_without_use_swanlab(self):
        """Test that register works when SwanLab is not enabled."""
        plugin = SwanLabPlugin()
        cfg = {"use_swanlab": False}
        # Should not raise
        plugin.register(cfg)

    def test_register_use_swanlab_missing_project(self):
        """Test that use_swanlab=True without project raises ValueError."""
        plugin = SwanLabPlugin()
        cfg = {"use_swanlab": True}

        with pytest.raises(ValueError) as exc_info:
            plugin.register(cfg)

        error_msg = str(exc_info.value)
        assert "swanlab_project" in error_msg
        assert "not set" in error_msg
        assert "Solutions" in error_msg

    def test_register_use_swanlab_with_project_valid(self):
        """Test that use_swanlab=True with project is valid."""
        plugin = SwanLabPlugin()
        cfg = {"use_swanlab": True, "swanlab_project": "my-project"}
        # Should not raise
        plugin.register(cfg)

    def test_register_invalid_mode(self):
        """Test that invalid swanlab_mode raises ValueError."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "swanlab_mode": "invalid-mode",
        }

        with pytest.raises(ValueError) as exc_info:
            plugin.register(cfg)

        error_msg = str(exc_info.value)
        assert "Invalid swanlab_mode" in error_msg
        assert "cloud" in error_msg
        assert "local" in error_msg

    def test_register_valid_modes(self):
        """Test that all valid modes are accepted."""
        plugin = SwanLabPlugin()
        valid_modes = ["cloud", "local", "offline", "disabled"]

        for mode in valid_modes:
            cfg = {
                "use_swanlab": True,
                "swanlab_project": "my-project",
                "swanlab_mode": mode,
            }
            # Should not raise
            plugin.register(cfg)

    def test_register_auto_enable_swanlab(self):
        """Test that providing swanlab_project auto-enables use_swanlab."""
        plugin = SwanLabPlugin()
        cfg = {"swanlab_project": "my-project"}

        plugin.register(cfg)

        assert cfg["use_swanlab"] is True

    def test_register_cloud_mode_without_api_key_warns(self, caplog):
        """Test that cloud mode without API key logs warning."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "swanlab_mode": "cloud",
        }

        # Clear environment variable to ensure it's not set
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                plugin.register(cfg)

            # Should log warning about missing API key
            warning_messages = [record.message for record in caplog.records]
            assert any("API key" in msg for msg in warning_messages)


class TestMultiLoggerDetection:
    """Tests for multi-logger conflict detection."""

    def test_single_logger_no_warning(self, caplog):
        """Test that single logger doesn't trigger warning."""
        plugin = SwanLabPlugin()
        cfg = {"use_swanlab": True, "swanlab_project": "my-project"}

        with caplog.at_level(logging.WARNING):
            plugin.register(cfg)

        # Should not log multi-logger warning
        warning_messages = [record.message for record in caplog.records]
        assert not any("Multiple logging tools" in msg for msg in warning_messages)

    def test_two_loggers_warning(self, caplog):
        """Test that two loggers trigger warning."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "use_wandb": True,
        }

        with caplog.at_level(logging.WARNING):
            plugin.register(cfg)

        # Should log multi-logger warning
        warning_messages = [record.message for record in caplog.records]
        assert any("Multiple logging tools" in msg for msg in warning_messages)
        assert any("SwanLab" in msg and "WandB" in msg for msg in warning_messages)

    def test_three_loggers_error(self, caplog):
        """Test that three loggers trigger error-level warning."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "use_wandb": True,
            "use_mlflow": True,
        }

        with caplog.at_level(logging.ERROR):
            plugin.register(cfg)

        # Should log error-level warning
        error_messages = [
            record.message for record in caplog.records if record.levelno >= logging.ERROR
        ]
        assert any("logging tools enabled" in msg for msg in error_messages)

    def test_multi_logger_with_comet(self, caplog):
        """Test that Comet is detected in multi-logger scenario."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "comet_api_key": "test-key",
        }

        with caplog.at_level(logging.WARNING):
            plugin.register(cfg)

        # Should detect Comet
        warning_messages = [record.message for record in caplog.records]
        assert any("Comet" in msg for msg in warning_messages)

    def test_multi_logger_with_comet_project(self, caplog):
        """Test that Comet is detected via comet_project_name."""
        plugin = SwanLabPlugin()
        cfg = {
            "use_swanlab": True,
            "swanlab_project": "my-project",
            "comet_project_name": "test-project",
        }

        with caplog.at_level(logging.WARNING):
            plugin.register(cfg)

        # Should detect Comet
        warning_messages = [record.message for record in caplog.records]
        assert any("Comet" in msg for msg in warning_messages)


class TestSwanLabPluginPreModelLoad:
    """Tests for SwanLabPlugin.pre_model_load() runtime checks."""

    def test_pre_model_load_disabled(self):
        """Test that pre_model_load does nothing when SwanLab is disabled."""
        plugin = SwanLabPlugin()
        cfg = MagicMock()
        cfg.use_swanlab = False

        # Should not raise
        plugin.pre_model_load(cfg)

    def test_pre_model_load_import_error(self):
        """Test that missing swanlab package raises clear ImportError."""
        plugin = SwanLabPlugin()
        cfg = MagicMock()
        cfg.use_swanlab = True

        with patch("builtins.__import__", side_effect=ImportError("No module named 'swanlab'")):
            with pytest.raises(ImportError) as exc_info:
                plugin.pre_model_load(cfg)

            error_msg = str(exc_info.value)
            assert "SwanLab is not installed" in error_msg
            assert "pip install swanlab" in error_msg

    @patch("axolotl.utils.distributed.is_main_process")
    @patch("axolotl.utils.distributed.get_world_size")
    def test_pre_model_load_non_main_process_skips(
        self, mock_get_world_size, mock_is_main_process
    ):
        """Test that non-main process skips SwanLab initialization."""
        mock_get_world_size.return_value = 2
        mock_is_main_process.return_value = False

        plugin = SwanLabPlugin()
        cfg = MagicMock()
        cfg.use_swanlab = True

        with patch("swanlab.init") as mock_init:
            plugin.pre_model_load(cfg)
            # Should NOT call swanlab.init
            mock_init.assert_not_called()

    @patch("axolotl.utils.distributed.is_main_process")
    @patch("axolotl.utils.distributed.get_world_size")
    def test_pre_model_load_distributed_logging(
        self, mock_get_world_size, mock_is_main_process, caplog
    ):
        """Test that distributed training logs world size info."""
        mock_get_world_size.return_value = 4
        mock_is_main_process.return_value = True

        plugin = SwanLabPlugin()
        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "cloud"

        with patch("swanlab.init"), patch("swanlab.__version__", "0.3.0"):
            with caplog.at_level(logging.INFO):
                plugin.pre_model_load(cfg)

            # Should log distributed training info
            info_messages = [record.message for record in caplog.records]
            assert any("world_size=4" in msg for msg in info_messages)
            assert any("Only rank 0" in msg for msg in info_messages)


class TestSwanLabPluginIntegration:
    """Integration tests for SwanLab plugin lifecycle."""

    def test_full_lifecycle_valid_config(self):
        """Test full plugin lifecycle with valid configuration."""
        plugin = SwanLabPlugin()

        # Register
        cfg_dict = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_mode": "local",
        }
        plugin.register(cfg_dict)

        # Pre-model load (mock SwanLab)
        cfg_obj = MagicMock()
        cfg_obj.use_swanlab = True
        cfg_obj.swanlab_project = "test-project"
        cfg_obj.swanlab_mode = "local"

        with patch("swanlab.init") as mock_init, patch("swanlab.__version__", "0.3.0"), \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):
            plugin.pre_model_load(cfg_obj)
            # Should call swanlab.init
            mock_init.assert_called_once()

    def test_lifecycle_with_multi_logger_warning(self, caplog):
        """Test lifecycle with multi-logger warning."""
        plugin = SwanLabPlugin()

        cfg_dict = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "use_wandb": True,
        }

        with caplog.at_level(logging.WARNING):
            plugin.register(cfg_dict)

        # Should have multi-logger warning
        warning_messages = [record.message for record in caplog.records]
        assert any("Multiple logging tools" in msg for msg in warning_messages)

    def test_lifecycle_invalid_config_fails_early(self):
        """Test that invalid config fails at register stage."""
        plugin = SwanLabPlugin()

        cfg_dict = {
            "use_swanlab": True,
            # Missing swanlab_project
        }

        # Should fail at register, not pre_model_load
        with pytest.raises(ValueError):
            plugin.register(cfg_dict)
