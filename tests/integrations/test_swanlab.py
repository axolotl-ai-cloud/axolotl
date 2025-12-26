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
import time
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


class TestSwanLabInitKwargs:
    """Tests for SwanLab initialization with direct parameter passing."""

    def test_custom_branding_added_to_config(self):
        """Test that Axolotl custom branding is added to SwanLab config."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        cfg = DictDefault({
            "use_swanlab": True,
            "swanlab_project": "test-project",
        })

        init_kwargs = plugin._get_swanlab_init_kwargs(cfg)

        # Verify custom branding is present
        assert "config" in init_kwargs
        assert init_kwargs["config"]["UPPERFRAME"] == "🦎 Axolotl"

    def test_api_key_passed_directly(self):
        """Test that API key is passed directly to swanlab.init() instead of via env var."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        cfg = DictDefault({
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_api_key": "test-api-key-12345",
        })

        init_kwargs = plugin._get_swanlab_init_kwargs(cfg)

        # Verify API key is in init_kwargs (not set as env var)
        assert "api_key" in init_kwargs
        assert init_kwargs["api_key"] == "test-api-key-12345"

    def test_private_deployment_hosts_passed_directly(self):
        """Test that private deployment hosts are passed directly to swanlab.init()."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        cfg = DictDefault({
            "use_swanlab": True,
            "swanlab_project": "internal-project",
            "swanlab_web_host": "https://swanlab.company.com",
            "swanlab_api_host": "https://api-swanlab.company.com",
        })

        init_kwargs = plugin._get_swanlab_init_kwargs(cfg)

        # Verify private deployment hosts are in init_kwargs
        assert "web_host" in init_kwargs
        assert init_kwargs["web_host"] == "https://swanlab.company.com"
        assert "api_host" in init_kwargs
        assert init_kwargs["api_host"] == "https://api-swanlab.company.com"

    @patch("axolotl.utils.distributed.is_main_process")
    def test_full_private_deployment_init(self, mock_is_main_process):
        """Test complete initialization with private deployment configuration."""
        mock_is_main_process.return_value = True

        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        cfg = DictDefault({
            "use_swanlab": True,
            "swanlab_project": "secure-project",
            "swanlab_experiment_name": "experiment-001",
            "swanlab_mode": "cloud",
            "swanlab_api_key": "private-key-xyz",
            "swanlab_web_host": "https://swanlab.internal.net",
            "swanlab_api_host": "https://api.swanlab.internal.net",
            "swanlab_workspace": "research-team",
        })

        with patch("swanlab.init") as mock_init:
            plugin.pre_model_load(cfg)

            # Verify swanlab.init was called with all parameters
            mock_init.assert_called_once()
            call_kwargs = mock_init.call_args[1]

            assert call_kwargs["project"] == "secure-project"
            assert call_kwargs["experiment_name"] == "experiment-001"
            assert call_kwargs["mode"] == "cloud"
            assert call_kwargs["api_key"] == "private-key-xyz"
            assert call_kwargs["web_host"] == "https://swanlab.internal.net"
            assert call_kwargs["api_host"] == "https://api.swanlab.internal.net"
            assert call_kwargs["workspace"] == "research-team"
            assert call_kwargs["config"]["UPPERFRAME"] == "🦎 Axolotl"

    def test_env_vars_not_set_for_api_params(self):
        """Test that environment variables are NOT set for API parameters."""
        import os
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        # Clear any existing env vars
        for key in ["SWANLAB_API_KEY", "SWANLAB_WEB_HOST", "SWANLAB_API_HOST", "SWANLAB_MODE"]:
            os.environ.pop(key, None)

        plugin = SwanLabPlugin()
        cfg = DictDefault({
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_api_key": "test-key",
            "swanlab_web_host": "https://test.com",
            "swanlab_api_host": "https://api-test.com",
            "swanlab_mode": "cloud",
        })

        with patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("swanlab.init"):
            plugin.pre_model_load(cfg)

        # Verify env vars were NOT set (simplified approach)
        # The old _setup_swanlab_env() method is removed, so these shouldn't be set
        # Note: SwanLab itself might set these, but our plugin shouldn't
        # We're just testing that our plugin doesn't call _setup_swanlab_env()


class TestLarkNotificationIntegration:
    """Tests for Lark (Feishu) notification integration."""

    def test_lark_callback_registration_with_webhook_only(self):
        """Test Lark callback registration with webhook URL only (no secret)."""
        plugin = SwanLabPlugin()

        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "local"
        cfg.swanlab_lark_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook"
        cfg.swanlab_lark_secret = None

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("swanlab.register_callbacks") as mock_register, \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            # Mock LarkCallback import
            with patch("swanlab.plugin.notification.LarkCallback") as MockLarkCallback:
                mock_lark_instance = MagicMock()
                MockLarkCallback.return_value = mock_lark_instance

                plugin.pre_model_load(cfg)

                # Verify LarkCallback was instantiated with correct params
                MockLarkCallback.assert_called_once_with(
                    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook",
                    secret=None,
                )

                # Verify callback was registered
                mock_register.assert_called_once_with([mock_lark_instance])

    def test_lark_callback_registration_with_secret(self):
        """Test Lark callback registration with webhook URL and HMAC secret."""
        plugin = SwanLabPlugin()

        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "local"
        cfg.swanlab_lark_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook"
        cfg.swanlab_lark_secret = "test-hmac-secret"

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("swanlab.register_callbacks") as mock_register, \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            with patch("swanlab.plugin.notification.LarkCallback") as MockLarkCallback:
                mock_lark_instance = MagicMock()
                MockLarkCallback.return_value = mock_lark_instance

                plugin.pre_model_load(cfg)

                # Verify LarkCallback was instantiated with secret
                MockLarkCallback.assert_called_once_with(
                    webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook",
                    secret="test-hmac-secret",
                )

                mock_register.assert_called_once_with([mock_lark_instance])

    def test_lark_callback_not_registered_without_webhook(self):
        """Test that Lark callback is NOT registered when webhook URL not provided."""
        plugin = SwanLabPlugin()

        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "local"
        cfg.swanlab_lark_webhook_url = None  # No webhook
        cfg.swanlab_lark_secret = None

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("swanlab.register_callbacks") as mock_register, \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            plugin.pre_model_load(cfg)

            # Verify register_callbacks was NOT called
            mock_register.assert_not_called()

    def test_lark_import_error_handled_gracefully(self, caplog):
        """Test that ImportError for Lark plugin is handled gracefully."""
        plugin = SwanLabPlugin()

        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "local"
        cfg.swanlab_lark_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook"
        cfg.swanlab_lark_secret = None

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            # Mock ImportError for LarkCallback
            with patch("swanlab.plugin.notification.LarkCallback", side_effect=ImportError("No module named 'swanlab.plugin.notification'")):
                with caplog.at_level(logging.WARNING):
                    plugin.pre_model_load(cfg)

                    # Should log warning about missing Lark plugin
                    warning_messages = [record.message for record in caplog.records]
                    assert any("Failed to import SwanLab Lark plugin" in msg for msg in warning_messages)
                    assert any("SwanLab >= 0.3.0" in msg for msg in warning_messages)

    def test_lark_warning_for_missing_secret(self, caplog):
        """Test that warning is logged when Lark webhook has no HMAC secret."""
        plugin = SwanLabPlugin()

        cfg = MagicMock()
        cfg.use_swanlab = True
        cfg.swanlab_project = "test-project"
        cfg.swanlab_mode = "local"
        cfg.swanlab_lark_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/test-webhook"
        cfg.swanlab_lark_secret = None  # No secret

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("swanlab.register_callbacks"), \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            with patch("swanlab.plugin.notification.LarkCallback"):
                with caplog.at_level(logging.WARNING):
                    plugin.pre_model_load(cfg)

                    # Should log warning about missing secret
                    warning_messages = [record.message for record in caplog.records]
                    assert any("no secret configured" in msg.lower() for msg in warning_messages)
                    assert any("swanlab_lark_secret" in msg for msg in warning_messages)


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
        cfg_obj.swanlab_lark_webhook_url = None  # No Lark

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

    def test_full_lifecycle_with_lark_notifications(self):
        """Test full lifecycle including Lark notification registration."""
        plugin = SwanLabPlugin()

        # Register
        cfg_dict = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_mode": "cloud",
        }
        plugin.register(cfg_dict)

        # Pre-model load with Lark config
        cfg_obj = MagicMock()
        cfg_obj.use_swanlab = True
        cfg_obj.swanlab_project = "test-project"
        cfg_obj.swanlab_mode = "cloud"
        cfg_obj.swanlab_lark_webhook_url = "https://open.feishu.cn/open-apis/bot/v2/hook/test"
        cfg_obj.swanlab_lark_secret = "secret123"

        with patch("swanlab.init"), \
             patch("swanlab.__version__", "0.3.0"), \
             patch("swanlab.register_callbacks") as mock_register, \
             patch("axolotl.utils.distributed.is_main_process", return_value=True), \
             patch("axolotl.utils.distributed.get_world_size", return_value=1):

            with patch("swanlab.plugin.notification.LarkCallback") as MockLarkCallback:
                mock_lark_instance = MagicMock()
                MockLarkCallback.return_value = mock_lark_instance

                plugin.pre_model_load(cfg_obj)

                # Verify both SwanLab init AND Lark callback registration
                MockLarkCallback.assert_called_once()
                mock_register.assert_called_once_with([mock_lark_instance])


class TestCompletionLogger:
    """Tests for CompletionLogger utility class."""

    def test_completion_logger_initialization(self):
        """Test CompletionLogger initializes with correct maxlen."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=64)
        assert logger.maxlen == 64
        assert len(logger) == 0

    def test_add_dpo_completion(self):
        """Test adding DPO completions to buffer."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)

        logger.add_dpo_completion(
            step=0,
            prompt="What is AI?",
            chosen="Artificial Intelligence is...",
            rejected="AI means...",
            reward_diff=0.5,
        )

        assert len(logger) == 1
        entry = logger.data[0]
        assert entry["step"] == 0
        assert entry["prompt"] == "What is AI?"
        assert entry["chosen"] == "Artificial Intelligence is..."
        assert entry["rejected"] == "AI means..."
        assert entry["reward_diff"] == 0.5

    def test_add_kto_completion(self):
        """Test adding KTO completions to buffer."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)

        logger.add_kto_completion(
            step=1,
            prompt="Explain quantum physics",
            completion="Quantum physics is...",
            label=True,
            reward=0.8,
        )

        assert len(logger) == 1
        entry = logger.data[0]
        assert entry["step"] == 1
        assert entry["prompt"] == "Explain quantum physics"
        assert entry["completion"] == "Quantum physics is..."
        assert entry["label"] == "desirable"
        assert entry["reward"] == 0.8

    def test_add_orpo_completion(self):
        """Test adding ORPO completions to buffer."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)

        logger.add_orpo_completion(
            step=2,
            prompt="Write a poem",
            chosen="Roses are red...",
            rejected="Violets are blue...",
            log_odds_ratio=1.2,
        )

        assert len(logger) == 1
        entry = logger.data[0]
        assert entry["step"] == 2
        assert entry["chosen"] == "Roses are red..."
        assert entry["rejected"] == "Violets are blue..."
        assert entry["log_odds_ratio"] == 1.2

    def test_add_grpo_completion(self):
        """Test adding GRPO completions to buffer."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)

        logger.add_grpo_completion(
            step=3,
            prompt="Solve this problem",
            completion="The answer is 42",
            reward=0.9,
            advantage=0.3,
        )

        assert len(logger) == 1
        entry = logger.data[0]
        assert entry["step"] == 3
        assert entry["completion"] == "The answer is 42"
        assert entry["reward"] == 0.9
        assert entry["advantage"] == 0.3

    def test_memory_bounded_buffer(self):
        """Test that buffer respects maxlen and drops oldest entries."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=3)

        # Add 5 completions
        for i in range(5):
            logger.add_dpo_completion(
                step=i,
                prompt=f"Prompt {i}",
                chosen=f"Chosen {i}",
                rejected=f"Rejected {i}",
            )

        # Should only keep last 3
        assert len(logger) == 3
        assert logger.data[0]["step"] == 2  # Oldest kept
        assert logger.data[1]["step"] == 3
        assert logger.data[2]["step"] == 4  # Newest

    def test_log_to_swanlab_when_not_initialized(self):
        """Test logging gracefully fails when SwanLab not initialized."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)
        logger.add_dpo_completion(
            step=0,
            prompt="Test",
            chosen="A",
            rejected="B",
        )

        with patch("swanlab.get_run", return_value=None):
            result = logger.log_to_swanlab()
            assert result is False  # Should fail gracefully

    def test_log_to_swanlab_success(self):
        """Test successful logging to SwanLab."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)
        logger.add_dpo_completion(
            step=0,
            prompt="Test prompt",
            chosen="Chosen response",
            rejected="Rejected response",
            reward_diff=0.5,
        )

        with patch("swanlab.get_run") as mock_get_run, \
             patch("swanlab.log") as mock_log, \
             patch("swanlab.echarts.Table") as MockTable:

            mock_get_run.return_value = MagicMock()  # SwanLab initialized
            mock_table_instance = MagicMock()
            MockTable.return_value = mock_table_instance

            result = logger.log_to_swanlab(table_name="test_table")

            assert result is True
            mock_log.assert_called_once()
            mock_table_instance.add.assert_called_once()

    def test_clear_buffer(self):
        """Test clearing the completion buffer."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=10)
        logger.add_dpo_completion(
            step=0,
            prompt="Test",
            chosen="A",
            rejected="B",
        )

        assert len(logger) == 1
        logger.clear()
        assert len(logger) == 0

    def test_repr(self):
        """Test string representation."""
        from axolotl.integrations.swanlab.completion_logger import CompletionLogger

        logger = CompletionLogger(maxlen=128)
        logger.add_dpo_completion(
            step=0,
            prompt="Test",
            chosen="A",
            rejected="B",
        )

        repr_str = repr(logger)
        assert "CompletionLogger" in repr_str
        assert "maxlen=128" in repr_str
        assert "buffered=1/128" in repr_str


class TestSwanLabRLHFCompletionCallback:
    """Tests for SwanLabRLHFCompletionCallback."""

    def test_callback_initialization(self):
        """Test callback initializes with correct parameters."""
        from axolotl.integrations.swanlab.callbacks import SwanLabRLHFCompletionCallback

        callback = SwanLabRLHFCompletionCallback(
            log_interval=50,
            max_completions=64,
            table_name="custom_table",
        )

        assert callback.log_interval == 50
        assert callback.logger.maxlen == 64
        assert callback.table_name == "custom_table"
        assert callback.trainer_type is None

    def test_trainer_type_detection_dpo(self):
        """Test DPO trainer type is detected correctly."""
        from axolotl.integrations.swanlab.callbacks import SwanLabRLHFCompletionCallback

        callback = SwanLabRLHFCompletionCallback()

        # Mock trainer with DPO in name
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "AxolotlDPOTrainer"

        callback.on_init_end(
            args=MagicMock(),
            state=MagicMock(),
            control=MagicMock(),
            trainer=mock_trainer,
        )

        assert callback.trainer_type == "dpo"

    def test_trainer_type_detection_kto(self):
        """Test KTO trainer type is detected correctly."""
        from axolotl.integrations.swanlab.callbacks import SwanLabRLHFCompletionCallback

        callback = SwanLabRLHFCompletionCallback()

        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "AxolotlKTOTrainer"

        callback.on_init_end(
            args=MagicMock(),
            state=MagicMock(),
            control=MagicMock(),
            trainer=mock_trainer,
        )

        assert callback.trainer_type == "kto"

    def test_on_train_end_logs_completions(self):
        """Test that completions are logged at end of training."""
        from axolotl.integrations.swanlab.callbacks import SwanLabRLHFCompletionCallback

        callback = SwanLabRLHFCompletionCallback()
        callback.trainer_type = "dpo"

        # Add some completions to buffer
        callback.logger.add_dpo_completion(
            step=0,
            prompt="Test",
            chosen="A",
            rejected="B",
        )

        with patch.object(callback.logger, "log_to_swanlab") as mock_log:
            callback.on_train_end(
                args=MagicMock(),
                state=MagicMock(global_step=100),
                control=MagicMock(),
            )

            # Should log remaining completions
            mock_log.assert_called_once()


class TestSwanLabPluginCompletionIntegration:
    """Integration tests for completion logging in SwanLabPlugin."""

    def test_completion_callback_registered_for_dpo_trainer(self):
        """Test that completion callback is registered for DPO trainer."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        plugin.swanlab_initialized = True  # Simulate SwanLab initialized

        cfg = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_log_completions": True,
            "swanlab_completion_log_interval": 50,
            "swanlab_completion_max_buffer": 64,
        }
        cfg_obj = DictDefault(cfg)

        # Mock DPO trainer
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "AxolotlDPOTrainer"
        mock_trainer.state = MagicMock(max_steps=1000)
        mock_trainer.args = MagicMock(
            num_train_epochs=3,
            train_batch_size=4,
            gradient_accumulation_steps=2,
        )

        with patch("swanlab.config.update"):
            plugin.post_trainer_create(cfg_obj, mock_trainer)

        # Verify callback was added
        mock_trainer.add_callback.assert_called_once()
        callback = mock_trainer.add_callback.call_args[0][0]
        assert callback.__class__.__name__ == "SwanLabRLHFCompletionCallback"
        assert callback.log_interval == 50
        assert callback.logger.maxlen == 64

    def test_completion_callback_not_registered_for_non_rlhf_trainer(self):
        """Test that completion callback is NOT registered for non-RLHF trainers."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        plugin.swanlab_initialized = True

        cfg = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_log_completions": True,
        }
        cfg_obj = DictDefault(cfg)

        # Mock regular SFT trainer (not RLHF)
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "AxolotlTrainer"  # Not RLHF
        mock_trainer.state = MagicMock(max_steps=1000)
        mock_trainer.args = MagicMock()

        with patch("swanlab.config.update"):
            plugin.post_trainer_create(cfg_obj, mock_trainer)

        # Callback should NOT be added for non-RLHF trainer
        mock_trainer.add_callback.assert_not_called()

    def test_completion_callback_not_registered_when_disabled(self):
        """Test that completion callback is not registered when disabled in config."""
        from axolotl.integrations.swanlab.plugins import SwanLabPlugin
        from axolotl.utils.dict import DictDefault

        plugin = SwanLabPlugin()
        plugin.swanlab_initialized = True

        cfg = {
            "use_swanlab": True,
            "swanlab_project": "test-project",
            "swanlab_log_completions": False,  # Disabled
        }
        cfg_obj = DictDefault(cfg)

        # Mock DPO trainer
        mock_trainer = MagicMock()
        mock_trainer.__class__.__name__ = "AxolotlDPOTrainer"
        mock_trainer.state = MagicMock(max_steps=1000)
        mock_trainer.args = MagicMock()

        with patch("swanlab.config.update"):
            plugin.post_trainer_create(cfg_obj, mock_trainer)

        # Callback should NOT be added when disabled
        mock_trainer.add_callback.assert_not_called()


class TestSwanLabProfiling:
    """Tests for SwanLab profiling utilities."""

    def test_profiling_context_logs_duration(self):
        """Test that profiling context logs execution duration."""
        from axolotl.integrations.swanlab.profiling import swanlab_profiling_context

        # Mock trainer with SwanLab enabled
        mock_trainer = MagicMock()
        mock_trainer.cfg = MagicMock(use_swanlab=True)
        mock_trainer.__class__.__name__ = "TestTrainer"

        with patch("swanlab.get_run") as mock_get_run, \
             patch("swanlab.log") as mock_log:

            mock_get_run.return_value = MagicMock()  # SwanLab initialized

            with swanlab_profiling_context(mock_trainer, "test_function"):
                time.sleep(0.01)  # Simulate work

            # Verify log was called with correct metric name
            mock_log.assert_called_once()
            logged_data = mock_log.call_args[0][0]
            assert "profiling/Time taken: TestTrainer.test_function" in logged_data
            # Duration should be > 0.01 seconds
            assert logged_data["profiling/Time taken: TestTrainer.test_function"] >= 0.01

    def test_profiling_context_skips_when_swanlab_disabled(self):
        """Test that profiling is skipped when SwanLab is disabled."""
        from axolotl.integrations.swanlab.profiling import swanlab_profiling_context

        mock_trainer = MagicMock()
        mock_trainer.cfg = MagicMock(use_swanlab=False)  # Disabled

        with patch("swanlab.log") as mock_log:
            with swanlab_profiling_context(mock_trainer, "test_function"):
                time.sleep(0.01)

            # Should NOT log when disabled
            mock_log.assert_not_called()

    def test_profiling_context_skips_when_swanlab_not_initialized(self):
        """Test that profiling is skipped when SwanLab not initialized."""
        from axolotl.integrations.swanlab.profiling import swanlab_profiling_context

        mock_trainer = MagicMock()
        mock_trainer.cfg = MagicMock(use_swanlab=True)

        with patch("swanlab.get_run", return_value=None), \
             patch("swanlab.log") as mock_log:

            with swanlab_profiling_context(mock_trainer, "test_function"):
                time.sleep(0.01)

            # Should NOT log when not initialized
            mock_log.assert_not_called()

    def test_profiling_decorator(self):
        """Test swanlab_profile decorator."""
        from axolotl.integrations.swanlab.profiling import swanlab_profile

        class MockTrainer:
            def __init__(self):
                self.cfg = MagicMock(use_swanlab=True)

            @swanlab_profile
            def expensive_method(self, x):
                time.sleep(0.01)
                return x * 2

        trainer = MockTrainer()

        with patch("swanlab.get_run") as mock_get_run, \
             patch("swanlab.log") as mock_log:

            mock_get_run.return_value = MagicMock()

            result = trainer.expensive_method(5)

            # Verify method still works correctly
            assert result == 10

            # Verify profiling was logged
            mock_log.assert_called_once()
            logged_data = mock_log.call_args[0][0]
            assert "profiling/Time taken: MockTrainer.expensive_method" in logged_data

    def test_profiling_config(self):
        """Test ProfilingConfig class."""
        from axolotl.integrations.swanlab.profiling import ProfilingConfig

        config = ProfilingConfig(
            enabled=True,
            min_duration_ms=1.0,
            log_interval=5,
        )

        # Test enabled check
        assert config.enabled is True

        # Test minimum duration filtering
        assert config.should_log("func1", 0.0001) is False  # 0.1ms < 1.0ms threshold
        assert config.should_log("func2", 0.002) is True  # 2.0ms > 1.0ms threshold

        # Test log interval
        assert config.should_log("func3", 0.002) is True  # 1st call
        assert config.should_log("func3", 0.002) is False  # 2nd call
        assert config.should_log("func3", 0.002) is False  # 3rd call
        assert config.should_log("func3", 0.002) is False  # 4th call
        assert config.should_log("func3", 0.002) is True  # 5th call (interval=5)

    def test_profiling_config_when_disabled(self):
        """Test ProfilingConfig when disabled."""
        from axolotl.integrations.swanlab.profiling import ProfilingConfig

        config = ProfilingConfig(enabled=False)

        # Should never log when disabled
        assert config.should_log("func1", 100.0) is False

    def test_profiling_context_advanced(self):
        """Test advanced profiling context with custom config."""
        from axolotl.integrations.swanlab.profiling import (
            ProfilingConfig,
            swanlab_profiling_context_advanced,
        )

        mock_trainer = MagicMock()
        mock_trainer.cfg = MagicMock(use_swanlab=True)
        mock_trainer.__class__.__name__ = "TestTrainer"

        # Config that filters out very fast operations
        config = ProfilingConfig(min_duration_ms=10.0)  # 10ms minimum

        with patch("swanlab.get_run") as mock_get_run, \
             patch("swanlab.log") as mock_log:

            mock_get_run.return_value = MagicMock()

            # Fast operation (< 10ms) - should NOT log
            with swanlab_profiling_context_advanced(mock_trainer, "fast_op", config):
                time.sleep(0.001)  # 1ms

            mock_log.assert_not_called()

            # Slow operation (> 10ms) - should log
            with swanlab_profiling_context_advanced(mock_trainer, "slow_op", config):
                time.sleep(0.015)  # 15ms

            mock_log.assert_called_once()

    def test_profiling_with_exception(self):
        """Test that profiling still logs even when exception occurs."""
        from axolotl.integrations.swanlab.profiling import swanlab_profiling_context

        mock_trainer = MagicMock()
        mock_trainer.cfg = MagicMock(use_swanlab=True)
        mock_trainer.__class__.__name__ = "TestTrainer"

        with patch("swanlab.get_run") as mock_get_run, \
             patch("swanlab.log") as mock_log:

            mock_get_run.return_value = MagicMock()

            try:
                with swanlab_profiling_context(mock_trainer, "error_function"):
                    time.sleep(0.01)
                    raise ValueError("Test error")
            except ValueError:
                pass  # Expected

            # Should still log duration even with exception
            mock_log.assert_called_once()
