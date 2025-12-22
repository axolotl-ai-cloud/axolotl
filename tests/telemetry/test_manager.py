"""Tests for TelemetryManager class and utilities"""

# pylint: disable=redefined-outer-name,protected-access

import os
from unittest.mock import patch

import pytest
import yaml

from axolotl.telemetry.manager import TelemetryManager


@pytest.fixture
def mock_whitelist(tmp_path):
    """Create a temporary whitelist file for testing"""
    whitelist_content = {
        "organizations": ["meta-llama", "mistralai"],
    }
    whitelist_file = tmp_path / "whitelist.yaml"
    with open(whitelist_file, "w", encoding="utf-8") as f:
        yaml.dump(whitelist_content, f)

    return str(whitelist_file)


@pytest.fixture
def telemetry_manager_class():
    """Reset the TelemetryManager singleton between tests"""
    original_instance = TelemetryManager._instance
    original_initialized = TelemetryManager._initialized
    TelemetryManager._instance = None
    TelemetryManager._initialized = False
    yield TelemetryManager
    TelemetryManager._instance = original_instance
    TelemetryManager._initialized = original_initialized


@pytest.fixture
def manager(telemetry_manager_class, mock_whitelist):
    """Create a TelemetryManager instance with mocked dependencies"""
    with (
        patch("posthog.capture"),
        patch("posthog.flush"),
        patch("time.sleep"),
        patch("axolotl.telemetry.manager.WHITELIST_PATH", mock_whitelist),
        patch.dict(os.environ, {"RANK": "0"}),
    ):
        manager = telemetry_manager_class()
        # Manually enable for most tests
        manager.enabled = True
        return manager


def test_singleton_instance(telemetry_manager_class):
    """Test that TelemetryManager is a singleton"""
    with (
        patch("posthog.capture"),
        patch("time.sleep"),
        patch.dict(os.environ, {"RANK": "0"}),
    ):
        first = telemetry_manager_class()
        second = telemetry_manager_class()
        assert first is second
        assert telemetry_manager_class.get_instance() is first


def test_telemetry_enabled_by_default(telemetry_manager_class):
    """Test that telemetry is enabled by default (opt-out)"""
    with (
        patch.dict(os.environ, {"RANK": "0"}, clear=True),
        patch("time.sleep"),
        patch("logging.Logger.info"),
    ):
        manager = telemetry_manager_class()
        assert manager.enabled


def test_telemetry_enabled_with_explicit_opt_in(telemetry_manager_class):
    """Test that telemetry is enabled when AXOLOTL_DO_NOT_TRACK=0"""
    with (
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0", "RANK": "0"}),
        patch("time.sleep"),
    ):
        manager = telemetry_manager_class()
        assert manager.enabled


def test_telemetry_disabled_with_axolotl_do_not_track(telemetry_manager_class):
    """Test that telemetry is disabled when AXOLOTL_DO_NOT_TRACK=1"""
    with (
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "1", "RANK": "0"}),
        patch("time.sleep"),
    ):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_telemetry_disabled_with_do_not_track(telemetry_manager_class):
    """Test that telemetry is disabled when DO_NOT_TRACK=1"""
    with (
        patch.dict(
            os.environ, {"AXOLOTL_DO_NOT_TRACK": "0", "DO_NOT_TRACK": "1", "RANK": "0"}
        ),
        patch("time.sleep"),
    ):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_telemetry_disabled_for_non_main_process(telemetry_manager_class):
    """Test that telemetry is disabled for non-main processes"""
    with (
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0", "RANK": "1"}),
        patch("time.sleep"),
    ):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_opt_in_info_displayed(telemetry_manager_class):
    """Test that opt-in info is displayed when telemetry is not configured"""
    with (
        patch.dict(os.environ, {"RANK": "0"}, clear=True),
        patch("logging.Logger.warning") as mock_warning,
        patch("time.sleep"),
    ):
        telemetry_manager_class()
        assert any(
            "Telemetry is now enabled by default" in str(call)
            for call in mock_warning.call_args_list
        )


def test_is_whitelisted(telemetry_manager_class, mock_whitelist):
    """Test org whitelist functionality"""
    with (
        patch("axolotl.telemetry.manager.WHITELIST_PATH", mock_whitelist),
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0"}),
    ):
        manager = telemetry_manager_class()

        # Should match organizations from the mock whitelist
        assert manager._is_whitelisted("meta-llama/llama-7b")
        assert manager._is_whitelisted("mistralai/mistral-7b-instruct")
        # Should not match
        assert not manager._is_whitelisted("unknown/model")
        # Should handle case insensitively
        assert manager._is_whitelisted("META-LLAMA/Llama-7B")
        # Should handle empty input
        assert not manager._is_whitelisted("")


def test_system_info_collection(manager):
    """Test system information collection"""
    system_info = manager._get_system_info()

    # Check essential keys
    assert "os" in system_info
    assert "python_version" in system_info
    assert "cpu_count" in system_info
    assert "memory_total" in system_info
    assert "accelerator_count" in system_info


def test_send_event(telemetry_manager_class):
    """Test basic event sending"""
    with (
        patch("posthog.capture") as mock_capture,
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0"}),
    ):
        manager = telemetry_manager_class()

        # Test with clean properties (no PII)
        manager.send_event("test_event", {"key": "value"})
        assert mock_capture.called
        assert mock_capture.call_args[1]["event"] == "test_event"
        assert mock_capture.call_args[1]["properties"] == {"key": "value"}
        assert mock_capture.call_args[1]["distinct_id"] == manager.run_id

        # Test with default properties (None)
        mock_capture.reset_mock()
        manager.send_event("simple_event")
        assert mock_capture.called
        assert mock_capture.call_args[1]["properties"] == {}


def test_send_system_info(telemetry_manager_class):
    """Test sending system info"""
    with (
        patch("posthog.capture") as mock_capture,
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0"}),
    ):
        manager = telemetry_manager_class()
        manager.send_system_info()
        assert mock_capture.called
        assert mock_capture.call_args[1]["event"] == "system-info"
        assert mock_capture.call_args[1]["properties"] == manager.system_info


def test_redacted_properties(telemetry_manager_class):
    """Test path redaction in send_event method"""
    with (
        patch("posthog.capture") as mock_capture,
        patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0"}),
    ):
        manager = telemetry_manager_class()
        # Test with properties containing various paths and non-paths
        test_properties = {
            "filepath": "/home/user/sensitive/data.txt",
            "windows_path": "C:\\Users\\name\\Documents\\project\\file.py",
            "output_dir": "/var/lib/data",
            "path_to_model": "models/llama/7b",
            "message": "Training started",  # Should not be redacted
            "metrics": {"loss": 0.5, "accuracy": 0.95},  # Should not be redacted
            "base_model": "models/local_model",
            "nested": {
                "model_path": "/models/my_model",
                "root_dir": "/home/user/projects",
                "stats": {"steps": 1000, "epochs": 3},  # Should not be redacted
            },
        }

        manager.send_event("test_event", test_properties)

        # Verify the call was made
        assert mock_capture.called

        # Get the sanitized properties that were sent
        sanitized = mock_capture.call_args[1]["properties"]

        # Check that path-like and base_model keys were redacted
        assert sanitized["filepath"] == "[REDACTED]"
        assert sanitized["windows_path"] == "[REDACTED]"
        assert sanitized["path_to_model"] == "[REDACTED]"
        assert sanitized["base_model"] == "[REDACTED]"

        # Check that non-path values were preserved
        assert sanitized["message"] == "Training started"
        assert sanitized["metrics"] == {"loss": 0.5, "accuracy": 0.95}

        # Check nested structure handling
        assert sanitized["nested"]["model_path"] == "[REDACTED]"
        assert sanitized["nested"]["root_dir"] == "[REDACTED]"
        assert sanitized["nested"]["stats"] == {"steps": 1000, "epochs": 3}


def test_disable_telemetry(manager):
    """Test that disabled telemetry doesn't send events"""
    with patch("posthog.capture") as mock_capture:
        manager.enabled = False
        manager.send_event("test_event")
        assert not mock_capture.called


def test_exception_handling_during_send(manager):
    """Test that exceptions in PostHog are handled gracefully"""
    with (
        patch("posthog.capture", side_effect=Exception("Test error")),
        patch("logging.Logger.warning") as mock_warning,
    ):
        manager.send_event("test_event")
        warning_logged = False
        for call in mock_warning.call_args_list:
            if "Failed to send telemetry event" in str(call):
                warning_logged = True
                break
        assert warning_logged


def test_shutdown(manager):
    """Test shutdown behavior"""
    with patch("posthog.shutdown") as mock_shutdown:
        manager.shutdown()
        assert mock_shutdown.called
