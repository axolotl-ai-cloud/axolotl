"""Tests for TelemetryManager class and utilities"""
# pylint: disable=redefined-outer-name,protected-access

import os
from unittest.mock import patch

import pytest
import yaml

from axolotl.telemetry.manager import TelemetryConfig, TelemetryManager


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
    with patch("posthog.capture"), patch("posthog.flush"), patch(
        "time.sleep"
    ), patch.object(TelemetryConfig, "whitelist_path", mock_whitelist), patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ):
        manager = telemetry_manager_class()
        # Manually enable for most tests
        manager.enabled = True
        return manager


def test_singleton_instance(telemetry_manager_class):
    """Test that TelemetryManager is a singleton"""
    with patch("posthog.capture"), patch("time.sleep"):
        first = telemetry_manager_class()
        second = telemetry_manager_class()
        assert first is second
        assert telemetry_manager_class.get_instance() is first


def test_telemetry_disabled_with_axolotl_do_not_track(telemetry_manager_class):
    """Test that telemetry is disabled when AXOLOTL_DO_NOT_TRACK=1"""
    with patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "1"}), patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_telemetry_disabled_with_do_not_track(telemetry_manager_class):
    """Test that telemetry is disabled when DO_NOT_TRACK=1"""
    with patch.dict(os.environ, {"DO_NOT_TRACK": "1"}), patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_telemetry_disabled_for_non_main_process(telemetry_manager_class):
    """Test that telemetry is disabled for non-main processes"""
    with patch("axolotl.telemetry.manager.is_main_process", return_value=False):
        manager = telemetry_manager_class()
        assert not manager.enabled


def test_telemetry_enabled_by_default(telemetry_manager_class):
    """Test that telemetry is enabled by default"""
    with patch.dict(os.environ, {}, clear=True), patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ), patch("time.sleep"), patch("logging.Logger.warning"):
        manager = telemetry_manager_class()
        assert manager.enabled
        assert not manager.explicit_enable


def test_explicit_enable_disables_warning(telemetry_manager_class):
    """Test that explicit enabling prevents warning"""
    with patch.dict(os.environ, {"AXOLOTL_DO_NOT_TRACK": "0"}), patch(
        "logging.Logger.warning"
    ) as mock_warning, patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ), patch(
        "time.sleep"
    ):
        manager = telemetry_manager_class()
        assert manager.enabled
        assert manager.explicit_enable
        for call in mock_warning.call_args_list:
            assert "Telemetry is enabled" not in str(call)


def test_warning_displayed_for_implicit_enable(telemetry_manager_class):
    """Test that warning is displayed when telemetry is implicitly enabled"""
    with patch.dict(os.environ, {}, clear=True), patch(
        "logging.Logger.warning"
    ) as mock_warning, patch(
        "axolotl.telemetry.manager.is_main_process", return_value=True
    ), patch(
        "time.sleep"
    ):
        manager = telemetry_manager_class()
        assert manager.enabled
        assert not manager.explicit_enable
        warning_displayed = False
        for call in mock_warning.call_args_list:
            if "Telemetry is enabled" in str(call):
                warning_displayed = True
                break
        assert warning_displayed


def test_is_whitelisted(manager, mock_whitelist):
    """Test org whitelist functionality"""
    with patch.object(TelemetryConfig, "whitelist_path", mock_whitelist):
        # Should match organizations from the mock whitelist
        assert manager._is_whitelisted("meta-llama/llama-7b")
        assert manager._is_whitelisted("mistralai/mistral-7b-instruct")
        # Should not match
        assert not manager._is_whitelisted("unknown/model")
        # Should handle case insensitively
        assert manager._is_whitelisted("META-LLAMA/Llama-7B")
        # Should handle empty input
        assert not manager._is_whitelisted("")
        assert not manager._is_whitelisted(None)


def test_system_info_collection(manager):
    """Test system information collection"""
    system_info = manager.system_info

    # Check essential keys
    assert "os" in system_info
    assert "python_version" in system_info
    assert "pytorch_version" in system_info
    assert "transformers_version" in system_info
    assert "axolotl_version" in system_info
    assert "cpu_count" in system_info
    assert "memory_total" in system_info
    assert "gpu_count" in system_info


def test_send_event(manager):
    """Test basic event sending"""
    with patch("posthog.capture") as mock_capture:
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


def test_send_system_info(manager):
    """Test sending system info"""
    with patch("posthog.capture") as mock_capture:
        manager.send_system_info()
        assert mock_capture.called
        assert mock_capture.call_args[1]["event"] == "system-info"
        assert mock_capture.call_args[1]["properties"] == manager.system_info


def test_sanitize_properties(manager):
    """Test property sanitization in send_event method"""
    with patch("posthog.capture") as mock_capture:
        # Test with properties containing various PII
        test_properties = {
            "filepath": "/home/user/sensitive/data.txt",
            "windows_path": "C:\\Users\\name\\Documents\\project\\file.py",
            "url": "https://example.com/private/user123",
            "message": "Error loading /tmp/axolotl/data.csv - check permissions",
            "cloud_path": "s3://my-bucket/data/user-files/",
            "nested": {
                "deep_path": "/var/log/axolotl/training.log",
                "list_paths": ["/home/user1/file1.txt", "/home/user2/file2.txt"],
            },
        }

        manager.send_event("test_event", test_properties)

        # Verify the call was made
        assert mock_capture.called

        # Get the sanitized properties that were sent
        sanitized = mock_capture.call_args[1]["properties"]

        # Check that PII was removed/sanitized
        assert "/home/user/sensitive" not in str(sanitized)
        assert "C:\\Users\\name" not in str(sanitized)
        assert "https://example.com/private" not in str(sanitized)
        assert "s3://my-bucket" not in str(sanitized)

        # Check that filenames were preserved
        assert "data.txt" in str(sanitized)
        assert "file.py" in str(sanitized)
        assert "data.csv" in str(sanitized)

        # Check that nested structures were properly sanitized
        assert "/var/log/axolotl" not in str(sanitized["nested"])
        assert "/home/user1" not in str(sanitized["nested"]["list_paths"])


def test_disable_telemetry(manager):
    """Test that disabled telemetry doesn't send events"""
    with patch("posthog.capture") as mock_capture:
        manager.enabled = False
        manager.send_event("test_event")
        assert not mock_capture.called


def test_exception_handling_during_send(manager):
    """Test that exceptions in PostHog are handled gracefully"""
    with patch("posthog.capture", side_effect=Exception("Test error")), patch(
        "logging.Logger.warning"
    ) as mock_warning:
        manager.send_event("test_event")
        warning_logged = False
        for call in mock_warning.call_args_list:
            if "Failed to send telemetry event" in str(call):
                warning_logged = True
                break
        assert warning_logged


def test_shutdown(manager):
    """Test shutdown behavior"""
    with patch("posthog.flush") as mock_flush:
        manager.shutdown()
        assert mock_flush.called
