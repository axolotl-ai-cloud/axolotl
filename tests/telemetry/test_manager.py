import os
from unittest.mock import patch

import pytest
import yaml

from axolotl.telemetry import TelemetryConfig, TelemetryManager, ModelConfig


@pytest.fixture
def mock_whitelist(tmp_path):
    """Create a temporary whitelist file for testing"""
    whitelist_content = {
        "organizations": ["meta", "mistral"],
        "models": ["llama", "mistral-7b"]
    }
    whitelist_file = tmp_path / "whitelist.yaml"
    with open(whitelist_file, "w") as f:
        yaml.dump(whitelist_content, f)
    return str(whitelist_file)


@pytest.fixture
def config(mock_whitelist):
    """Create a TelemetryConfig with test settings"""
    return TelemetryConfig(
        host="https://test.posthog.com",
        whitelist_path=mock_whitelist
    )


@pytest.fixture
def manager(config):
    """Create a TelemetryManager instance with mocked PostHog"""
    with patch("posthog.capture"):
        return TelemetryManager(config)


def test_telemetry_disabled_by_default():
    """Test that telemetry is disabled by default"""
    manager = TelemetryManager(TelemetryConfig())
    assert not manager.enabled


def test_telemetry_opt_in():
    """Test that telemetry can be enabled via environment variable"""
    with patch.dict(os.environ, {"AXOLOTL_TELEMETRY": "1"}):
        manager = TelemetryManager(TelemetryConfig())
        assert manager.enabled


def test_do_not_track_override():
    """Test that DO_NOT_TRACK overrides AXOLOTL_TELEMETRY"""
    with patch.dict(os.environ, {
        "AXOLOTL_TELEMETRY": "1",
        "DO_NOT_TRACK": "1"
    }):
        manager = TelemetryManager(TelemetryConfig())
        assert not manager.enabled


def test_whitelist_checking(manager):
    """Test model whitelist functionality"""
    with patch.dict(os.environ, {"AXOLOTL_TELEMETRY": "1"}):
        # Should match organization
        assert manager._is_whitelisted("meta/llama-7b")
        # Should match model name
        assert manager._is_whitelisted("mistral-7b-instruct")
        # Should not match
        assert not manager._is_whitelisted("unknown/model")
        # Should handle case insensitively
        assert manager._is_whitelisted("meta/Llama-7b")


def test_event_tracking(manager):
    """Test basic event tracking"""
    with patch("posthog.capture") as mock_capture:
        manager.enabled = True
        manager.track_event("test_event", {"key": "value"})
        
        assert mock_capture.called
        assert mock_capture.call_args[1]["event"] == "test_event"
        assert mock_capture.call_args[1]["properties"]["key"] == "value"
        assert "run_id" in mock_capture.call_args[1]["properties"]
        assert "system_info" in mock_capture.call_args[1]["properties"]


def test_model_tracking(manager):
    """Test model load tracking"""
    with patch.dict(os.environ, {"AXOLOTL_TELEMETRY": "1"}):
        model_config = ModelConfig(
            base_model="meta/llama-7b",
            model_type="decoder",
            hidden_size=4096,
            num_layers=32,
            num_attention_heads=32,
            tokenizer_config={},
            flash_attention=True,
            quantization_config=None,
            training_approach="lora"
        )
        
        with patch("posthog.capture") as mock_capture:
            manager.enabled = True
            manager.track_model_load(model_config)
            
            assert mock_capture.called
            assert mock_capture.call_args[1]["event"] == "model_load"
            assert mock_capture.call_args[1]["properties"]["model_config"] == model_config.to_dict()


def test_training_context(manager):
    """Test training context manager"""
    config = {"model": "llama", "batch_size": 8}
    
    with patch("posthog.capture") as mock_capture:
        manager.enabled = True
        
        with manager.track_training(config):
            pass  # Simulate successful training
            
        # Should have captured training_start and training_complete
        events = [call[1]["event"] for call in mock_capture.call_args_list]
        assert "training_start" in events
        assert "training_complete" in events


def test_training_error(manager):
    """Test training context manager with error"""
    config = {"model": "llama", "batch_size": 8}
    
    with patch("posthog.capture") as mock_capture:
        manager.enabled = True
        
        with pytest.raises(ValueError):
            with manager.track_training(config):
                raise ValueError("Test error")
            
        # Should have captured training_start and training_error
        events = [call[1]["event"] for call in mock_capture.call_args_list]
        assert "training_start" in events
        assert "training_error" in events


def test_path_sanitization(manager):
    """Test path sanitization"""
    path = "/home/user/sensitive/data.txt"
    sanitized = manager._sanitize_path(path)
    assert sanitized == "data.txt"
    assert "/home/user" not in sanitized


def test_error_sanitization(manager):
    """Test error message sanitization"""
    error = "Failed to load /home/user/sensitive/data.txt: File not found"
    sanitized = manager._sanitize_error(error)
    assert "sensitive" not in sanitized
    assert "/home/user" not in sanitized


def test_shutdown(manager):
    """Test shutdown behavior"""
    with patch("posthog.flush") as mock_flush:
        manager.enabled = True
        manager.shutdown()
        assert mock_flush.called
