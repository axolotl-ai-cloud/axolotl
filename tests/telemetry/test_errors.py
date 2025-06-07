"""Tests for telemetry error utilities"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest

from axolotl.telemetry.errors import sanitize_stack_trace, send_errors


@pytest.fixture(autouse=True)
def reset_error_flag(monkeypatch):
    """Reset ERROR_HANDLED flag using monkeypatch"""
    import axolotl.telemetry.errors

    monkeypatch.setattr(axolotl.telemetry.errors, "ERROR_HANDLED", False)
    yield
    monkeypatch.setattr(axolotl.telemetry.errors, "ERROR_HANDLED", False)


@pytest.fixture
def example_stack_trace():
    """Provide a sample stack trace with mixed paths"""
    return """Traceback (most recent call last):
  File "/home/user/.local/lib/python3.9/site-packages/axolotl/cli/train.py", line 83, in main
    trainer = get_trainer(cfg)
  File "/home/user/.local/lib/python3.9/site-packages/axolotl/train.py", line 214, in get_trainer
    model = get_model(cfg, tokenizer)
  File "/home/user/.local/lib/python3.9/site-packages/axolotl/utils/models.py", line 120, in get_model
    raise ValueError("Model path not found")
ValueError: Model path not found
"""


@pytest.fixture
def windows_stack_trace():
    """Provide a sample stack trace with Windows paths"""
    return """Traceback (most recent call last):
  File "C:\\Users\\name\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\axolotl\\cli\\train.py", line 83, in main
    trainer = get_trainer(cfg)
  File "C:\\Users\\name\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\axolotl\\train.py", line 214, in get_trainer
    model = get_model(cfg, tokenizer)
  File "C:\\Users\\name\\AppData\\Local\\Programs\\Python\\Python39\\lib\\site-packages\\transformers\\models\\auto\\modeling_auto.py", line 482, in from_pretrained
    raise ValueError(f"Unrecognized configuration class {config.__class__}")
ValueError: Unrecognized configuration class <class 'transformers.models.llama.configuration_llama.LlamaConfig'>
"""


@pytest.fixture
def mixed_stack_trace():
    """Provide a sample stack trace with both axolotl and non-axolotl paths"""
    return """Traceback (most recent call last):
  File "/home/user/.local/lib/python3.9/site-packages/axolotl/cli/train.py", line 83, in main
    trainer = get_trainer(cfg)
  File "/home/user/.local/lib/python3.9/site-packages/transformers/trainer.py", line 520, in train
    self._inner_training_loop()
  File "/home/user/.local/lib/python3.9/site-packages/axolotl/utils/trainer.py", line 75, in _inner_training_loop
    super()._inner_training_loop()
  File "/home/user/.local/lib/python3.9/site-packages/torch/utils/data/dataloader.py", line 631, in __next__
    data = self._next_data()
RuntimeError: CUDA out of memory
"""


@pytest.fixture
def venv_stack_trace():
    """Provide a sample stack trace with virtual environment paths"""
    return """Traceback (most recent call last):
  File "/home/user/venv/lib/python3.9/site-packages/transformers/trainer.py", line 1729, in train
    self._inner_training_loop()
  File "/home/user/venv/lib/python3.9/site-packages/transformers/trainer.py", line 2013, in _inner_training_loop
    self.accelerator.backward(loss)
  File "/home/user/venv/lib/python3.9/site-packages/accelerate/accelerator.py", line 1851, in backward
    self.scaler.scale(loss).backward(**kwargs)
  File "/home/user/venv/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(self, gradient, retain_graph, create_graph, inputs=inputs)
RuntimeError: CUDA out of memory
"""


@pytest.fixture
def dist_packages_stack_trace():
    """Provide a sample stack trace with dist-packages paths"""
    return """Traceback (most recent call last):
  File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py", line 631, in __next__
    data = self._next_data()
  File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py", line 675, in _next_data
    data = self._dataset_fetcher.fetch(index)
  File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/_utils/fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/usr/local/lib/python3.8/dist-packages/datasets/arrow_dataset.py", line 2808, in __getitem__
    raise IndexError(f"Index {key} out of range for dataset of length {len(self)}.")
IndexError: Index 10000 out of range for dataset of length 9832.
"""


@pytest.fixture
def project_stack_trace():
    """Provide a sample stack trace from a project directory (not a virtual env)"""
    return """Traceback (most recent call last):
  File "/home/user/projects/myproject/run.py", line 25, in <module>
    main()
  File "/home/user/projects/myproject/src/cli.py", line 45, in main
    app.run()
  File "/home/user/projects/myproject/src/app.py", line 102, in run
    raise ValueError("Configuration missing")
ValueError: Configuration missing
"""


def test_sanitize_stack_trace(example_stack_trace):
    """Test that sanitize_stack_trace properly preserves axolotl paths"""
    sanitized = sanitize_stack_trace(example_stack_trace)

    # Check that personal paths are removed
    assert "/home/user" not in sanitized
    assert ".local/lib/python3.9" not in sanitized

    # Check that site-packages is preserved
    assert "site-packages/axolotl/cli/train.py" in sanitized
    assert "site-packages/axolotl/train.py" in sanitized
    assert "site-packages/axolotl/utils/models.py" in sanitized

    # Check that error message is preserved
    assert "ValueError: Model path not found" in sanitized


def test_sanitize_windows_paths(windows_stack_trace):
    """Test that sanitize_stack_trace handles Windows paths"""
    sanitized = sanitize_stack_trace(windows_stack_trace)

    # Check that personal paths are removed
    assert "C:\\Users\\name" not in sanitized
    assert "AppData\\Local\\Programs\\Python" not in sanitized

    # Check that both axolotl and transformers packages are preserved
    assert (
        "site-packages\\axolotl\\cli\\train.py" in sanitized
        or "site-packages/axolotl/cli/train.py" in sanitized
    )
    assert (
        "site-packages\\axolotl\\train.py" in sanitized
        or "site-packages/axolotl/train.py" in sanitized
    )
    assert (
        "site-packages\\transformers\\models\\auto\\modeling_auto.py" in sanitized
        or "site-packages/transformers/models/auto/modeling_auto.py" in sanitized
    )

    # Check that error message is preserved
    assert "ValueError: Unrecognized configuration class" in sanitized


def test_sanitize_mixed_paths(mixed_stack_trace):
    """Test that sanitize_stack_trace preserves all package paths"""
    sanitized = sanitize_stack_trace(mixed_stack_trace)

    # Check that all package paths are preserved
    assert "site-packages/axolotl/cli/train.py" in sanitized
    assert "site-packages/transformers/trainer.py" in sanitized
    assert "site-packages/axolotl/utils/trainer.py" in sanitized
    assert "site-packages/torch/utils/data/dataloader.py" in sanitized

    # Check that error message is preserved
    assert "RuntimeError: CUDA out of memory" in sanitized


def test_sanitize_venv_paths(venv_stack_trace):
    """Test that sanitize_stack_trace preserves virtual environment package paths"""
    sanitized = sanitize_stack_trace(venv_stack_trace)

    # Check that personal paths are removed
    assert "/home/user/venv" not in sanitized

    # Check that all package paths are preserved
    assert "site-packages/transformers/trainer.py" in sanitized
    assert "site-packages/accelerate/accelerator.py" in sanitized
    assert "site-packages/torch/_tensor.py" in sanitized

    # Check that error message is preserved
    assert "RuntimeError: CUDA out of memory" in sanitized


def test_sanitize_dist_packages(dist_packages_stack_trace):
    """Test that sanitize_stack_trace preserves dist-packages paths"""
    sanitized = sanitize_stack_trace(dist_packages_stack_trace)

    # Check that system paths are removed
    assert "/usr/local/lib/python3.8" not in sanitized

    # Check that all package paths are preserved
    assert "dist-packages/torch/utils/data/dataloader.py" in sanitized
    assert "dist-packages/torch/utils/data/_utils/fetch.py" in sanitized
    assert "dist-packages/datasets/arrow_dataset.py" in sanitized

    # Check that error message is preserved
    assert (
        "IndexError: Index 10000 out of range for dataset of length 9832." in sanitized
    )


def test_sanitize_project_paths(project_stack_trace):
    """Test handling of project paths (non-virtual env)"""
    sanitized = sanitize_stack_trace(project_stack_trace)

    # Check that personal paths are removed
    assert "/home/user/projects" not in sanitized

    # For non-package paths, we should at least preserve the filename
    assert "run.py" in sanitized
    assert "cli.py" in sanitized
    assert "app.py" in sanitized

    # Check that error message is preserved
    assert "ValueError: Configuration missing" in sanitized


@pytest.fixture
def mock_telemetry_manager():
    """Create a mock TelemetryManager"""
    with patch("axolotl.telemetry.errors.TelemetryManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.enabled = True
        mock_manager_class.get_instance.return_value = mock_manager
        yield mock_manager


def test_send_errors_successful_execution(mock_telemetry_manager):
    """Test that send_errors doesn't send telemetry for successful function execution"""

    @send_errors
    def test_func():
        return "success"

    result = test_func()
    assert result == "success"
    mock_telemetry_manager.send_event.assert_not_called()


def test_send_errors_with_exception(mock_telemetry_manager):
    """Test that send_errors sends telemetry when an exception occurs"""
    test_error = ValueError("Test error")

    @send_errors
    def test_func():
        raise test_error

    with pytest.raises(ValueError) as excinfo:
        test_func()

    assert excinfo.value == test_error
    mock_telemetry_manager.send_event.assert_called_once()

    # Check that the error info was passed correctly
    call_args = mock_telemetry_manager.send_event.call_args[1]
    assert "test_func-error" in call_args["event_type"]
    assert "Test error" in call_args["properties"]["exception"]
    assert "stack_trace" in call_args["properties"]


def test_send_errors_nested_calls(mock_telemetry_manager):
    """Test that send_errors only sends telemetry once for nested decorated functions"""

    @send_errors
    def inner_func():
        raise ValueError("Inner error")

    @send_errors
    def outer_func():
        return inner_func()

    with pytest.raises(ValueError):
        outer_func()

    # Telemetry should be sent only once for the inner function
    assert mock_telemetry_manager.send_event.call_count == 1
    call_args = mock_telemetry_manager.send_event.call_args[1]
    assert "inner_func-error" in call_args["event_type"]


def test_send_errors_telemetry_disable():
    """Test that send_errors doesn't attempt to send telemetry when disabled"""

    with patch("axolotl.telemetry.errors.TelemetryManager") as mock_manager_class:
        mock_manager = MagicMock()
        mock_manager.enabled = False
        mock_manager_class.get_instance.return_value = mock_manager

        @send_errors
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_func()

        mock_manager.send_event.assert_not_called()


def test_error_handled_reset():
    """Test that ERROR_HANDLED flag is properly reset"""
    with patch("axolotl.telemetry.errors.TelemetryManager") as mock_manager_class:
        # Create and configure the mock manager
        mock_manager = MagicMock()
        mock_manager.enabled = True
        mock_manager_class.get_instance.return_value = mock_manager

        from axolotl.telemetry.errors import ERROR_HANDLED

        @send_errors
        def test_func():
            raise ValueError("Test error")

        assert not ERROR_HANDLED

        with pytest.raises(ValueError):
            test_func()

        from axolotl.telemetry.errors import ERROR_HANDLED

        assert ERROR_HANDLED


def test_module_path_resolution(mock_telemetry_manager):
    """Test that the module path is correctly resolved for the event type"""
    import inspect

    current_module = inspect.getmodule(test_module_path_resolution).__name__

    @send_errors
    def test_func():
        raise ValueError("Test error")

    with pytest.raises(ValueError):
        test_func()

    assert mock_telemetry_manager.send_event.called
    event_type = mock_telemetry_manager.send_event.call_args[1]["event_type"]

    expected_event_type = f"{current_module}.test_func-error"
    assert expected_event_type == event_type
