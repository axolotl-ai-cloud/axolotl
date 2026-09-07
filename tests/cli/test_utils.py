"""pytest tests for axolotl CLI utils."""

import json
from unittest.mock import Mock, patch

import click
import pytest
import requests

from axolotl.cli.utils import fetch_from_github

# Sample GitHub API response
MOCK_TREE_RESPONSE = {
    "tree": [
        {"path": "examples/config1.yml", "type": "blob", "sha": "abc123"},
        {"path": "examples/config2.yml", "type": "blob", "sha": "def456"},
        {"path": "other/file.txt", "type": "blob", "sha": "xyz789"},
    ]
}


@pytest.fixture
def mock_responses():
    """Mock responses for API and file downloads"""

    def mock_get(url, timeout=None):
        response = Mock()
        if "api.github.com" in url:
            response.text = json.dumps(MOCK_TREE_RESPONSE)
        else:
            response.content = b"file content"
        return response

    return mock_get


def test_fetch_from_github_new_files(tmp_path, mock_responses):
    """Test fetching new files"""
    with patch("requests.get", mock_responses):
        fetch_from_github("examples/", tmp_path)

        # Verify files were created
        assert (tmp_path / "config1.yml").exists()
        assert (tmp_path / "config2.yml").exists()
        assert not (tmp_path / "file.txt").exists()


def test_fetch_from_github_unchanged_files(tmp_path, mock_responses):
    """Test handling of unchanged files"""
    # Create existing file with matching SHA
    existing_file = tmp_path / "config1.yml"
    existing_file.write_bytes(b"file content")

    with patch("requests.get", mock_responses):
        fetch_from_github("examples/", tmp_path)

        # File should not be downloaded again
        assert existing_file.read_bytes() == b"file content"


def test_fetch_from_github_invalid_prefix(mock_responses):
    """Test error handling for invalid directory prefix"""
    with patch("requests.get", mock_responses):
        with pytest.raises(click.ClickException):
            fetch_from_github("nonexistent/", None)


def test_fetch_from_github_network_error():
    """Test handling of network errors"""
    with patch("requests.get", side_effect=requests.RequestException):
        with pytest.raises(requests.RequestException):
            fetch_from_github("examples/", None)


def assert_launcher_args_in_command(
    mock_subprocess_call,
    launcher: str,
    expected_launcher_args: list[str],
    command_module: str,
):
    """
    Helper function to verify launcher arguments are properly passed in subprocess calls.

    Args:
        mock_subprocess_call: The mock subprocess.run call
        launcher: Expected launcher ("accelerate", "torchrun", etc.)
        expected_launcher_args: List of expected launcher arguments
        command_module: Expected module name (e.g., "axolotl.cli.train")
    """
    assert mock_subprocess_call.called, "subprocess.run should have been called"
    called_cmd = mock_subprocess_call.call_args.args[0]

    # Verify launcher
    assert called_cmd[0] == launcher, (
        f"Expected launcher {launcher}, got {called_cmd[0]}"
    )

    # Verify launcher args are present
    for arg in expected_launcher_args:
        assert arg in called_cmd, (
            f"Expected launcher arg '{arg}' not found in command: {called_cmd}"
        )

    # Verify module is present
    assert "-m" in called_cmd, "Expected -m flag for module execution"
    assert command_module in called_cmd, (
        f"Expected module {command_module} not found in command: {called_cmd}"
    )


def assert_no_launcher_args_contamination(mock_subprocess_call, launcher: str):
    """
    Helper function to verify no unwanted launcher arguments are present.

    Args:
        mock_subprocess_call: The mock subprocess.run call
        launcher: Expected launcher ("accelerate", "torchrun", etc.)
    """
    assert mock_subprocess_call.called, "subprocess.run should have been called"
    called_cmd = mock_subprocess_call.call_args.args[0]

    if launcher == "accelerate":
        # For accelerate, launcher args should be between 'launch' and '-m'
        launch_idx = called_cmd.index("launch")
        m_idx = called_cmd.index("-m")
        launcher_section = called_cmd[launch_idx + 1 : m_idx]
        assert len(launcher_section) == 0, (
            f"Unexpected launcher args found: {launcher_section}"
        )
    elif launcher == "torchrun":
        # For torchrun, launcher args should be between 'torchrun' and '-m'
        torchrun_idx = called_cmd.index("torchrun")
        m_idx = called_cmd.index("-m")
        launcher_section = called_cmd[torchrun_idx + 1 : m_idx]
        assert len(launcher_section) == 0, (
            f"Unexpected launcher args found: {launcher_section}"
        )


@pytest.fixture
def common_launcher_args():
    """Fixture providing common launcher argument combinations for testing."""
    return {
        "torchrun": ["--nproc_per_node=2", "--nnodes=1"],
        "accelerate": ["--config_file=accelerate_config.yml", "--num_processes=4"],
    }


def test_add_default_rdzv_args_with_endpoint():
    """Test that default RDZV args are added when rdzv_endpoint is present."""
    from axolotl.cli.utils.train import _add_default_rdzv_args

    launcher_args = ["--nnodes=2", "--rdzv_endpoint=127.0.0.1:29400"]
    result = _add_default_rdzv_args(launcher_args)

    # Should have added rdzv_backend
    assert "--rdzv_backend" in result
    assert "c10d" in result

    # Original args should still be present
    assert "--nnodes=2" in result
    assert "--rdzv_endpoint=127.0.0.1:29400" in result


def test_add_default_rdzv_args_with_existing_backend():
    """Test that existing rdzv_backend is not overridden."""
    from axolotl.cli.utils.train import _add_default_rdzv_args

    launcher_args = [
        "--nnodes=2",
        "--rdzv_endpoint=127.0.0.1:29400",
        "--rdzv_backend=static",
    ]
    result = _add_default_rdzv_args(launcher_args)

    # Should not add another rdzv_backend
    backend_count = sum(1 for arg in result if "--rdzv_backend" in arg)
    assert backend_count == 1
    assert "--rdzv_backend=static" in result


def test_add_default_rdzv_args_with_existing_id():
    """Test that existing rdzv_id is not overridden."""
    from axolotl.cli.utils.train import _add_default_rdzv_args

    launcher_args = [
        "--nnodes=2",
        "--rdzv_endpoint=127.0.0.1:29400",
        "--rdzv_id=my_job_123",
    ]
    result = _add_default_rdzv_args(launcher_args)

    # Should not add another rdzv_id
    id_count = sum(1 for arg in result if "--rdzv_id" in arg)
    assert id_count == 1
    assert "--rdzv_id=my_job_123" in result

    # Should still add rdzv_backend
    assert "--rdzv_backend" in result
    assert "c10d" in result


def test_add_default_rdzv_args_without_endpoint():
    """Test that no RDZV args are added when rdzv_endpoint is not present."""
    from axolotl.cli.utils.train import _add_default_rdzv_args

    launcher_args = ["--nnodes=2", "--nproc_per_node=4"]
    result = _add_default_rdzv_args(launcher_args)

    # Should not add any rdzv args
    assert "--rdzv_backend" not in result
    assert result == launcher_args


def test_add_default_rdzv_args_with_all_existing():
    """Test that no defaults are added when all RDZV args are present."""
    from axolotl.cli.utils.train import _add_default_rdzv_args

    launcher_args = [
        "--nnodes=2",
        "--rdzv_endpoint=127.0.0.1:29400",
        "--rdzv_backend=static",
        "--rdzv_id=existing_job",
    ]
    result = _add_default_rdzv_args(launcher_args)

    # Should not add any additional args
    assert len(result) == len(launcher_args)
    assert result == launcher_args
