"""pytest tests for axolotl CLI utils."""

# pylint: disable=redefined-outer-name

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

    def mock_get(url, timeout=None):  # pylint: disable=unused-argument
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
