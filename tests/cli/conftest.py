"""
shared pytest fixtures for cli module
"""
import pytest
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()
