"""
shared pytest fixtures for cli module
"""
from unittest.mock import DEFAULT

import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def default_config(tmp_path):
    """Create a minimal valid config file"""
    config = {
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
    }
    config_path = tmp_path / "config.yml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    return config_path


@pytest.fixture
def common_mocks():
    return {
        "load_datasets": DEFAULT,
        "load_rl_datasets": DEFAULT,
        "load_cfg": DEFAULT,
        "check_accelerate_default_config": DEFAULT,
        "check_user_token": DEFAULT,
        "print_axolotl_text_art": DEFAULT,
    }
