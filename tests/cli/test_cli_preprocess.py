"""
pytest tests for axolotl CLI preprocess command
"""
import shutil
from pathlib import Path

import pytest

from axolotl.cli.main import cli
from axolotl.common.const import DEFAULT_DATASET_PREPARED_PATH

from .conftest import VALID_TEST_CONFIG


@pytest.fixture(autouse=True)
def cleanup_last_run_prepared():
    yield

    if Path("last_run_prepared").exists():
        shutil.rmtree("last_run_prepared")


def test_preprocess_config_not_found(cli_runner):
    """Test preprocess fails when config not found"""
    result = cli_runner.invoke(cli, ["preprocess", "nonexistent.yml"])
    assert result.exit_code != 0


def test_preprocess_basic(cli_runner, config_path):
    """Test basic preprocessing with minimal config"""
    result = cli_runner.invoke(cli, ["preprocess", str(config_path)])
    assert result.exit_code == 0

    # Verify dataset was prepared
    prepared_path = Path(DEFAULT_DATASET_PREPARED_PATH)
    assert prepared_path.exists()

    # Get the hash-named directory
    dataset_dirs = list(prepared_path.iterdir())
    assert len(dataset_dirs) == 1
    dataset_path = dataset_dirs[0]

    # Verify expected files exist
    assert (dataset_path / "data-00000-of-00001.arrow").exists()
    assert (dataset_path / "state.json").exists()
    assert (dataset_path / "dataset_info.json").exists()


def test_preprocess_rl(cli_runner, config_path):
    """Test preprocessing with RL config"""
    result = cli_runner.invoke(cli, ["preprocess", str(config_path)])
    assert result.exit_code == 0

    # Verify dataset was prepared
    prepared_path = Path(DEFAULT_DATASET_PREPARED_PATH)
    assert prepared_path.exists()

    # Get the hash-named directory
    dataset_dirs = list(prepared_path.iterdir())
    assert len(dataset_dirs) == 1
    dataset_path = dataset_dirs[0]

    # Verify expected files exist
    import os

    print(os.listdir(dataset_path))
    assert (dataset_path / "data-00000-of-00001.arrow").exists()
    assert (dataset_path / "state.json").exists()
    assert (dataset_path / "dataset_info.json").exists()


def test_preprocess_without_download(cli_runner, config_path):
    """Test preprocessing without model download"""
    result = cli_runner.invoke(cli, ["preprocess", str(config_path), "--no-download"])
    assert result.exit_code == 0

    # Verify dataset was prepared
    prepared_path = Path(DEFAULT_DATASET_PREPARED_PATH)
    assert prepared_path.exists()

    # Get the hash-named directory
    dataset_dirs = list(prepared_path.iterdir())
    assert len(dataset_dirs) == 1
    dataset_path = dataset_dirs[0]

    # Verify expected files exist
    assert (dataset_path / "data-00000-of-00001.arrow").exists()
    assert (dataset_path / "state.json").exists()
    assert (dataset_path / "dataset_info.json").exists()

    # Model shouldn't be downloaded
    model_path = Path("HuggingFaceTB/SmolLM2-135M")
    assert not model_path.exists()


def test_preprocess_custom_path(cli_runner, tmp_path):
    """Test preprocessing with custom dataset path"""
    config_path = tmp_path / "config.yml"
    custom_path = tmp_path / "custom_prepared"
    config_path.write_text(VALID_TEST_CONFIG)

    result = cli_runner.invoke(
        cli,
        [
            "preprocess",
            str(config_path),
            "--dataset-prepared-path",
            str(custom_path.absolute()),
        ],
    )
    print(result.output)
    assert result.exit_code == 0

    # Verify dataset was prepared
    assert custom_path.exists()

    # Get the hash-named directory
    dataset_dirs = list(custom_path.iterdir())
    assert len(dataset_dirs) == 1
    dataset_path = dataset_dirs[0]

    # Verify expected files exist
    assert (dataset_path / "data-00000-of-00001.arrow").exists()
    assert (dataset_path / "state.json").exists()
    assert (dataset_path / "dataset_info.json").exists()

    # Verify default path wasn't used
    assert not Path(DEFAULT_DATASET_PREPARED_PATH).exists()
