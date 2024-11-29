"""
shared pytest fixtures
"""
import shutil
import tempfile

import pytest
from huggingface_hub import snapshot_download


@pytest.fixture(scope="session", autouse=True)
def download_smollm2_135m_model():
    # download the model
    snapshot_download("HuggingFaceTB/SmolLM2-135M")


@pytest.fixture(scope="session", autouse=True)
def download_qwen_2_5_half_billion_model():
    # download the model
    snapshot_download("Qwen/Qwen2.5-0.5B")


@pytest.fixture(scope="session", autouse=True)
def download_tatsu_lab_alpaca_dataset():
    # download the model
    snapshot_download("tatsu-lab/alpaca", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_mhenrichsen_alpaca_2k_dataset():
    # download the model
    snapshot_download("mhenrichsen/alpaca_2k_test", repo_type="dataset")


def download_mlabonne_finetome_100k_dataset():
    # download the model
    snapshot_download("mlabonne/FineTome-100k", repo_type="dataset")


@pytest.fixture
def temp_dir():
    # Create a temporary directory
    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    # Clean up the directory after the test
    shutil.rmtree(_temp_dir)
