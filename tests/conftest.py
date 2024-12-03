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
def download_llama_68m_random_model():
    # download the model
    snapshot_download("JackFram/llama-68m")


@pytest.fixture(scope="session", autouse=True)
def download_qwen_2_5_half_billion_model():
    # download the model
    snapshot_download("Qwen/Qwen2.5-0.5B")


@pytest.fixture(scope="session", autouse=True)
def download_tatsu_lab_alpaca_dataset():
    # download the dataset
    snapshot_download("tatsu-lab/alpaca", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_mhenrichsen_alpaca_2k_dataset():
    # download the dataset
    snapshot_download("mhenrichsen/alpaca_2k_test", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_mhenrichsen_alpaca_2k_w_revision_dataset():
    # download the dataset
    snapshot_download(
        "mhenrichsen/alpaca_2k_test", repo_type="dataset", revision="d05c1cb"
    )


@pytest.fixture(scope="session", autouse=True)
def download_mlabonne_finetome_100k_dataset():
    # download the dataset
    snapshot_download("mlabonne/FineTome-100k", repo_type="dataset")


@pytest.fixture
def download_argilla_distilabel_capybara_dpo_7k_binarized_dataset():
    # download the dataset
    snapshot_download(
        "argilla/distilabel-capybara-dpo-7k-binarized", repo_type="dataset"
    )


@pytest.fixture
def download_arcee_ai_distilabel_intel_orca_dpo_pairs_dataset():
    # download the dataset
    snapshot_download(
        "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized", repo_type="dataset"
    )


@pytest.fixture
def temp_dir():
    # Create a temporary directory
    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    # Clean up the directory after the test
    shutil.rmtree(_temp_dir)
