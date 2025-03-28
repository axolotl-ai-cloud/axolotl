"""
shared pytest fixtures
"""

import functools
import importlib
import shutil
import sys
import tempfile
import time

import pytest
import requests
from huggingface_hub import snapshot_download
from utils import disable_hf_offline


def retry_on_request_exceptions(max_retries=3, delay=1):
    # pylint: disable=duplicate-code
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # pylint: disable=inconsistent-return-statements
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.HTTPError,
                ) as exc:
                    if attempt < max_retries - 1:
                        wait = 2**attempt * delay  # in seconds
                        time.sleep(wait)
                    else:
                        raise exc

        return wrapper

    return decorator


@retry_on_request_exceptions(max_retries=3, delay=5)
def snapshot_download_w_retry(*args, **kwargs):
    return snapshot_download(*args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_smollm2_135m_model():
    # download the model
    snapshot_download_w_retry("HuggingFaceTB/SmolLM2-135M", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_llama_68m_random_model():
    # download the model
    snapshot_download_w_retry("JackFram/llama-68m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_qwen_2_5_half_billion_model():
    # download the model
    snapshot_download_w_retry("Qwen/Qwen2.5-0.5B", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_tatsu_lab_alpaca_dataset():
    # download the dataset
    snapshot_download_w_retry("tatsu-lab/alpaca", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_mhenrichsen_alpaca_2k_dataset():
    # download the dataset
    snapshot_download_w_retry("mhenrichsen/alpaca_2k_test", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_mhenrichsen_alpaca_2k_w_revision_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "mhenrichsen/alpaca_2k_test", repo_type="dataset", revision="d05c1cb"
    )


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_mlabonne_finetome_100k_dataset():
    # download the dataset
    snapshot_download_w_retry("mlabonne/FineTome-100k", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_argilla_distilabel_capybara_dpo_7k_binarized_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/distilabel-capybara-dpo-7k-binarized", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_argilla_ultrafeedback_binarized_preferences_cleaned_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/ultrafeedback-binarized-preferences-cleaned", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_fozzie_alpaca_dpo_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "fozziethebeat/alpaca_messages_2k_dpo_test", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_arcee_ai_distilabel_intel_orca_dpo_pairs_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_tiny_imdb_dataset():
    # download the dataset
    snapshot_download_w_retry("iamholmes/tiny-imdb", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_deepseek_model_fixture():
    snapshot_download_w_retry("axolotl-ai-co/DeepSeek-V3-11M", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_huggyllama_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "huggyllama/llama-7b",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_llama_1b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Llama-3.2-1B",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_llama3_8b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Meta-Llama-3-8B", repo_type="model", allow_patterns=["*token*"]
    )


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_llama3_8b_instruct_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Meta-Llama-3-8B-Instruct",
        repo_type="model",
        allow_patterns=["*token*"],
    )


@pytest.fixture(scope="session", autouse=True)
@disable_hf_offline
def download_phi_35_mini_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "microsoft/Phi-3.5-mini-instruct", repo_type="model", allow_patterns=["*token*"]
    )


@pytest.fixture(scope="session", autouse=True)
def download_phi_3_medium_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "microsoft/Phi-3-medium-128k-instruct",
        repo_type="model",
        allow_patterns=["*token*"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_mistral_7b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "casperhansen/mistral-7b-instruct-v0.1-awq",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_gemma_2b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "unsloth/gemma-2b-it",
        revision="703fb4a",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_mlx_mistral_7b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama2_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Llama-2-7b-hf",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture
def temp_dir():
    # Create a temporary directory
    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    # Clean up the directory after the test
    shutil.rmtree(_temp_dir)


@pytest.fixture(scope="function", autouse=True)
def cleanup_monkeypatches():
    from transformers import Trainer
    from transformers.models.llama.modeling_llama import (  # LlamaFlashAttention2,
        LlamaAttention,
        LlamaForCausalLM,
    )

    # original_fa2_forward = LlamaFlashAttention2.forward
    original_llama_attn_forward = LlamaAttention.forward
    original_llama_forward = LlamaForCausalLM.forward
    original_trainer_inner_training_loop = (
        Trainer._inner_training_loop  # pylint: disable=protected-access
    )
    original_trainer_training_step = Trainer.training_step
    # monkey patches can happen inside the tests
    yield
    # Reset LlamaFlashAttention2 forward
    # LlamaFlashAttention2.forward = original_fa2_forward
    LlamaAttention.forward = original_llama_attn_forward
    LlamaForCausalLM.forward = original_llama_forward
    Trainer._inner_training_loop = (  # pylint: disable=protected-access
        original_trainer_inner_training_loop
    )
    Trainer.training_step = original_trainer_training_step

    # Reset other known monkeypatches
    modules_to_reset: list[tuple[str, list[str]]] = [
        ("transformers.models.llama",),
        (
            "transformers.models.llama.modeling_llama",
            [
                # "LlamaFlashAttention2",
                "LlamaAttention",
            ],
        ),
        ("transformers.trainer",),
        ("transformers", ["Trainer"]),
        ("transformers.loss.loss_utils",),
    ]
    for module_name_tuple in modules_to_reset:
        module_name = module_name_tuple[0]

        spec = importlib.util.spec_from_file_location(
            module_name, sys.modules[module_name].__file__
        )
        sys.modules[module_name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[module_name])

        sys.modules[module_name] = importlib.reload(sys.modules[module_name])
        if len(module_name_tuple) > 1:
            module_globals = module_name_tuple[1]
            for module_global in module_globals:
                globals().pop(module_global, None)
