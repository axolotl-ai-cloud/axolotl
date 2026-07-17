"""Shared pytest fixtures"""

import collections
import functools
import importlib
import logging
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import pytest
import requests

from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import (
    enable_hf_offline,
    hf_offline_context,
)

logging.getLogger("filelock").setLevel(logging.CRITICAL)


@contextmanager
def capture_axolotl_warnings(caplog):
    """Re-enable propagation on the `axolotl` logger so caplog captures records."""
    ax_logger = logging.getLogger("axolotl")
    old_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            yield
    finally:
        ax_logger.propagate = old_propagate


def _apply_transformers_test_shims():
    import transformers.utils as _transformers_utils
    import transformers.utils.import_utils as _import_utils

    # Shim for deepseek v3
    if not hasattr(_import_utils, "is_torch_fx_available"):

        def _is_torch_fx_available():
            try:
                import torch.fx  # noqa: F401  # pylint: disable=unused-import

                return True
            except ImportError:
                return False

        _import_utils.is_torch_fx_available = _is_torch_fx_available

    if not hasattr(_transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        from transformers.utils import (
            is_flash_attn_greater_or_equal as _is_flash_attn_gte,
        )

        _transformers_utils.is_flash_attn_greater_or_equal_2_10 = lambda: (
            _is_flash_attn_gte("2.10")
        )


def pytest_configure(config):  # pylint: disable=unused-argument
    _apply_transformers_test_shims()


# A device-side assert / illegal access poisons the process-wide CUDA context, so
# every later GPU test errors at setup. Abort the session instead of cascading.
_CUDA_FATAL_MARKERS = (
    "device-side assert triggered",
    "an illegal memory access was encountered",
    "misaligned address",
)
_cuda_context_poisoned = False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # pylint: disable=unused-argument
    outcome = yield
    report = outcome.get_result()
    global _cuda_context_poisoned  # pylint: disable=global-statement
    if report.failed and call.excinfo is not None:
        if any(marker in str(call.excinfo.value) for marker in _CUDA_FATAL_MARKERS):
            _cuda_context_poisoned = True


def pytest_runtest_setup(item):
    if _cuda_context_poisoned:
        item.session.shouldstop = (
            "CUDA context corrupted by an earlier test; aborting to avoid "
            "cascading setup errors. Re-run the job."
        )
        pytest.skip("CUDA context corrupted by an earlier test; aborting suite.")


def retry_on_request_exceptions(max_retries=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
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
    """
    download a model or dataset from HF Hub, retrying in requests failures. We also try to fetch it from the local
    cache first using hf_hub_offline to avoid hitting HF Hub API rate limits. If it doesn't exist in the cache,
    disable hf_hub_offline and actually fetch from the hub
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    with hf_offline_context(True):
        try:
            return snapshot_download(*args, local_files_only=True, **kwargs)
        except LocalEntryNotFoundError:
            pass
    with hf_offline_context(False):
        return snapshot_download(*args, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def download_ds_fixture_bundle():
    ds_dir = snapshot_download_w_retry(
        "axolotl-ai-internal/axolotl-oss-dataset-fixtures", repo_type="dataset"
    )
    return Path(ds_dir)


@pytest.fixture(scope="session", autouse=True)
def download_smollm2_135m_model():
    # download the model
    snapshot_download_w_retry("HuggingFaceTB/SmolLM2-135M", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_smollm2_135m_instruct_model():
    # download the model
    snapshot_download_w_retry("HuggingFaceTB/SmolLM2-135M-Instruct", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_smollm2_135m_gptq_model():
    # download the model
    snapshot_download_w_retry("lilmeaty/SmolLM2-135M-Instruct-GPTQ", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_qwen3_half_billion_model():
    # download the model (still used as the KD teacher in tests/e2e/integrations/test_kd.py)
    snapshot_download_w_retry("Qwen/Qwen3-0.6B", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_llama_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-llama-50m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_mistral_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-mistral-25m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_mixtral_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-mixtral-30m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_phi_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-phi-64m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_falcon_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-falcon-42m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_qwen2_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-qwen2-129m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_qwen3_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-qwen3-129m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tiny_gemma2_model():
    snapshot_download_w_retry("axolotl-ai-co/tiny-gemma2-137m", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_tatsu_lab_alpaca_dataset():
    # download the dataset
    snapshot_download_w_retry("tatsu-lab/alpaca", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_mhenrichsen_alpaca_2k_dataset():
    # download the dataset
    snapshot_download_w_retry("mhenrichsen/alpaca_2k_test", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_mhenrichsen_alpaca_2k_w_revision_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "mhenrichsen/alpaca_2k_test", repo_type="dataset", revision="d05c1cb"
    )


@pytest.fixture(scope="session", autouse=True)
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
def download_argilla_distilabel_intel_orca_dpo_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/distilabel-intel-orca-dpo-pairs", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_argilla_ultrafeedback_binarized_preferences_cleaned_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/ultrafeedback-binarized-preferences-cleaned", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_argilla_ultrafeedback_binarized_preferences_cleaned_kto_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/ultrafeedback-binarized-preferences-cleaned-kto", repo_type="dataset"
    )


# @pytest.fixture(scope="session", autouse=True)
# def download_fozzie_alpaca_dpo_dataset():
#     # download the dataset
#     snapshot_download_w_retry(
#         "fozziethebeat/alpaca_messages_2k_dpo_test", repo_type="dataset"
#     )
#     snapshot_download_w_retry(
#         "fozziethebeat/alpaca_messages_2k_dpo_test",
#         repo_type="dataset",
#         revision="ea82cff",
#     )


# @pytest.fixture(scope="session")
# @disable_hf_offline
# def dataset_fozzie_alpaca_dpo_dataset(
#     download_fozzie_alpaca_dpo_dataset,
# ):
#     return load_dataset("fozziethebeat/alpaca_messages_2k_dpo_test", split="train")
#
#
# @pytest.fixture(scope="session")
# @disable_hf_offline
# def dataset_fozzie_alpaca_dpo_dataset_rev_ea82cff(
#     download_fozzie_alpaca_dpo_dataset,
# ):
#     return load_dataset(
#         "fozziethebeat/alpaca_messages_2k_dpo_test", split="train", revision="ea82cff"
#     )


@pytest.fixture(scope="session", autouse=True)
def download_arcee_ai_distilabel_intel_orca_dpo_pairs_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "arcee-ai/distilabel-intel-orca-dpo-pairs-binarized", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_argilla_dpo_pairs_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "argilla/distilabel-intel-orca-dpo-pairs", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_tiny_shakespeare_dataset():
    # download the dataset
    snapshot_download_w_retry("winglian/tiny-shakespeare", repo_type="dataset")


@pytest.fixture(scope="session", autouse=True)
def download_evolkit_kd_sample_dataset():
    # download the dataset
    snapshot_download_w_retry(
        "axolotl-ai-co/evolkit-logprobs-pipeline-75k-v2-sample", repo_type="dataset"
    )


@pytest.fixture(scope="session", autouse=True)
def download_deepseek_model_fixture():
    snapshot_download_w_retry("axolotl-ai-co/DeepSeek-V3-11M", repo_type="model")


@pytest.fixture(scope="session", autouse=True)
def download_huggyllama_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "huggyllama/llama-7b",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama33_70b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "axolotl-ai-co/Llama-3.3-70B-Instruct-tokenizer",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama_1b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Llama-3.2-1B",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama3_8b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Meta-Llama-3-8B",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama3_8b_instruct_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Meta-Llama-3-8B-Instruct",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_phi_35_mini_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "microsoft/Phi-3.5-mini-instruct",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_phi_4_reasoning_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "microsoft/Phi-4-reasoning",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_phi_3_mini_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "microsoft/Phi-3-mini-4k-instruct",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
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
def download_gemma3_4b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "mlx-community/gemma-3-4b-it-8bit",
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
def download_gemma2_9b_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "mlx-community/gemma-2-9b-it-4bit",
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


@pytest.fixture
def download_llama2_model_fixture():
    # download the tokenizer only
    snapshot_download_w_retry(
        "NousResearch/Llama-2-7b-hf",
        repo_type="model",
        allow_patterns=["*token*", "config.json"],
    )


@pytest.fixture(scope="session", autouse=True)
def download_llama32_1b_model_fixture():
    snapshot_download_w_retry(
        "osllmai-community/Llama-3.2-1B",
        repo_type="model",
    )


@pytest.fixture
@enable_hf_offline
def tokenizer_huggyllama(
    download_huggyllama_model_fixture,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
    tokenizer.pad_token = "</s>"

    return tokenizer


@pytest.fixture
@enable_hf_offline
def tokenizer_huggyllama_w_special_tokens(
    tokenizer_huggyllama,
):
    tokenizer_huggyllama.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
    )

    return tokenizer_huggyllama


@pytest.fixture
@enable_hf_offline
def tokenizer_llama2_7b(
    download_llama2_model_fixture,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

    return tokenizer


@pytest.fixture
@enable_hf_offline
def tokenizer_mistral_7b_instruct(
    download_mlx_mistral_7b_model_fixture,
):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("casperhansen/mistral-7b-instruct-v0.1-awq")


@pytest.fixture
def tokenizer_mistral_7b_instruct_chatml(tokenizer_mistral_7b_instruct):
    from tokenizers import AddedToken

    tokenizer_mistral_7b_instruct.add_special_tokens(
        {
            "eos_token": AddedToken(
                "<|im_end|>", rstrip=False, lstrip=False, normalized=False
            )
        }
    )
    tokenizer_mistral_7b_instruct.add_tokens(
        [
            AddedToken("<|im_start|>", rstrip=False, lstrip=False, normalized=False),
        ]
    )
    return tokenizer_mistral_7b_instruct


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    # Create a temporary directory
    _temp_dir = tempfile.mkdtemp()
    yield _temp_dir
    # Clean up the directory after the test
    shutil.rmtree(_temp_dir)


def _clear_plugin_manager():
    from axolotl.integrations.base import PluginManager

    PluginManager._cfg = None
    # Don't reset _instance to None — module-level PLUGIN_MANAGER references
    # in train.py, model.py, etc. would become stale
    if PluginManager._instance is not None:
        PluginManager._instance.plugins = collections.OrderedDict()


@pytest.fixture(scope="function", autouse=True)
def reset_plugin_manager():
    _clear_plugin_manager()
    yield
    _clear_plugin_manager()


@pytest.fixture(scope="function", autouse=True)
def torch_manual_seed():
    import torch

    torch.manual_seed(42)


_TRANSFORMERS_MODULES_TO_RESET = (
    "transformers.models.llama",
    "transformers.models.llama.modeling_llama",
    "transformers.trainer",
    "transformers",
    "transformers.loss.loss_utils",
)

_TRANSFORMERS_PATCH_TARGETS = (
    ("transformers.models.llama.modeling_llama", ("LlamaAttention", "forward")),
    ("transformers.models.llama.modeling_llama", ("LlamaForCausalLM", "forward")),
    ("transformers.trainer", ("Trainer", "_inner_training_loop")),
    ("transformers.trainer", ("Trainer", "training_step")),
    ("transformers", ("Trainer",)),
    # explicit targets so patches are reverted even when loss_utils was already imported
    ("transformers.loss.loss_utils", ("fixed_cross_entropy",)),
    ("transformers.loss.loss_utils", ("ForCausalLMLoss",)),
)


def _get_nested_attr(obj, attr_path):
    for attr in attr_path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _set_nested_attr(obj, attr_path, value):
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attr_path[-1], value)


@pytest.fixture(scope="function", autouse=True)
def cleanup_monkeypatches():
    seen_modules = {
        module_name
        for module_name in _TRANSFORMERS_MODULES_TO_RESET
        if module_name in sys.modules
    }
    snapshots = {}
    for module_name, attr_path in _TRANSFORMERS_PATCH_TARGETS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        value = _get_nested_attr(module, attr_path)
        if value is not None:
            snapshots[(module_name, attr_path)] = value

    yield

    modules_to_reload = set()
    for (module_name, attr_path), original_value in snapshots.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        current_value = _get_nested_attr(module, attr_path)
        if current_value is not original_value:
            _set_nested_attr(module, attr_path, original_value)
            modules_to_reload.add(module_name)

    modules_to_reload.update(
        module_name
        for module_name in _TRANSFORMERS_MODULES_TO_RESET
        if module_name not in seen_modules and module_name in sys.modules
    )

    for module_name in _TRANSFORMERS_MODULES_TO_RESET:
        if module_name not in modules_to_reload:
            continue
        module = sys.modules.get(module_name)
        if module is None or not getattr(module, "__file__", None):
            continue
        sys.modules[module_name] = importlib.reload(module)


@pytest.fixture
def dataset_winglian_tiny_shakespeare(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = download_ds_fixture_bundle / "winglian__tiny-shakespeare"
    return load_from_disk(ds_path)


@pytest.fixture
def dataset_tatsu_lab_alpaca(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = download_ds_fixture_bundle / "tatsu-lab__alpaca"
    return load_from_disk(ds_path)["train"]


@pytest.fixture
def dataset_mhenrichsen_alpaca_2k_test(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = download_ds_fixture_bundle / "mhenrichsen__alpaca_2k_test"
    return load_from_disk(ds_path)["train"]


@pytest.fixture
def dataset_argilla_ultrafeedback_binarized_preferences_cleaned(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = (
        download_ds_fixture_bundle
        / "argilla__ultrafeedback-binarized-preferences-cleaned"
    )
    return load_from_disk(ds_path)["train"]


@pytest.fixture
def dataset_fozziethebeat_alpaca_messages_2k_dpo_test(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = download_ds_fixture_bundle / "fozziethebeat__alpaca_messages_2k_dpo_test"
    return load_from_disk(ds_path)["train"]


@pytest.fixture
def dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff(
    download_ds_fixture_bundle: Path,
):
    from datasets import load_from_disk

    ds_path = (
        download_ds_fixture_bundle
        / "fozziethebeat__alpaca_messages_2k_dpo_test__rev_ea82cff"
    )
    return load_from_disk(ds_path)["train"]


@pytest.fixture(name="min_base_cfg")
def fixture_min_base_cfg():
    return DictDefault(
        base_model="HuggingFaceTB/SmolLM2-135M",
        learning_rate=1e-3,
        datasets=[
            {
                "path": "mhenrichsen/alpaca_2k_test",
                "type": "alpaca",
            },
        ],
        micro_batch_size=1,
        gradient_accumulation_steps=1,
    )


#
@pytest.mark.skipif(
    os.environ.get("AXOLOTL_IS_CI_CACHE_PRELOAD", "-1") != "1",
    reason="Not running in CI cache preload",
)
def test_load_fixtures(
    download_smollm2_135m_model,
    download_qwen3_half_billion_model,
    download_tiny_llama_model,
    download_tiny_mistral_model,
    download_tiny_mixtral_model,
    download_tiny_phi_model,
    download_tiny_falcon_model,
    download_tiny_qwen2_model,
    download_tiny_qwen3_model,
    download_tiny_gemma2_model,
    download_tatsu_lab_alpaca_dataset,
    download_mhenrichsen_alpaca_2k_dataset,
    download_mhenrichsen_alpaca_2k_w_revision_dataset,
    download_mlabonne_finetome_100k_dataset,
    download_argilla_ultrafeedback_binarized_preferences_cleaned_dataset,
    download_argilla_ultrafeedback_binarized_preferences_cleaned_kto_dataset,
    download_argilla_distilabel_capybara_dpo_7k_binarized_dataset,
    download_arcee_ai_distilabel_intel_orca_dpo_pairs_dataset,
    download_argilla_dpo_pairs_dataset,
    download_tiny_shakespeare_dataset,
    download_deepseek_model_fixture,
    download_huggyllama_model_fixture,
    download_llama_1b_model_fixture,
    download_llama3_8b_model_fixture,
    download_llama3_8b_instruct_model_fixture,
    download_phi_35_mini_model_fixture,
    download_phi_3_medium_model_fixture,
    download_phi_4_reasoning_model_fixture,
    download_mistral_7b_model_fixture,
    download_gemma_2b_model_fixture,
    download_gemma2_9b_model_fixture,
    download_mlx_mistral_7b_model_fixture,
    download_llama2_model_fixture,
):
    pass


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch):
    monkeypatch.setenv("AXOLOTL_DO_NOT_TRACK", "1")
    yield
