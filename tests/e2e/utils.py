"""
helper utils for tests
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from functools import wraps
from pathlib import Path

import torch
from packaging import version
try:
    from tbparse import SummaryReader
except ImportError:  # pragma: no cover - optional dependency
    SummaryReader = None

from axolotl.utils.dict import DictDefault


def with_temp_dir(test_func):
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Pass the temporary directory to the test function
            test_func(*args, temp_dir=temp_dir, **kwargs)
        finally:
            # Clean up the directory after the test
            shutil.rmtree(temp_dir)

    return wrapper


def most_recent_subdir(path):
    base_path = Path(path)
    subdirectories = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirectories:
        return None
    subdir = max(subdirectories, key=os.path.getctime)

    return subdir


def require_torch_2_4_1(test_case):
    """
    Decorator marking a test that requires torch >= 2.5.1
    """

    def is_min_2_4_1():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.4.1")

    return unittest.skipUnless(is_min_2_4_1(), "test requires torch>=2.4.1")(test_case)


def require_torch_2_5_1(test_case):
    """
    Decorator marking a test that requires torch >= 2.5.1
    """

    def is_min_2_5_1():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.5.1")

    return unittest.skipUnless(is_min_2_5_1(), "test requires torch>=2.5.1")(test_case)


def require_torch_2_6_0(test_case):
    """
    Decorator marking a test that requires torch >= 2.6.0
    """

    def is_min_2_6_0():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.6.0")

    return unittest.skipUnless(is_min_2_6_0(), "test requires torch>=2.6.0")(test_case)


def require_torch_2_7_0(test_case):
    """
    Decorator marking a test that requires torch >= 2.7.0
    """

    def is_min_2_7_0():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.7.0")

    return unittest.skipUnless(is_min_2_7_0(), "test requires torch>=2.7.0")(test_case)


def require_torch_2_8_0(test_case):
    """
    Decorator marking a test that requires torch >= 2.7.0
    """

    def is_min_2_8_0():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.8.0")

    return unittest.skipUnless(is_min_2_8_0(), "test requires torch>=2.8.0")(test_case)


def require_torch_lt_2_6_0(test_case):
    """
    Decorator marking a test that requires torch < 2.6.0
    """

    def is_max_2_6_0():
        torch_version = version.parse(torch.__version__)
        return torch_version < version.parse("2.6.0")

    return unittest.skipUnless(is_max_2_6_0(), "test requires torch<2.6.0")(test_case)


def require_vllm(test_case):
    """
    Decorator marking a test that requires a vllm to be installed
    """

    def is_vllm_installed():
        return importlib.util.find_spec("vllm") is not None

    return unittest.skipUnless(
        is_vllm_installed(), "test requires vllm to be installed"
    )(test_case)


def require_llmcompressor(test_case):
    """
    Decorator marking a test that requires a llmcompressor to be installed
    """

    def is_llmcompressor_installed():
        return importlib.util.find_spec("llmcompressor") is not None

    return unittest.skipUnless(
        is_llmcompressor_installed(), "test requires llmcompressor to be installed"
    )(test_case)


def requires_sm_ge_100(test_case):
    is_sm_ge_100 = (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (10, 0)
    )
    return unittest.skipUnless(is_sm_ge_100, "test requires sm>=100")(test_case)


def requires_cuda_ge_8_9(test_case):
    is_cuda_ge_8_9 = (
        torch.cuda.is_available()
        and torch.version.cuda
        and torch.cuda.get_device_capability() >= (8, 9)
    )
    return unittest.skipUnless(is_cuda_ge_8_9, "test requires cuda>=8.9")(test_case)


def is_hopper():
    compute_capability = torch.cuda.get_device_capability()
    return compute_capability == (9, 0)


def require_hopper(test_case):
    return unittest.skipUnless(is_hopper(), "test requires h100/hopper GPU")(test_case)


def check_tensorboard(
    temp_run_dir: str,
    tag: str,
    lt_val: float,
    assertion_err: str,
    rtol: float = 0.02,
) -> None:
    """
    helper function to parse and check tensorboard logs
    """
    if SummaryReader is None:
        raise unittest.SkipTest("tbparse is not installed; skipping tensorboard assertions")
    tb_log_path = most_recent_subdir(temp_run_dir)
    event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
    reader = SummaryReader(event_file)
    df = reader.scalars
    df = df[(df.tag == tag)]
    lt_val = (1 + rtol) * lt_val
    if "%s" in assertion_err:
        assert df.value.values[-1] < lt_val, assertion_err % df.value.values[-1]
    else:
        assert df.value.values[-1] < lt_val, assertion_err


def check_model_output_exists(temp_dir: str, cfg: DictDefault) -> None:
    """
    helper function to check if a model output file exists after training

    checks based on adapter or not and if safetensors saves are enabled or not
    """

    if cfg.save_safetensors:
        if not cfg.adapter:
            assert (Path(temp_dir) / "model.safetensors").exists()
        else:
            assert (Path(temp_dir) / "adapter_model.safetensors").exists()
    else:
        # check for both, b/c in trl, it often defaults to saving safetensors
        if not cfg.adapter:
            assert (Path(temp_dir) / "pytorch_model.bin").exists() or (
                Path(temp_dir) / "model.safetensors"
            ).exists()
        else:
            assert (Path(temp_dir) / "adapter_model.bin").exists() or (
                Path(temp_dir) / "adapter_model.safetensors"
            ).exists()
