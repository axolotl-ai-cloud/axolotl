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
from tbparse import SummaryReader

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


def supports_fp8(test_case):
    compute_capability = torch.cuda.get_device_capability()
    return unittest.skipUnless(
        compute_capability >= (9, 0), "test requires h100 or newer GPU"
    )(test_case)


def check_tensorboard(
    temp_run_dir: str,
    tag: str,
    lt_val: float,
    assertion_err: str,
    rtol: float = 0.05,
    gt_zero: bool = True,
) -> None:
    """
    helper function to parse and check tensorboard logs
    """
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
    if gt_zero:
        assert df.value.values[-1] > 1e-5, "Expected loss to be greater than zero"


def check_tensorboard_loss_decreased(
    temp_run_dir: str,
    tag: str | None = None,
    initial_window: int = 1,
    final_window: int = 1,
    min_delta: float | None = None,
    max_initial: float | None = None,
    max_final: float | None = None,
    max_loss_ratio: float = 1.10,
) -> None:
    """Check that training didn't regress — loss stayed in a sensible range.

    Used with the tiny ``axolotl-ai-co/tiny-*`` CI models, where pretraining
    was brief enough that final loss won't clear the absolute thresholds used
    for 135M+ models — but the training pipeline should still behave.

    ``train/train_loss`` is only logged once (end-of-training aggregate). The
    per-step tag is ``train/loss`` for SFT/LM trainers and may vary across
    trainers (e.g. DPO). When ``tag`` is None we try common per-step tags in
    order and use the first with enough samples.

    Two kinds of regression we guard against:

    1. **Loss blew up.** A silent bug (e.g. broken label masking) can start
       training at an absurdly high loss. ``max_initial`` / ``max_final``
       assert the measured means stay at-or-below bounds measured from a
       known-good run. Both are optional but strongly encouraged — loss
       going *down* from a bad starting scale still looks like "learning."

    2. **Training diverged.** ``max_loss_ratio`` (default 1.10) requires
       ``final <= initial * ratio``. Allows small noise in flat-loss cases
       (common with tiny pretrained models that start near optimum), but
       a final loss 10%+ above initial flags instability / NaNs / drift.

    ``min_delta`` is optional; when set, additionally requires
    ``final + min_delta <= initial`` — use for configs with enough signal
    to demand a strict decrease.
    """
    tb_log_path = most_recent_subdir(temp_run_dir)
    event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
    reader = SummaryReader(event_file)
    df = reader.scalars

    if tag is None:
        candidates = ["train/loss", "train/train_loss"]
    else:
        candidates = [tag]

    required = initial_window + final_window
    chosen_tag, values = None, None
    for candidate in candidates:
        sub = df[df.tag == candidate]
        if len(sub) >= required:
            chosen_tag = candidate
            values = sub.value.values
            break

    available = sorted({t for t in df.tag.unique() if "loss" in t.lower()})
    assert values is not None, (
        f"None of the tags {candidates} had ≥{required} logged steps. "
        f"Loss tags present: {available}"
    )

    initial = float(values[:initial_window].mean())
    final = float(values[-final_window:].mean())
    print(
        f"[check_tensorboard_loss_decreased] tag={chosen_tag} n={len(values)} "
        f"initial_mean{initial_window}={initial:.4f} final_mean{final_window}={final:.4f}"
    )
    assert final > 1e-5, "Expected loss to be greater than zero"
    assert final <= initial * max_loss_ratio, (
        f"Loss regressed for {chosen_tag}: "
        f"initial(mean of first {initial_window})={initial:.4f}, "
        f"final(mean of last {final_window})={final:.4f}, "
        f"ratio={final / initial:.4f} (max allowed {max_loss_ratio})"
    )
    if min_delta is not None:
        assert final + min_delta <= initial, (
            f"Expected loss to decrease by at least {min_delta} for {chosen_tag}: "
            f"initial={initial:.4f}, final={final:.4f}, delta={initial - final:.4f}"
        )
    if max_initial is not None:
        assert initial <= max_initial, (
            f"Initial loss {initial:.4f} is above the expected max {max_initial}. "
            f"Absolute scale is wrong — probably a silent regression "
            f"(e.g. bad label masking) that bumped the starting point."
        )
    if max_final is not None:
        assert final <= max_final, (
            f"Final loss {final:.4f} is above the expected max {max_final}. "
            f"Absolute scale is wrong — probably a silent regression "
            f"(e.g. bad label masking) that bumped the endpoint."
        )


def check_model_output_exists(temp_dir: str, cfg: DictDefault) -> None:
    """
    helper function to check if a model output file exists after training

    checks based on adapter or not (always safetensors in Transformers V5)
    """

    if not cfg.adapter:
        assert (Path(temp_dir) / "model.safetensors").exists()
    else:
        assert (Path(temp_dir) / "adapter_model.safetensors").exists()
