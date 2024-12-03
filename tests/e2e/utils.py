"""
helper utils for tests
"""
import os
import shutil
import tempfile
import unittest
from functools import wraps
from pathlib import Path

import torch

# from importlib.metadata import version
from packaging import version
from tbparse import SummaryReader


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


def require_torch_2_3_1(test_case):
    """
    Decorator marking a test that requires torch >= 2.3.1
    """

    def is_min_2_3_1():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.3.1")

    return unittest.skipUnless(is_min_2_3_1(), "test torch 2.3.1")(test_case)


def require_torch_2_5_1(test_case):
    """
    Decorator marking a test that requires torch >= 2.5.1
    """

    def is_min_2_5_1():
        torch_version = version.parse(torch.__version__)
        return torch_version >= version.parse("2.5.1")

    return unittest.skipUnless(is_min_2_5_1(), "test torch 2.5.1")(test_case)


def is_hopper():
    compute_capability = torch.cuda.get_device_capability()
    return compute_capability == (9, 0)


def check_tensorboard(
    temp_run_dir: str, tag: str, lt_val: float, assertion_err: str
) -> None:
    """
    helper function to parse and check tensorboard logs
    """
    tb_log_path = most_recent_subdir(temp_run_dir)
    event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
    reader = SummaryReader(event_file)
    df = reader.scalars  # pylint: disable=invalid-name
    df = df[(df.tag == tag)]  # pylint: disable=invalid-name
    assert df.value.values[-1] < lt_val, assertion_err
