"""
helper function to parse and check tensorboard logs
"""
import os

from e2e.utils import most_recent_subdir
from tbparse import SummaryReader


def check_tensorboard(
    temp_run_dir: str, tag: str, lt_val: float, assertion_err: str
) -> None:
    tb_log_path = most_recent_subdir(temp_run_dir)
    event_file = os.path.join(tb_log_path, sorted(os.listdir(tb_log_path))[0])
    reader = SummaryReader(event_file)
    df = reader.scalars  # pylint: disable=invalid-name
    df = df[(df.tag == tag)]  # pylint: disable=invalid-name
    assert df.value.values[-1] < lt_val, assertion_err
