"""Per-step publish metrics on a 2-rank gloo Trainer: every rank logs every step."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port

from axolotl.core.publishing.callback import METRIC_PREFIX


@pytest.mark.distributed_cpu
def test_publish_metrics_keep_ranks_logging_in_step(tmp_path):
    worker = Path(__file__).with_name("_publish_log_dist.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            str(tmp_path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=240)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("ranks deadlocked:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr

    rows = json.loads((tmp_path / "log-0.json").read_text())
    steps = [row for row in rows if "loss" in row]
    assert [row["step"] for row in steps] == [1, 2, 3]
    # step N's row carries the publish of step N - 1
    assert [row[METRIC_PREFIX + "step"] for row in steps] == [0, 1, 2]
