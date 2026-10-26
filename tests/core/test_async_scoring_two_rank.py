"""Trainer-side scoring on two gloo ranks: collective ordering and dispatch."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port

WORKER = Path(__file__).with_name("_async_scoring_two_rank_worker.py")


@pytest.mark.distributed_cpu
def test_two_rank_scoring_steps_and_stops_together(
    download_smollm2_135m_instruct_model, tmp_path
):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            f"--master-port={get_torch_dist_unique_port()}",
            str(WORKER),
            str(tmp_path),
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "TRL_EXPERIMENTAL_SILENCE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Two-rank scoring worker timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for rank in (0, 1):
        assert f"PASS two-rank-scoring rank={rank}" in stdout, stdout + stderr
