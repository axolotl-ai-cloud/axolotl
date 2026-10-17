"""Compact expert training must be exercised without requiring GPU CI runners."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port


@pytest.mark.distributed_cpu
def test_compact_expert_gloo_resume(tmp_path):
    worker = Path(__file__).with_name("_distributed.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            "--master-addr=127.0.0.1",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            str(tmp_path),
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Gloo compact-expert test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for topology in ("fsdp2_ep", "hsdp_ep"):
        for selected in ([0, 1], [7, 0]):
            assert f"PASS {topology}-{selected}" in stdout
