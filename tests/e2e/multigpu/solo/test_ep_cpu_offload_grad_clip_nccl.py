"""NCCL validation for CPU-offloaded EP gradient clipping."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs required")
def test_ep_cpu_offload_gradient_clipping_nccl(tmp_path):
    worker = Path(__file__).with_name("_ep_cpu_offload_grad_clip_nccl_worker.py")
    log = tmp_path / "worker.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
            ],
            env=os.environ | {"OMP_NUM_THREADS": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    output = log.read_text()
    assert process.returncode == 0, output
    assert "EP_CPU_OFFLOAD_NCCL_CLIP_PASS" in output
