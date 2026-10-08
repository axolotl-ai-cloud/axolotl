"""Launch two-rank NCCL per-tensor ratio clipping coverage on FSDP2 (incl. CPU offload)."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_grad_norm_ratio_fsdp2_nccl(tmp_path):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")

    worker = Path(__file__).with_name("_grad_norm_ratio_fsdp2_nccl_worker.py")
    log_path = tmp_path / "worker.log"
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        OMP_NUM_THREADS="1",
    )
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
            ],
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            pytest.fail(log_path.read_text())
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert output.count("GRAD_NORM_RATIO_FSDP2_NCCL_PASS") == 1, output
