"""Launch two-rank CPU-offloaded DTensor clipping coverage."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_fsdp2_cpu_offload_gradient_clipping(tmp_path):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")

    worker = Path(__file__).with_name("_fsdp2_cpu_offload_grad_clip_worker.py")
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
    assert output.count("FSDP2_CPU_OFFLOAD_GRAD_CLIP_PASS") == 1, output
