"""Two-rank FSDP2 NVFP4 recipe broadcast regression."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_native_nvfp4_fsdp2_recipe(tmp_path):
    if __import__("torch").cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    worker = Path(__file__).with_name("_native_nvfp4_fsdp2_recipe_worker.py")
    log_path = tmp_path / "worker.log"
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
            env=os.environ,
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
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert "NATIVE_NVFP4_FSDP2_RECIPE_PASS" in output
