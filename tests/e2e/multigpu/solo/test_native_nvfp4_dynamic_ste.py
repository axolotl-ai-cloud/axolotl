"""Two-rank GPU oracle for frozen dynamic NVFP4 input-gradient STEs."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_native_nvfp4_dynamic_ste_non_target_lora_gradients(tmp_path):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    if any(torch.cuda.get_device_capability(index)[0] < 10 for index in range(2)):
        pytest.skip("dynamic NVFP4 requires two SM100+ GPUs")
    worker = Path(__file__).with_name("_native_nvfp4_dynamic_ste_worker.py")
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
            env=dict(os.environ),
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    output = log_path.read_text()
    assert not timed_out, output
    assert process.returncode == 0, output
    assert "NATIVE_NVFP4_DYNAMIC_STE_PASS" in output
