"""Two-rank FSDP2 static native-NVFP4 LoRA parity."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_native_nvfp4_fsdp2_lora_parity(tmp_path, offload):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")

    worker = Path(__file__).with_name("_native_nvfp4_fsdp2_lora_parity_worker.py")
    log_path = tmp_path / f"worker-offload-{offload}.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["NVFP4_FSDP2_OFFLOAD"] = str(int(offload))

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
        timed_out = False
        try:
            process.wait(timeout=360)
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
    assert "NATIVE_NVFP4_FSDP2_LORA_PARITY_PASS" in output
