"""End-to-end parity for ordinary LoRA on native NVFP4 tensor parallel weights."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires two CUDA GPUs"
)
def test_native_nvfp4_tp_lora_parity(tmp_path, dtype):
    if __import__("torch").cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    worker = Path(__file__).with_name("_native_nvfp4_tp_lora_parity_worker.py")
    environment = os.environ | {
        "NVFP4_TP_GATE_PATH": str(tmp_path / "checkpoint"),
        "NVFP4_TP_LORA_DTYPE": dtype,
    }
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        (tmp_path / f"worker-{dtype}.log").write_text(stdout + stderr)
        raise
    (tmp_path / f"worker-{dtype}.log").write_text(stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    assert "HF_NVFP4_TP_LORA_PARITY" in stdout
