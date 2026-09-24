"""Two-rank native NVFP4 Hugging Face tensor-parallel checkpoint loading."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="requires two CUDA GPUs"
)
def test_native_nvfp4_hf_tp_load(tmp_path):
    if __import__("torch").cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    worker = Path(__file__).with_name("_native_nvfp4_tp_hf_load_worker.py")
    environment = os.environ | {"NVFP4_TP_GATE_PATH": str(tmp_path / "checkpoint")}
    completed = subprocess.run(
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
        capture_output=True,
        timeout=300,
        check=False,
    )
    (tmp_path / "worker.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "HF_NVFP4_TP_GATE_PASS" in completed.stdout
    assert sys.executable in completed.stdout
