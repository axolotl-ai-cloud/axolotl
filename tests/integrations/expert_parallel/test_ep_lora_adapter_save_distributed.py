"""The gathered EP-LoRA adapter save must not hold the full adapter on every rank's GPU."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port


@pytest.mark.distributed_cpu
def test_ep_lora_adapter_save_keeps_tensors_on_writer_only(tmp_path):
    worker = Path(__file__).with_name("_ep_lora_adapter_save_worker.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            str(tmp_path),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
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
        pytest.fail("Distributed adapter save test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for case in (
        "ep-lora-adapter-save-writer-only",
        "fsdp2-lora-adapter-save-writer-only",
        "ep-lora-adapter-save-rejects-unfilled",
        "ep-lora-adapter-trainer-save-exact",
    ):
        assert f"PASS {case}" in stdout
