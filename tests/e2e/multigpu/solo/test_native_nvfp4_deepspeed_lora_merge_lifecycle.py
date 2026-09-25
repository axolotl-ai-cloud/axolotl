"""Two-rank Trainer lifecycle gate for native NVFP4 DeepSpeed LoRA."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs required")
@pytest.mark.parametrize("dynamic", [False, True])
def test_native_nvfp4_deepspeed_lora_merge_lifecycle(tmp_path, dynamic):
    pytest.importorskip("deepspeed")
    worker = Path(__file__).with_name("_native_nvfp4_deepspeed_lora_merge_lifecycle.py")
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
        "TORCHAO_LORA_DEEPSPEED_CHECKPOINT_TMP": str(tmp_path),
        "OMP_NUM_THREADS": "1",
        "ACCELERATE_DEEPSPEED_ZERO_STAGE": "3",
        "TORCHAO_LORA_DEEPSPEED_DYNAMIC": str(int(dynamic)),
    }
    for phase in ("reference", "resume"):
        log = tmp_path / f"{phase}.log"
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
                env=env | {"TORCHAO_LORA_DEEPSPEED_PHASE": phase},
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            timed_out = False
            try:
                process.wait(timeout=900)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        output = log.read_text()
        assert not timed_out, output
        assert process.returncode == 0, output
        assert output.count("NATIVE_NVFP4_DEEPSPEED_LIFECYCLE_PASS") == 2, output
