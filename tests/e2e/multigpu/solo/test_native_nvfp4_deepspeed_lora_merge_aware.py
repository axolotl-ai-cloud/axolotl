"""Two-rank DeepSpeed static native-NVFP4 merge-aware LoRA validation."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs required",
)
def test_native_nvfp4_deepspeed_lora_merge_aware(tmp_path, stage):
    pytest.importorskip("deepspeed")
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        pytest.skip("TorchAO NVFP4 unavailable")
    if NVFP4Tensor is None:
        pytest.skip("TorchAO NVFP4 unavailable")
    worker = Path(__file__).with_name("_native_nvfp4_deepspeed_lora_merge_aware.py")
    log_path = tmp_path / f"worker-zero{stage}.log"
    env = os.environ | {
        "NATIVE_NVFP4_DEEPSPEED_LORA_TMP": str(tmp_path),
        "ZERO_STAGE": str(stage),
        "OMP_NUM_THREADS": "1",
    }
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
            process.wait(timeout=600)
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
    assert output.count("NATIVE_NVFP4_DEEPSPEED_LORA_MERGE_AWARE_PASS") == 2, output
