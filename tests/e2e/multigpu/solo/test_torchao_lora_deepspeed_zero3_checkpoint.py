"""Native NVFP4 ordinary-LoRA ZeRO-3 Trainer checkpoint regression."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _dynamic_nvfp4_supported():
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 2
        and all(
            torch.cuda.get_device_capability(device) >= (10, 0) for device in range(2)
        )
    )


@pytest.mark.parametrize(
    "dynamic_activation",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not _dynamic_nvfp4_supported(),
                reason="dynamic NVFP4 requires two SM100+ GPUs",
            ),
        ),
    ],
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA devices required")
def test_native_nvfp4_lora_deepspeed_zero3_checkpoint_resume(
    tmp_path, dynamic_activation
):
    env = os.environ | {
        "TORCHAO_LORA_DEEPSPEED_CHECKPOINT_TMP": str(tmp_path),
        "ZERO_STAGE": "3",
        "OMP_NUM_THREADS": "1",
        "ACCELERATE_DEEPSPEED_ZERO_STAGE": "3",
        "TORCHAO_LORA_DEEPSPEED_DYNAMIC": str(int(dynamic_activation)),
    }
    worker = str(
        Path(__file__).with_name("_torchao_lora_deepspeed_zero3_checkpoint.py")
    )
    for phase in ("reference", "resume"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                worker,
            ],
            env=env | {"TORCHAO_LORA_DEEPSPEED_PHASE": phase},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert result.stdout.count("TORCHAO_LORA_DEEPSPEED_CHECKPOINT_OK") == 2
