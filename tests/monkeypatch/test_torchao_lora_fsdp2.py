"""Native TorchAO NVFP4 ordinary-LoRA FSDP2 checkpoint regression."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs required",
)
def test_native_nvfp4_lora_fsdp2_checkpoint_resume(tmp_path):
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    env = os.environ | {"TORCHAO_LORA_FSDP2_TMP": str(tmp_path), "OMP_NUM_THREADS": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_torchao_lora_fsdp2.py")),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=900,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert output.count("TORCHAO_LORA_FSDP2_OK rank=") == 2, output
