"""Distributed native TorchAO NVFP4 ordinary-LoRA regression."""

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
def test_native_nvfp4_lora_ddp(tmp_path):
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        pytest.skip("TorchAO NVFP4 unavailable")
    if NVFP4Tensor is None:
        pytest.skip("TorchAO NVFP4 unavailable")
    env = os.environ | {"TORCHAO_LORA_DDP_TMP": str(tmp_path), "OMP_NUM_THREADS": "1"}
    worker = Path(__file__).with_name("_torchao_lora_ddp.py")
    result = subprocess.run(
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
        capture_output=True,
        check=False,
        timeout=600,
    )
    output = result.stdout + "\n" + result.stderr
    assert result.returncode == 0, output
    assert output.count("TORCHAO_LORA_DDP_OK rank=") == 2, output
