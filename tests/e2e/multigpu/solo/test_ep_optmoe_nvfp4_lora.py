"""optmoe ``nvfp4_lora`` (frozen packed NVFP4 experts + PEFT expert LoRA on the SM100
persistent kernels) against the SonicMoE merge-aware path under EP2 x FSDP2."""

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.nvfp4, pytest.mark.gpu]


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires at least two CUDA GPUs",
)
def test_optmoe_nvfp4_lora_matches_sonicmoe_merge_aware(tmp_path):
    world = 4 if torch.cuda.device_count() >= 4 else 2
    if any(torch.cuda.get_device_capability(i)[0] < 10 for i in range(world)):
        pytest.skip("nvfp4_lora needs SM100 GPUs")
    if find_spec("optmoe") is None or find_spec("optmoe_deepgemm") is None:
        pytest.skip("requires the optmoe package and its kernel extension")
    pytest.importorskip("torchao.prototype.mx_formats.nvfp4_tensor")
    root = Path(__file__).resolve().parents[4]
    worker = Path(__file__).with_name("_ep_optmoe_nvfp4_lora_worker.py")
    log = tmp_path / "worker.log"
    with log.open("w") as handle:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc-per-node={world}",
                str(worker),
            ],
            env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "1"},
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1500,
            check=False,
        )
    text = log.read_text()
    assert result.returncode == 0, text[-20000:]
    assert '"max_adapter_grad_relative_l2"' in text, text[-5000:]
