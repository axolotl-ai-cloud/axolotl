"""Per-tensor / per-expert ratio clipping on NCCL: EP (incl. pure EP), EP x FSDP, TP layouts."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

WORKER = (
    Path(__file__).resolve().parents[3]
    / "core"
    / "trainers"
    / "_grad_norm_guard_worker.py"
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason="requires four CUDA GPUs",
)
@pytest.mark.parametrize(
    "mode,nproc",
    [("ep", 4), ("ep_resume", 4), ("fsdp2", 4), ("resume", 2), ("hsdp_tp", 4)],
)
def test_grad_norm_ratio_nccl(tmp_path, mode, nproc):
    src = str(Path(__file__).resolve().parents[4] / "src")
    env = os.environ | {
        "GRAD_NORM_GUARD_DEVICE": "cuda",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [src, os.environ.get("PYTHONPATH")])
        ),
    }
    log_path = tmp_path / f"{mode}.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={nproc}",
                str(WORKER),
                mode,
            ],
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False,
        )
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert output.count(f"GRAD_NORM_GUARD_{mode.upper()}_PASS") == 1, output
