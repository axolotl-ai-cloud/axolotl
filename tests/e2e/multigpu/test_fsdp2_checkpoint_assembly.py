"""FSDP2 full-checkpoint assembly over NCCL: chunks land in rank 0's slice views through the
pinned CPU staging buffer, bit-identical to the index assembly."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

WORKER = (
    Path(__file__).parents[2] / "monkeypatch" / "_fsdp2_assembly_worker.py"
).resolve()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_nccl_slice_assembly_matches_index_assembly():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(WORKER),
        ],
        env=os.environ | {"AXOLOTL_ASSEMBLY_DEVICE": "cuda", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS assembly-") == 15, result.stdout
