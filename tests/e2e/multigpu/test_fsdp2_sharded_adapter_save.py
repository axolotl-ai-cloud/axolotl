"""Sharded FSDP2 adapter checkpoints over NCCL: shard writes stage through pinned CPU memory
and consolidate into the gathered adapter file's exact bytes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

WORKER = (
    Path(__file__).parents[2] / "monkeypatch" / "_fsdp2_sharded_adapter_worker.py"
).resolve()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_nccl_sharded_adapter_matches_gathered_adapter(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            str(WORKER),
            str(tmp_path),
        ],
        env=os.environ
        | {"AXOLOTL_SHARDED_ADAPTER_DEVICE": "cuda", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS sharded-") == 6, result.stdout
