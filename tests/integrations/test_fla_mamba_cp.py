"""Optional distributed FLA Mamba adapter regression."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("lora", [False, True], ids=["full", "lora"])
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA FLA kernels")
def test_fla_mamba_packed_cp_four_ranks(lora):
    pytest.importorskip("fla")
    pytest.importorskip("ringmaster.fla_mamba")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "--module",
            "tests.integrations._fla_mamba_cp_probe",
        ],
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "RM_LORA": "1" if lora else "0",
            "PYTHONPATH": str(Path(__file__).resolve().parents[2])
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        },
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("max_parameter_relative_l2=") == 16
