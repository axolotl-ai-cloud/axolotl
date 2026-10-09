"""Distributed compact-expert training and exact checkpoint continuation."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.parametrize(
    "case,devices",
    [
        ("fsdp2", 2),
        ("ep", 2),
        ("cp", 2),
        ("fsdp2_ep", 4),
        ("cp_ep", 4),
        ("hsdp", 4),
        ("hsdp_ep", 8),
        ("fsdp2_cp_ep", 8),
    ],
)
@pytest.mark.parametrize("kernel", ["eager", "scattermoe", "sonicmoe"])
def test_compact_expert_distributed_resume(tmp_path, case, devices, kernel):
    if torch.cuda.device_count() < devices:
        pytest.skip(f"requires {devices} CUDA GPUs")
    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ",".join(map(str, range(torch.cuda.device_count())))
    ).split(",")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("_moe_sieve.py"))],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": ",".join(visible[:devices]),
            "MOE_SIEVE_TEST_CASE": case,
            "MOE_SIEVE_EXPERT_KERNEL": kernel,
            "MOE_SIEVE_EMPTY_OWNER": "1",
            "MOE_SIEVE_TEST_OUTPUT": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
