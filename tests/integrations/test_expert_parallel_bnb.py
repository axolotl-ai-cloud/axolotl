"""Packed fused-expert weights and scales must retain EP ownership."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers.testing_utils import get_torch_dist_unique_port


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_packed_experts_preserve_weights_and_ownership(device):
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < 2
    ):
        pytest.skip("Requires two CUDA devices")
    pytest.importorskip("bitsandbytes.nn.parametrize")
    worker = Path(__file__).parents[1] / "monkeypatch" / "_expert_parallel_bnb.py"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=2",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            device,
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Distributed packed-expert test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for case in (
        "nested-aligned",
        "nested-scale-cut",
        "uncompressed",
        "rank-zero-materialized",
    ):
        assert f"PASS {case}" in stdout
