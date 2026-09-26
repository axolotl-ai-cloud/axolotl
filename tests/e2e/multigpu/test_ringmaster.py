"""Ringmaster on released Transformers/Accelerate, without pretrained downloads."""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import torch


@pytest.mark.slow
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize(
    "backend,inner",
    [
        ("ulysses", "sdpa"),
        ("ulysses", "flash_attention_2"),
        ("ring", "flash_attention_2"),
    ],
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_ringmaster_fsdp2_parity(backend, inner, packed):
    pytest.importorskip("ringmaster")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_ringmaster_parity.py")),
        ],
        env=os.environ
        | {
            "RM_BACKEND": backend,
            "RM_INNER": inner,
            "OMP_NUM_THREADS": "1",
            "RM_PACKED": "1" if packed else "0",
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == 4


@pytest.mark.slow
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("hub", [False, True], ids=["fallback", "hub"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_ringmaster_mamba_fsdp2_parity(hub, packed):
    pytest.importorskip("ringmaster")
    if hub:
        pytest.importorskip("kernels")
        from kernels import has_kernel

        try:
            available = has_kernel("kernels-community/mamba-ssm", version=2)
        except httpx.HTTPError as exc:
            pytest.skip(f"Mamba Hub kernel availability probe failed: {exc}")
        if not available:
            pytest.skip("Mamba Hub kernel is unavailable for this platform")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_ringmaster_parity.py")),
        ],
        env=os.environ
        | {
            "RM_MODEL": "mamba2",
            "RM_HUB": "1" if hub else "0",
            "RM_BACKEND": "ulysses",
            "RM_INNER": "sdpa",
            "USE_HUB_KERNELS": "1" if hub else "0",
            "OMP_NUM_THREADS": "1",
            "RM_PACKED": "1" if packed else "0",
        },
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == 4


@pytest.mark.slow
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("family", ["gdn", "kda", "kimi"])
def test_fla_cp_four_rank_parity(family, packed):
    pytest.importorskip("fla")
    pytest.importorskip("tilelang")
    shared = os.environ.get("AXOLOTL_FLA_CP_SHARED_GPU") == "1"
    if not torch.cuda.is_available() or (torch.cuda.device_count() < 4 and not shared):
        pytest.skip("requires four GPUs (or opt-in shared-GPU Gloo validation)")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "--module",
            "tests.e2e.multigpu._fla_cp_parity",
        ],
        env=os.environ
        | {
            "OMP_NUM_THREADS": "1",
            "RM_PACKED": "1" if packed else "0",
            "USE_HUB_KERNELS": "0",
            "RM_FLA_FAMILY": family,
            "RM_FLA_HYBRID": "1",
        },
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS FLA CP=4") == 4


@pytest.mark.slow
@pytest.mark.parametrize("family", ["mamba", "mamba2"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_ringmaster_fla_mamba_fsdp2_parity(family, packed):
    pytest.importorskip("fla")
    pytest.importorskip("ringmaster.fla_mamba")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_ringmaster_parity.py")),
        ],
        env=os.environ
        | {
            "RM_MODEL": family,
            "RM_FLA": "1",
            "RM_HUB": "1",
            "RM_BACKEND": "ulysses",
            "RM_INNER": "sdpa",
            "OMP_NUM_THREADS": "1",
            "RM_PACKED": "1" if packed else "0",
        },
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == 4
