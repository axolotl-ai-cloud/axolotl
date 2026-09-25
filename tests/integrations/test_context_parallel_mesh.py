"""Ensure USP subgroups never span different data-parallel samples."""

import os
import subprocess
import sys

import pytest


@pytest.mark.slow
def test_usp_mesh_preserves_data_parallel_groups():
    pytest.importorskip("ringmaster")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "--module",
            "tests.integrations._ringmaster_mesh_probe",
        ],
        env=os.environ | {"OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS mesh rank") == 8


@pytest.mark.slow
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_mamba_four_rank_forward_backward(packed):
    pytest.importorskip("ringmaster")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=4",
            "--module",
            "tests.integrations._mamba_cp_parity",
        ],
        env=os.environ
        | {
            "OMP_NUM_THREADS": "1",
            "USE_HUB_KERNELS": "0",
            "RM_PACKED": "1" if packed else "0",
        },
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS Mamba CP=4") == 4


@pytest.mark.slow
def test_ringmaster_nd_cpu_parity():
    pytest.importorskip("ringmaster")
    torch = pytest.importorskip("torch")
    if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 13):
        pytest.skip("CPU FSDP2 parity requires torch >= 2.13")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=16",
            "--module",
            "tests.integrations._ringmaster_nd_probe",
        ],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS ND rank=") == 128
