"""CPU (gloo) multi-process coverage of per-tensor ratio clipping under FSDP2, TP and EP."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKER = Path(__file__).with_name("_grad_norm_guard_worker.py")


def _run(tmp_path, mode, nproc):
    log_path = tmp_path / f"{mode}-{nproc}.log"
    src = str(Path(__file__).resolve().parents[3] / "src")
    env = os.environ | {
        "OMP_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [src, os.environ.get("PYTHONPATH")])
        ),
    }
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
            timeout=300,
            check=False,
        )
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert output.count(f"GRAD_NORM_GUARD_{mode.upper()}_PASS") == 1, output


@pytest.mark.distributed_cpu
@pytest.mark.parametrize("nproc", [2, 4])
def test_ratio_clip_fsdp2_matches_unsharded_reference(tmp_path, nproc):
    _run(tmp_path, "fsdp2", nproc)


@pytest.mark.distributed_cpu
def test_ratio_clip_hsdp_and_tensor_parallel_layouts(tmp_path):
    _run(tmp_path, "hsdp_tp", 4)


@pytest.mark.distributed_cpu
def test_ratio_clip_per_expert_matches_across_ep_sizes(tmp_path):
    _run(tmp_path, "ep", 4)


@pytest.mark.distributed_cpu
def test_ratio_clip_averages_resume_through_fsdp2_optimizer_checkpoints(tmp_path):
    _run(tmp_path, "resume", 2)


@pytest.mark.distributed_cpu
def test_ratio_clip_expert_averages_resume_for_every_ep_rank(tmp_path):
    _run(tmp_path, "ep_resume", 4)
