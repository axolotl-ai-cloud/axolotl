"""Checkpoint mirroring across two simulated nodes with real gloo collectives."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port

SHARE = [
    "rng_state_0.pth",
    "rng_state_1.pth",
    "rng_state_2.pth",
    "rng_state_3.pth",
    "trainer_state.json",
]


def _run(tmp_path, mode):
    worker = Path(__file__).with_name("_checkpoint_mirror_dist.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            str(tmp_path),
            mode,
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("checkpoint mirror worker timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return {
        rank: json.loads((tmp_path / f"report-{rank}.json").read_text())
        for rank in range(4)
    }


@pytest.mark.distributed_cpu
def test_node_local_output_dirs_combine_into_resumable_checkpoints(tmp_path):
    reports = _run(tmp_path, "node-local")

    assert [reports[r]["layout"] for r in range(4)] == [
        [True, 0, 2],
        [False, 0, 2],
        [True, 1, 2],
        [False, 1, 2],
    ]
    # HF rotated node-a; the mirror rotated node-b, which HF never touches
    assert reports[0]["local"] == reports[2]["local"] == [2, 3]
    assert reports[0]["mirror"] == {"checkpoint-2": SHARE, "checkpoint-3": SHARE}
    assert reports[0]["final"] == "final"
    assert reports[0]["staging"] == []
    # neither node's output_dir is complete, so every rank resumes from the mirror
    assert {reports[r]["resume"] for r in range(4)} == {"nfs/checkpoint-3"}


@pytest.mark.distributed_cpu
def test_shared_output_dir_uploads_once(tmp_path):
    reports = _run(tmp_path, "shared")

    assert [reports[r]["layout"] for r in range(4)] == [
        [True, 0, 1],
        [False, 0, 1],
        [False, 0, 1],
        [False, 0, 1],
    ]
    assert reports[0]["mirror"] == {"checkpoint-2": SHARE, "checkpoint-3": SHARE}
    assert {reports[r]["resume"] for r in range(4)} == {"shared/checkpoint-3"}
