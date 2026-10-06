"""Native full checkpoints must preserve packed BNB weights and metadata."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from transformers.testing_utils import get_torch_dist_unique_port


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.distributed_cpu),
        pytest.param("cuda", marks=pytest.mark.gpu),
    ],
)
def test_native_packed_bnb_full_checkpoint(device, tmp_path):
    pytest.importorskip("bitsandbytes.nn.parametrize")
    if device == "cuda" and torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices")
    worker = Path(__file__).with_name("_fsdp2_bnb_checkpoint.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc-per-node={4 if device == 'cpu' else 2}",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
            device,
            str(tmp_path),
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
        pytest.fail("Distributed packed checkpoint test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for case in (
        "nested-nf4",
        "nested-scale-cut",
        "uncompressed-fp4",
        "reordered-owners",
        "packed-int8",
        "dp-shards-to-replicas",
        "real-peft-native-routes",
        "rejected-missing-owner",
        "rejected-changed-ep-grouping",
        "rejected-malformed-codebook",
        "rejected-expert-logical-shape",
        "rejected-dense-logical-shape",
        "rejected-legacy-packed",
        "rejected-dense-int8",
    ):
        assert f"PASS {case}" in stdout
