"""Distributed ownership and optimizer parity for tied embedding adapters."""

import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


@pytest.mark.distributed_cpu
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
def test_tied_lora_distributed_meshes(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    worker = Path(__file__).with_name("_lora_tying_distributed.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=4",
            str(worker),
            str(tmp_path),
        ],
        env=env,
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
        pytest.fail("Distributed tying test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for layout in (
        "fsdp",
        "hsdp",
        "fsdp-cp",
        "fsdp-tp",
        "fsdp-ep",
        "hsdp-tp",
        "hsdp-cp",
    ):
        for policy in ("TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"):
            for dtype in ("torch.float32", "torch.bfloat16"):
                assert f"PASS {layout} {policy} {dtype}" in stdout
