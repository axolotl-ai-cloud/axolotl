"""TRL's fused LM head trains like one process under axolotl's FSDP2 wrapping."""

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
def test_fused_lm_head_gradients_match_one_process():
    """Every layout keeps the head where its forward gathers it, including size-based
    wrapping, which would otherwise give the head a unit of its own."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    if sys.platform == "darwin":
        env["GLOO_SOCKET_IFNAME"] = "lo0"
    worker = Path(__file__).with_name("_fsdp2_fused_lm_head_distributed.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={port}",
            "--nproc-per-node=4",
            str(worker),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail("Distributed fused LM head test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    passed = {line for line in stdout.splitlines() if line.startswith("PASS")}
    for kind in ("untied", "tied", "modules_to_save"):
        for policy in ("TRANSFORMER_BASED_WRAP", "SIZE_BASED_WRAP"):
            layouts = ["fsdp", "hsdp", "fsdp-cp", "fsdp-ep"]
            if kind != "tied":
                layouts += ["fsdp-tp", "hsdp-tp"]
            for layout in layouts:
                assert f"PASS {kind} {layout} {policy}" in passed, stdout
