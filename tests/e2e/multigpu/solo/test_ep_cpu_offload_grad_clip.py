"""Four-rank CPU-only EP gradient-clipping regression."""

import os
import subprocess
import sys
from pathlib import Path


def test_ep_cpu_offload_gradient_clipping(tmp_path):
    worker = Path(__file__).with_name("_ep_cpu_offload_grad_clip_worker.py")
    log_path = tmp_path / "worker.log"
    env = os.environ | {"OMP_NUM_THREADS": "1"}
    with log_path.open("w") as log:
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                str(worker),
            ],
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=False,
        )
    output = log_path.read_text()
    assert process.returncode == 0, output
    assert output.count("EP_CPU_OFFLOAD_GRAD_CLIP_PASS") == 1, output
