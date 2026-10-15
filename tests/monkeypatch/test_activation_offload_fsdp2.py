import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize("layout", ["root", "leaf"])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices required")
def test_fsdp2_gathered_parameter_views_are_not_offloaded(layout):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(Path(__file__).with_name("_dist_activation_offload_fsdp2.py")),
        ],
        env={**os.environ, "OMP_NUM_THREADS": "1", "OFFLOAD_FSDP_LAYOUT": layout},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("FSDP2_LIVE_OFFLOAD_OK") == 2, output
