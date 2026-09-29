"""Two-rank native-NVFP4 merge-aware FSDP2 trainer resume coverage."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("cpu_ram_efficient", "cpu_offload"),
    [
        pytest.param(False, False, id="normal"),
        pytest.param(True, False, id="cpu-ram-efficient"),
        pytest.param(False, True, id="cpu-offload"),
    ],
)
@pytest.mark.parametrize("dynamic_activation", [False, True])
@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_native_nvfp4_fsdp2_merge_aware_checkpoint_resume(
    tmp_path, cpu_ram_efficient, cpu_offload, dynamic_activation
):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    if dynamic_activation and any(
        torch.cuda.get_device_capability(index)[0] < 10 for index in range(2)
    ):
        pytest.skip("dynamic NVFP4 requires two SM100+ GPUs")

    worker = Path(__file__).with_name(
        "_native_nvfp4_fsdp2_merge_aware_resume_worker.py"
    )
    log_path = tmp_path / "worker.log"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
    env["NVFP4_FSDP2_RESUME_ROOT"] = str(tmp_path)
    env["NVFP4_FSDP2_CPU_RAM_EFFICIENT"] = str(int(cpu_ram_efficient))
    env["NVFP4_FSDP2_CPU_OFFLOAD"] = str(int(cpu_offload))
    env["NVFP4_FSDP2_DYNAMIC_ACTIVATION"] = str(int(dynamic_activation))
    env["OMP_NUM_THREADS"] = "1"

    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                str(worker),
            ],
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    output = log_path.read_text()
    assert not timed_out, output
    assert process.returncode == 0, output
    assert output.count("NATIVE_NVFP4_FSDP2_MERGE_AWARE_RESUME_PASS") == 2, output
