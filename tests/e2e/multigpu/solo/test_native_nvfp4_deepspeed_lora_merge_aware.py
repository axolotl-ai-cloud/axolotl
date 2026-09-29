"""Two-rank DeepSpeed native-NVFP4 merge-aware LoRA validation."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _require_dynamic_nvfp4():
    if any(torch.cuda.get_device_capability(index)[0] < 10 for index in range(2)):
        pytest.skip("dynamic NVFP4 requires two SM100+ GPUs")


def _run_worker(tmp_path, stage, dynamic, *, opt_out=False):
    if dynamic:
        _require_dynamic_nvfp4()
    pytest.importorskip("deepspeed")
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        pytest.skip("TorchAO NVFP4 unavailable")
    if NVFP4Tensor is None:
        pytest.skip("TorchAO NVFP4 unavailable")
    worker = Path(__file__).with_name("_native_nvfp4_deepspeed_lora_merge_aware.py")
    log_path = tmp_path / f"worker-zero{stage}-optout{int(opt_out)}.log"
    env = os.environ | {
        "NATIVE_NVFP4_DEEPSPEED_LORA_TMP": str(tmp_path),
        "ZERO_STAGE": str(stage),
        "NATIVE_NVFP4_DEEPSPEED_DYNAMIC": str(int(dynamic)),
        "NATIVE_NVFP4_DEEPSPEED_OPT_OUT": str(int(opt_out)),
        "OMP_NUM_THREADS": "1",
    }
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
            process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    output = log_path.read_text()
    assert not timed_out, output
    assert process.returncode == 0, output
    assert output.count("NATIVE_NVFP4_DEEPSPEED_LORA_MERGE_AWARE_PASS") == 2, output


@pytest.mark.parametrize("stage", [1, 2, 3])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs required",
)
def test_native_nvfp4_deepspeed_lora_merge_aware(tmp_path, stage, dynamic):
    _run_worker(tmp_path, stage, dynamic)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="two CUDA GPUs required",
)
def test_native_nvfp4_deepspeed_dynamic_lora_opt_out(tmp_path, stage):
    _run_worker(tmp_path, stage, True, opt_out=True)
