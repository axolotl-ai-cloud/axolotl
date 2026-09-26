"""Two-rank FSDP2 gate for SonicMoE packed-NVFP4 expert LoRA."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


def test_sonicmoe_grouped_lora_dispatch_keeps_scaling_separate():
    from ._sonicmoe_nvfp4_fsdp2_worker import _grouped_lora

    gate = (object(), object(), 0.5)
    down = (object(), object(), 0.75)
    lora = {"gate_up_proj": gate, "down_proj": down}

    assert _grouped_lora(lora, "gate_up_proj") == (gate[:2], gate[2])
    assert _grouped_lora(lora, "down_proj") == (down[:2], down[2])


def test_sonicmoe_snapshot_forward_passes_grouped_factors_and_scaling(monkeypatch):
    import torch

    from . import _sonicmoe_nvfp4_fsdp2_worker as worker

    captured = {}

    def grouped(*args, **kwargs):
        captured["lora"] = args[7:9]
        captured["scaling"] = (kwargs["scaling1"], kwargs["scaling2"])
        return torch.zeros((4, 16), dtype=args[0].dtype)

    monkeypatch.setattr(worker, "grouped_moe_reference_forward", grouped)
    gate = (object(), object(), 0.5)
    down = (object(), object(), 0.75)
    worker._sonic_forward(torch.ones((4, 16)), object(), object(), gate, down)

    assert captured["lora"] == (gate[:2], down[:2])
    assert captured["scaling"] == (0.5, 0.75)


def test_sonicmoe_native_snapshot_owns_all_components():
    import torch

    from ._sonicmoe_nvfp4_fsdp2_worker import _clone_native_components

    qdata = torch.arange(8, dtype=torch.uint8)
    scale = torch.ones(2, dtype=torch.float8_e4m3fn)
    per_tensor_scale = torch.tensor([[[0.25]], [[0.5]]])
    snapshot = _clone_native_components(qdata, scale, per_tensor_scale)
    qdata.fill_(0)
    scale.fill_(0)
    per_tensor_scale.fill_(0)

    assert snapshot[0].data_ptr() != qdata.data_ptr()
    assert snapshot[1].data_ptr() != scale.data_ptr()
    assert snapshot[2].data_ptr() != per_tensor_scale.data_ptr()
    assert torch.equal(snapshot[0], torch.arange(8, dtype=torch.uint8))
    assert torch.equal(snapshot[2], torch.tensor([[[0.25]], [[0.5]]]))


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="requires CUDA")
def test_sonicmoe_nvfp4_packed_experts_fsdp2(tmp_path):
    torch = __import__("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA GPUs")
    if any(torch.cuda.get_device_capability(index)[0] < 10 for index in range(2)):
        pytest.skip("packed Sonic NVFP4 FSDP2 coverage requires two SM100+ GPUs")
    for package in ("peft", "torchao"):
        pytest.importorskip(package)

    worker = Path(__file__).with_name("_sonicmoe_nvfp4_fsdp2_worker.py")
    log_path = tmp_path / "worker.log"
    environment = os.environ | {"OMP_NUM_THREADS": "1"}
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
            env=environment,
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
    assert output.count("SONICMOE_NVFP4_FSDP2_PASS") == 2, output
