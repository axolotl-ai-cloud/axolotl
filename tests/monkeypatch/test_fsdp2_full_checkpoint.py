"""Full model and 8-bit optimizer resumes must retain EP ownership."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port


def test_full_checkpoint_ownership_and_next_update(tmp_path):
    pytest.importorskip("torchao.optim")
    worker = Path(__file__).with_name("_fsdp2_full_checkpoint.py")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc-per-node=4",
            f"--master-port={get_torch_dist_unique_port()}",
            str(worker),
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
        pytest.fail("Distributed checkpoint test timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    for case in (
        "ep-full-parameter-save-route",
        "native-trainer-non-peft-full-checkpoint",
        "ep-dp",
        "ep-permuted-ranks",
        "ep-cp",
        "ep-hsdp",
        "ep-dp-to-hsdp",
        "quantized-to-small-fp32-shards",
        "tp-cross-global-blocks-same-layout",
        "strided-dp-tp",
        "incompatible-restore-rejected",
    ):
        assert f"PASS {case}" in stdout


def test_full_model_file_written_without_optimizer(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn
    from transformers import Trainer

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    trainer = object.__new__(DistributedParallelMixin)
    trainer.args = SimpleNamespace(
        should_save=True, output_dir=str(tmp_path), save_only_model=True
    )
    base_model = nn.Module()
    base_model.dense = nn.Linear(2, 2)
    trainer.model = get_peft_model(
        base_model, LoraConfig(target_modules=["dense"], r=2)
    )
    trainer.model._moe_experts_quantized = True
    trainer.accelerator = SimpleNamespace(
        parallelism_config=None,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    monkeypatch.setattr(Trainer, "_save", lambda *args, **kwargs: None)
    state = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    trainer._save(str(tmp_path), state_dict=state)
    saved = torch.load(tmp_path / "pytorch_model_fsdp.bin", weights_only=True)
    for name, value in state.items():
        torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
