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
        "sft-model-only-ep-adapter-resume",
        "final-ep-adapter-base-layout",
        "full-peft-state-dict",
        "restore-collectives-batched",
        "mixtral-final-original-layout",
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


@pytest.mark.parametrize("trainer_type", ["sft", "mixin"])
@pytest.mark.parametrize("save_only_model", [True, False])
@pytest.mark.parametrize("internal_call", [True, False])
def test_full_model_file_written_without_optimizer(
    tmp_path, trainer_type, save_only_model, internal_call
):
    from types import SimpleNamespace

    import torch
    from peft import LoraConfig, get_peft_model
    from torch import nn

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    trainer = object.__new__(
        AxolotlTrainer if trainer_type == "sft" else DistributedParallelMixin
    )
    trainer.args = SimpleNamespace(
        should_save=True, output_dir=str(tmp_path), save_only_model=save_only_model
    )
    trainer.processing_class = None
    trainer.data_collator = None
    trainer._axolotl_saving_checkpoint = internal_call
    base_model = nn.Module()
    base_model.dense = nn.Linear(2, 2)
    trainer.model = get_peft_model(
        base_model, LoraConfig(target_modules=["dense"], r=2)
    )
    trainer.model._moe_experts_quantized = True
    trainer.accelerator = SimpleNamespace(
        parallelism_config=None,
        is_main_process=True,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    state = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }
    trainer._save(str(tmp_path), state_dict=state)
    assert (tmp_path / "adapter_model.safetensors").is_file()
    if not (save_only_model and internal_call):
        assert not (tmp_path / "pytorch_model_fsdp.bin").exists()
        return
    saved = torch.load(tmp_path / "pytorch_model_fsdp.bin", weights_only=True)
    for name, value in state.items():
        torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
