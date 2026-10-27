"""Full model and 8-bit optimizer resumes must retain EP ownership."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from transformers.testing_utils import get_torch_dist_unique_port


def _run_worker(name, tmp_path):
    worker = Path(__file__).with_name(name)
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
        pytest.fail(f"Distributed worker {name} timed out:\n" + stdout + stderr)
    assert process.returncode == 0, stdout + stderr
    return stdout


@pytest.mark.distributed_cpu
def test_full_checkpoint_ownership_and_next_update(tmp_path):
    pytest.importorskip("torchao.optim")
    stdout = _run_worker("_fsdp2_full_checkpoint.py", tmp_path)
    for case in (
        "ep-full-parameter-save-route",
        "native-trainer-non-peft-full-checkpoint",
        "sft-model-only-ep-adapter-resume",
        "sft-regular-ep-adapter-resume",
        "final-ep-adapter-base-layout",
        "full-peft-state-dict",
        "non-ep-frozen-adapter-regular",
        "non-ep-frozen-adapter-model-only",
        "non-ep-frozen-adapter-final",
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
@pytest.mark.parametrize("adapters", [1, 2])
def test_full_model_file_written_without_optimizer(
    tmp_path, trainer_type, save_only_model, internal_call, adapters
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
    if adapters > 1:
        trainer.model.add_adapter("other", LoraConfig(target_modules=["dense"], r=2))
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
    if adapters == 1:
        assert saved == {"_axolotl_adapter_reference": 1}
        return
    for name, value in state.items():
        torch.testing.assert_close(saved[name], value, rtol=0, atol=0)


class _Group:
    """A process group with ``backends`` ({device type: backend name}) registered."""

    def __init__(self, backends):
        self.backends = backends

    def _get_backend(self, device):
        if device.type not in self.backends:
            raise RuntimeError(f"No backend type associated with device type {device}")
        name = self.backends[device.type]
        return type(f"ProcessGroup{name.capitalize()}", (), {})()


@pytest.mark.parametrize(
    "backend, config, registered, device",
    [
        ("nccl", "cuda:nccl", {"cuda": "nccl"}, "cuda"),
        ("cuda:nccl,cpu:gloo", "cuda:nccl,cpu:gloo", None, "cuda"),
        # the default group init_device_mesh creates when no backend was given
        ("undefined", "cuda:nccl", {"cuda": "nccl"}, "cuda"),
        ("undefined", None, {"cuda": "nccl"}, "cuda"),
        ("gloo", "cpu:gloo,cuda:gloo", None, "cpu"),
        ("gloo", None, {"cpu": "gloo", "cuda": "gloo"}, "cpu"),
        ("nccl", None, {}, "cuda"),
    ],
)
def test_transfer_stages_on_cuda_whenever_a_cuda_backend_is_registered(
    monkeypatch, backend, config, registered, device
):
    import torch

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    dist = fsdp2_checkpoint.dist

    def get_backend_config(group=None):
        if config is None:
            raise ValueError("no backend config")
        return config

    group = _Group(registered or {})
    monkeypatch.setattr(dist, "get_backend", lambda group=None: backend)
    monkeypatch.setattr(dist, "get_backend_config", get_backend_config)
    monkeypatch.setattr(dist.distributed_c10d, "_get_default_group", lambda: group)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert fsdp2_checkpoint.collective_device(group).type == device
    assert fsdp2_checkpoint._transfer_device().type == device


def test_transfer_stages_on_cpu_over_a_gloo_group(tmp_path):
    import torch.distributed as dist

    from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import _transfer_device

    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        assert _transfer_device().type == "cpu"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 1])
def test_transfer_to_self_skips_the_collective(rank):
    from unittest.mock import patch

    import torch

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    with (
        patch.object(fsdp2_checkpoint.dist, "get_rank", return_value=rank),
        patch.object(
            fsdp2_checkpoint.dist,
            "broadcast",
            side_effect=AssertionError("broadcast to self"),
        ),
    ):
        out = fsdp2_checkpoint._transfer(
            value if rank == 0 else None, [2, 3], torch.bfloat16, source=0
        )
    if rank == 0:
        assert out.device.type == "cpu" and out.dtype == torch.bfloat16
        torch.testing.assert_close(out, value.to(torch.bfloat16))
        assert out.data_ptr() != value.data_ptr()
    else:
        assert out is None


@pytest.mark.distributed_cpu
def test_slice_assembly_matches_index_assembly(tmp_path):
    stdout = _run_worker("_fsdp2_assembly_worker.py", tmp_path)
    for case in (
        "shard0-uneven",
        "shard1-uneven",
        "shard0-empty-ranks",
        "shard0-bf16",
        "2d-shard0-shard1",
        "2d-shard0-shard0",
        "kind-a",
        "kind-a-uneven",
        "kind-b-ep4",
        "kind-b-dp-uneven",
        "kind-expert",
        "strided-index-fallback",
        "strided-contiguous-rows",
        "gap-rejected",
        "overlap-rejected",
    ):
        assert f"PASS assembly-{case}" in stdout, stdout


@pytest.mark.parametrize(
    "shape, start, stop",
    [
        ((5, 7), 0, 35),
        ((5, 7), 3, 4),
        ((5, 7), 3, 30),
        ((5, 7), 7, 21),
        ((3, 4, 5), 6, 53),
        ((3, 4, 5), 21, 39),
        ((), 0, 1),
        ((9,), 2, 7),
    ],
)
def test_write_fills_a_flat_range_of_a_strided_view(shape, start, stop):
    import torch

    from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import _write

    base = torch.zeros((2, *shape, 3))
    out = base[1, ..., 2]
    expected = torch.zeros(shape).reshape(-1)
    data = torch.arange(1.0, stop - start + 1)
    expected[start:stop] = data
    _write(out, start, data)
    assert torch.equal(out.reshape(-1), expected)
    assert base[0].abs().sum() == 0 and base[1, ..., :2].abs().sum() == 0


def test_copy_threads_raise_and_restore(monkeypatch):
    import torch

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        monkeypatch.setattr(fsdp2_checkpoint, "_MAX_COPY_THREADS", 3)
        monkeypatch.setattr(
            fsdp2_checkpoint.os,
            "sched_getaffinity",
            lambda pid: {0, 1, 2, 3},
            raising=False,
        )
        with fsdp2_checkpoint._copy_threads():
            assert torch.get_num_threads() == 3
        assert torch.get_num_threads() == 1
        torch.set_num_threads(8)
        with fsdp2_checkpoint._copy_threads():
            assert torch.get_num_threads() == 8  # never lowers
    finally:
        torch.set_num_threads(before)


def test_only_the_receiver_raises_copy_threads():
    from unittest.mock import patch

    import torch

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    seen = []
    real = fsdp2_checkpoint._transfer_chunks

    def spy(*args):
        seen.append(torch.get_num_threads())
        return real(*args)

    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (
            patch.object(fsdp2_checkpoint, "_transfer_chunks", spy),
            patch.object(fsdp2_checkpoint, "_MAX_COPY_THREADS", 2),
            patch.object(fsdp2_checkpoint.dist, "get_rank", return_value=0),
            patch.object(
                fsdp2_checkpoint.os,
                "sched_getaffinity",
                lambda pid: {0, 1, 2, 3},
                create=True,
            ),
        ):
            out = fsdp2_checkpoint._transfer(
                torch.ones(4), [4], torch.float32, source=0
            )
        assert torch.equal(out, torch.ones(4))
        assert seen == [2]
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(before)


def test_transfer_writes_into_the_destination_view():
    from unittest.mock import patch

    import torch

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    full = torch.zeros(6, 8)
    destination = full[1:4, 2:7]
    value = torch.arange(15.0).reshape(3, 5)
    with (
        patch.object(fsdp2_checkpoint.dist, "get_rank", return_value=0),
        patch.object(fsdp2_checkpoint, "_TRANSFER_BYTES", 16),
    ):
        out = fsdp2_checkpoint._transfer(
            value, [3, 5], torch.float32, source=0, out=destination
        )
    assert out is destination
    assert torch.equal(full[1:4, 2:7], value)
    assert full.sum() == value.sum()


@pytest.mark.parametrize(
    "case, expected",
    [
        ("lora", False),
        ("lora-embedding", True),
        ("lora-embedding-trainable-tokens", False),
        ("modules-to-save-embedding", False),
        ("unfrozen-embedding", True),
        ("resized-local-base", True),
        ("hub-base-not-cached", False),
    ],
)
def test_save_embedding_layers_is_explicit_without_hub_requests(
    tmp_path, case, expected
):
    from unittest.mock import patch

    from peft import LoraConfig, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from transformers import GPT2Config, GPT2LMHeadModel

    from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
        peft_save_embedding_layers,
    )

    config = GPT2Config(vocab_size=32, n_embd=8, n_head=2, n_layer=1)
    config.save_pretrained(tmp_path)
    base = GPT2LMHeadModel(config)
    base.name_or_path = (
        "axolotl-ai-co/not-a-cached-model" if case == "hub-base-not-cached" else ""
    )
    if case == "resized-local-base":
        base.resize_token_embeddings(40)
        base.name_or_path = str(tmp_path)
    lora = dict(target_modules=["c_attn"], r=2)
    if case == "lora-embedding":
        lora["target_modules"] = ["c_attn", "wte"]
    if case == "lora-embedding-trainable-tokens":
        lora["target_modules"] = ["c_attn", "lm_head"]
        lora["trainable_token_indices"] = {"wte": [1, 2]}
    if case == "modules-to-save-embedding":
        lora["modules_to_save"] = ["wte"]
    model = get_peft_model(base, LoraConfig(**lora))
    if case == "unfrozen-embedding":
        model.base_model.model.transformer.wte.weight.requires_grad_(True)
    with patch(
        "peft.utils.save_and_load.check_file_exists_on_hf_hub",
        side_effect=AssertionError("Hub request during save"),
    ):
        flag = peft_save_embedding_layers(model)
        get_peft_model_state_dict(model, save_embedding_layers=flag)
        if case == "hub-base-not-cached":
            with pytest.raises(AssertionError, match="Hub request"):
                get_peft_model_state_dict(model)
    assert flag is expected
