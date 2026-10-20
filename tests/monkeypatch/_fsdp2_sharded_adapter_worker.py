"""Distributed worker: sharded adapter checkpoints consolidate into the same bytes as the gathered
``adapter_model.safetensors``, for FSDP2-only and FSDP2 + EP (interleaved expert LoRA) layouts."""

import filecmp
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import _StridedShard

from axolotl.monkeypatch.accelerate import fsdp2_sharded_adapter
from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _check_errors,
    full_model_state,
)
from axolotl.monkeypatch.accelerate.fsdp2_sharded_adapter import (
    CONSOLIDATIONS,
    SHARD_DIRNAME,
    consolidate,
    write_adapter_shards,
)
from axolotl.utils.dict import DictDefault

# "cuda" runs the NCCL path (pinned staging for the shard writes) on 4 GPUs
DEVICE = os.environ.get("AXOLOTL_SHARDED_ADAPTER_DEVICE", "cpu")


def on(tensor):
    return tensor.to(DEVICE, copy=True)


def make_trainer(model, output_dir, adapter_format=None):
    from axolotl.core.trainers.base import AxolotlTrainer

    trainer = object.__new__(AxolotlTrainer)
    trainer.model = model
    trainer.accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0,
        num_processes=dist.get_world_size(),
        wait_for_everyone=dist.barrier,
        unwrap_model=lambda model, **kwargs: model,
        parallelism_config=None,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    trainer.is_fsdp_enabled = True
    trainer.is_deepspeed_enabled = False
    trainer.processing_class = None
    trainer.data_collator = None
    trainer.args = SimpleNamespace(
        output_dir=str(output_dir),
        should_save=dist.get_rank() == 0,
        save_only_model=True,
        push_to_hub=False,
    )
    cfg = {"adapter": "lora"}
    if adapter_format is not None:
        cfg["adapter_checkpoint_format"] = adapter_format
    trainer._axolotl_cfg = DictDefault(cfg)
    return trainer


def fill(model, seed_of):
    """Distinct values per element, identical on every replica of a slice."""
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            local = (
                parameter.to_local() if isinstance(parameter, DTensor) else parameter
            )
            generator = torch.Generator().manual_seed(seed_of(name))
            local.copy_(torch.randn(local.shape, generator=generator).to(local.dtype))


class Spy:
    """Count calls into the sharded save's cross-rank paths."""

    def __init__(self, *names):
        self.calls = dict.fromkeys(names, 0)
        self.patches = [
            patch.object(fsdp2_sharded_adapter, name, self.wrap(name)) for name in names
        ]

    def wrap(self, name):
        original = getattr(fsdp2_sharded_adapter, name)

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return original(*args, **kwargs)

        return counted

    def __enter__(self):
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in self.patches:
            item.stop()


def files(directory):
    return sorted(
        str(path.relative_to(directory))
        for path in Path(directory).rglob("*")
        if path.is_file()
    )


def expect_identical(label, gathered, sharded):
    if dist.get_rank() == 0:
        CONSOLIDATIONS.wait()
        assert not (sharded / SHARD_DIRNAME).exists(), label
        assert files(gathered) == files(sharded), (
            label,
            files(gathered),
            files(sharded),
        )
        a = gathered / "adapter_model.safetensors"
        b = sharded / "adapter_model.safetensors"
        assert filecmp.cmp(a, b, shallow=False), label
        from safetensors.torch import load_file

        left, right = load_file(a), load_file(b)
        assert left.keys() == right.keys() and left, label
        for key, value in left.items():
            assert value.dtype == right[key].dtype, (label, key)
            assert value.shape == right[key].shape, (label, key)
            assert torch.equal(
                value.view(-1).view(torch.uint8), right[key].view(-1).view(torch.uint8)
            )
        for name in ("adapter_config.json",):
            assert filecmp.cmp(gathered / name, sharded / name, shallow=False), label
        print(f"PASS sharded-{label}", flush=True)
    dist.barrier()


def fsdp_only_model(flat, mesh):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(7)
    base = nn.Module()
    base.embed = nn.Embedding(29, 12)
    base.q = nn.Linear(12, 40, bias=False)
    base.k = nn.Linear(40, 13, bias=False)
    base.v = nn.Linear(13, 16, bias=False)
    base.o = nn.Linear(16, 12, bias=False)
    model = get_peft_model(
        base, LoraConfig(target_modules=["embed", "q", "k", "v", "o"], r=6)
    )
    placements = {
        "q.lora_B": (flat, (Shard(0),)),  # 40 rows over 4 ranks
        "k.lora_B": (flat, (Shard(0),)),  # 13 rows: uneven
        "k.lora_A": (flat, (Shard(1),)),  # column shards
        "v.lora_B": (
            mesh,
            (Shard(0), _StridedShard(0, split_factor=2)),
        ),  # index fallback
        "o.lora_A": (flat, (Replicate(),)),
        "embed.lora_embedding_A": (flat, (Shard(1),)),
    }
    for module_name, module in model.named_modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if not parameter.requires_grad:
                continue
            full = f"{module_name}.{name}"
            target, placement = next(
                (value for key, value in placements.items() if key in full),
                (flat, (Shard(0),)),
            )
            value = parameter.detach().clone()
            if "o.lora_B" in full:
                value = value.to(torch.bfloat16)
            setattr(
                module,
                name,
                nn.Parameter(distribute_tensor(on(value), target, placement)),
            )
    fill(model, lambda name: sum(map(ord, name)))
    return model


def check_fsdp_only(flat, mesh, root):
    model = fsdp_only_model(flat, mesh)
    gathered, sharded = root / "fsdp-gather", root / "fsdp-sharded"
    reference = make_trainer(model, gathered)
    state = full_model_state(model, adapter_only=True)
    if dist.get_rank() == 0:
        reference._save(str(gathered), state_dict=state)
    dist.barrier()
    with Spy("_transfer", "_gather_tensor") as spy:
        make_trainer(model, sharded, "sharded").save_model(_internal_call=True)
    # only the _StridedShard factor needs rank 0 to assemble it
    assert spy.calls == {"_transfer": 0, "_gather_tensor": 1}, spy.calls
    expect_identical("fsdp-only", gathered, sharded)
    return model


def slice_cpu_experts(model, mesh):
    from axolotl.integrations.expert_parallel.shard import _detect_experts_modules

    for _, module in _detect_experts_modules(model):
        total = module.num_experts
        local = total // mesh["ep"].size()
        offset = mesh.get_coordinate()[1] * local
        for name in ("gate_up_proj", "down_proj"):
            parameter = getattr(module, name)
            setattr(
                module,
                name,
                nn.Parameter(parameter[offset : offset + local].detach().clone()),
            )
        module.num_experts_global = total
        module.num_local_experts = local
        module.local_expert_offset = offset
        module.num_experts = local


def ep_model(mesh):
    from peft import LoraConfig, get_peft_model

    from axolotl.integrations.expert_parallel.shard import shard_expert_lora

    torch.manual_seed(456)
    base = nn.Module()
    base.experts = nn.Module()
    base.experts.num_experts = 8
    base.experts.gate_up_proj = nn.Parameter(torch.ones(8, 24, 20))
    base.experts.down_proj = nn.Parameter(torch.ones(8, 20, 12))
    base.dense = nn.Linear(20, 33, bias=False)
    base.register_buffer("counter", torch.tensor(3))
    slice_cpu_experts(base, mesh)
    model = get_peft_model(
        base,
        LoraConfig(
            target_modules=["dense"],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            r=4,
        ),
    )
    assert shard_expert_lora(model, 2) == 4
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.requires_grad:
                setattr(
                    module,
                    name,
                    nn.Parameter(
                        distribute_tensor(
                            on(parameter.detach()), mesh["dp"], (Shard(0),)
                        )
                    ),
                )
    dp_rank, ep_rank = mesh.get_coordinate()

    def seed(name):
        base = sum(map(ord, name)) * 16 + dp_rank
        return base + 4 * ep_rank if "experts" in name else base

    fill(model, seed)
    return model


def check_ep(mesh, root):
    model = ep_model(mesh)
    gathered, sharded = root / "ep-gather", root / "ep-sharded"
    # the default format is the gather path
    make_trainer(model, gathered).save_model(_internal_call=True)
    dist.barrier()
    shard_files = []

    def counting(shard_dir, *args, **kwargs):
        shard_files.append(len(list(Path(shard_dir).glob("shard-*.safetensors"))))
        return consolidate(shard_dir, *args, **kwargs)

    with (
        Spy("_transfer", "_gather_tensor") as spy,
        patch.object(fsdp2_sharded_adapter, "consolidate", counting),
    ):
        make_trainer(model, sharded, "sharded").save_model(_internal_call=True)
        if dist.get_rank() == 0:
            CONSOLIDATIONS.wait()
    assert spy.calls == {"_transfer": 0, "_gather_tensor": 0}, spy.calls
    # every rank holds its own experts' factors
    assert shard_files == ([4] if dist.get_rank() == 0 else []), shard_files
    expect_identical("ep", gathered, sharded)
    if dist.get_rank() == 0:
        from safetensors.torch import load_file

        exported = load_file(sharded / "adapter_model.safetensors")
        # interleaved expert lora_B [out, r * E] and lora_A [r * E, in] at full EP size
        assert any(v.shape == (24, 32) for k, v in exported.items() if "lora_B" in k)
        assert any(v.shape == (32, 20) for k, v in exported.items() if "lora_A" in k)
        assert torch.load(sharded / "pytorch_model_fsdp.bin", weights_only=True) == {
            "_axolotl_adapter_reference": 1
        }
    dist.barrier()

    remote = root / "ep-remote"
    original = fsdp2_sharded_adapter._probe

    def other_node(shard_dir):
        visible = original(shard_dir)
        return [seen and rank < 2 for rank, seen in enumerate(visible)]

    with (
        patch.object(fsdp2_sharded_adapter, "_probe", other_node),
        Spy("_transfer") as spy,
    ):
        make_trainer(model, remote, "sharded").save_model(_internal_call=True)
    # every unique slice ranks 2 and 3 hold travels to rank 0; replicas stay put
    assert spy.calls["_transfer"] > 0, spy.calls
    expect_identical("ep-remote-ranks", gathered, remote)
    return model


def check_regular_checkpoint_resume(model, root):
    """save_only_model: false: the FSDP model file references the consolidating adapter, and
    a resume reads it back without a second gather."""
    from accelerate.utils import fsdp_utils

    from axolotl.monkeypatch.accelerate import fsdp2_checkpoint

    fsdp2_checkpoint.patch_fsdp2_full_checkpoint()
    directory = root / "ep-regular"
    trainer = make_trainer(model, directory, "sharded")
    trainer.args.save_only_model = False
    expected = {
        name: parameter.to_local().detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    def no_gather(*args, **kwargs):
        raise AssertionError("gathered the adapter for the FSDP model file")

    with patch.object(fsdp2_checkpoint, "full_model_state", no_gather):
        trainer.save_model(_internal_call=True)
        fsdp_utils.save_fsdp_model(
            trainer.accelerator.state.fsdp_plugin,
            trainer.accelerator,
            model,
            directory,
            adapter_only=True,
        )
    if dist.get_rank() == 0:
        assert torch.load(directory / "pytorch_model_fsdp.bin", weights_only=True) == {
            "_axolotl_adapter_reference": 1
        }
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.to_local().zero_()
    with patch.object(
        model, "load_adapter", side_effect=AssertionError("Used local PEFT restore")
    ):
        trainer._load_from_checkpoint(str(directory))
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter.to_local(), expected[name]), name
    if dist.get_rank() == 0:
        assert not (directory / SHARD_DIRNAME).exists()
        print("PASS sharded-regular-checkpoint-resume", flush=True)
    dist.barrier()


def check_torch_consolidation(model, root):
    from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        consolidate_safetensors_files,
    )

    directory = root / "torch-interop"
    shards = write_adapter_shards(model, directory)
    if dist.get_rank() == 0:
        import json

        from safetensors.torch import load_file

        manifest = json.loads((shards.directory / "manifest.json").read_text())
        copy = root / "torch-interop-copy"
        shutil.copytree(shards.directory, copy)
        out = root / "torch-interop-out"
        out.mkdir()
        mapping = dict.fromkeys((t["name"] for t in manifest["tensors"]), 1)
        consolidate_safetensors_files(str(copy), str(out), mapping, num_threads=2)
        path, _, _ = consolidate(shards.directory)
        ours = load_file(path)
        theirs = load_file(next(out.glob("*.safetensors")))
        assert ours.keys() == theirs.keys()
        for key, value in ours.items():
            assert torch.equal(theirs[key].reshape(value.shape), value), key
        print("PASS sharded-torch-dcp-consolidation", flush=True)
    dist.barrier()


def check_interrupted_resume(model, root):
    directory = root / "interrupted"
    write_adapter_shards(model, directory)
    error = None
    if dist.get_rank() == 0:
        try:
            assert not (directory / "adapter_model.safetensors").exists()
            assert fsdp2_sharded_adapter.complete_interrupted_consolidation(directory)
            assert (directory / "adapter_model.safetensors").is_file()
            assert not (directory / SHARD_DIRNAME).exists()
            assert not fsdp2_sharded_adapter.complete_interrupted_consolidation(
                directory
            )
        except AssertionError as exc:
            error = f"interrupted consolidation: {exc!r}"
    _check_errors(error)
    if dist.get_rank() == 0:
        print("PASS sharded-interrupted-consolidation", flush=True)


def main():
    root = Path(sys.argv[1])
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("cuda:nccl,cpu:gloo")
    else:
        dist.init_process_group("gloo")
    flat = init_device_mesh(DEVICE, (4,), mesh_dim_names=("flat",))
    mesh = init_device_mesh(DEVICE, (2, 2), mesh_dim_names=("dp", "ep"))
    check_fsdp_only(flat, mesh, root)
    model = check_ep(mesh, root)
    check_regular_checkpoint_resume(model, root)
    check_torch_consolidation(model, root)
    check_interrupted_resume(model, root)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
