"""Distributed worker: sharded adapter checkpoints consolidate into the same bytes as the gathered
``adapter_model.safetensors``, for FSDP2-only and FSDP2 + EP (interleaved expert LoRA) layouts."""

import filecmp
import os
import shutil
import sys
import time
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
    _model_needs_ownership,
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
            # seeded before distributing, so every rank's slice holds different values
            generator = torch.Generator().manual_seed(sum(map(ord, full)))
            value = torch.randn(parameter.shape, generator=generator)
            if "o.lora_B" in full:
                value = value.to(torch.bfloat16)
            setattr(
                module,
                name,
                nn.Parameter(distribute_tensor(on(value), target, placement)),
            )
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


def fill_unique(model):
    """Per-rank values on FSDP shards, identical values on plain (replicated) tensors."""
    rank = dist.get_rank()
    fill(
        model,
        lambda name: (
            sum(map(ord, name)) * 16
            + (rank if isinstance(model.get_parameter(name), DTensor) else 0)
        ),
    )


def save_both(model, root, label, spy_names=("_transfer", "_gather_tensor")):
    """Gather-path and sharded saves of ``model``; returns the spy calls and each shard file's
    keys as consolidation saw them."""
    gathered, sharded = root / f"{label}-gather", root / f"{label}-sharded"
    if _model_needs_ownership(model):
        make_trainer(model, gathered).save_model(_internal_call=True)
    else:
        # without expert ownership the trainer's gather goes through accelerate's state dict
        state = full_model_state(model, adapter_only=True)
        if dist.get_rank() == 0:
            make_trainer(model, gathered)._save(str(gathered), state_dict=state)
    dist.barrier()
    shard_keys: dict = {}

    def recording(shard_dir, *args, **kwargs):
        from safetensors import safe_open

        for path in sorted(Path(shard_dir).glob("shard-*.safetensors")):
            with safe_open(path, "pt") as handle:
                shard_keys[path.name] = sorted(handle.keys())
        return consolidate(shard_dir, *args, **kwargs)

    with (
        Spy(*spy_names) as spy,
        patch.object(fsdp2_sharded_adapter, "consolidate", recording),
    ):
        make_trainer(model, sharded, "sharded").save_model(_internal_call=True)
        if dist.get_rank() == 0:
            CONSOLIDATIONS.wait()
    expect_identical(label, gathered, sharded)
    return spy.calls, shard_keys, sharded


def ep_fsdp_model():
    """Tiny Mixtral laid out the way ``fsdp2_prepare_model`` composes EP with dp_shard: experts
    sliced over ``ep``, expert LoRA wrappers FSDP2-sharded on ``dp_shard``, the rest on the
    flattened ``dp_shard x ep`` mesh."""
    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import fully_shard
    from transformers import MixtralConfig, MixtralForCausalLM

    from axolotl.integrations.expert_parallel import shard
    from axolotl.integrations.expert_parallel.plugin import (
        ExpertParallelPlugin,
        expert_fsdp_mesh,
    )

    torch.manual_seed(11)
    model = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=50,
            hidden_size=16,
            intermediate_size=10,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_local_experts=6,
            num_experts_per_tok=2,
            max_position_embeddings=32,
        )
    ).to(DEVICE)
    mesh = init_device_mesh(DEVICE, (2, 2), mesh_dim_names=("dp_shard", "ep"))
    ep_group = mesh["ep"].get_group()
    dense = mesh[("dp_shard", "ep")]._flatten("dp_shard_ep")
    ep_rank = dist.get_rank(ep_group)

    def slice_local(module, name, count, _ranks):
        shard._replace_with_slice(module, name, ep_rank * count, (ep_rank + 1) * count)

    with patch.object(shard, "_scatter_expert_from_rank0", slice_local):
        assert shard.shard_expert_weights(model, ep_group) == 2
    decoder = model.model
    model = get_peft_model(
        model,
        LoraConfig(
            r=3,
            lora_alpha=6,
            target_modules=["q_proj", "o_proj"],
            target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
        ),
    )
    model._ep_lora_group = ep_group
    assert shard.shard_expert_lora(model, mesh["ep"].size()) == 8
    ExpertParallelPlugin.fully_shard_experts(model, expert_fsdp_mesh(mesh), {})
    for layer in decoder.layers:
        fully_shard(layer, mesh=dense)
    fully_shard(model, mesh=dense)
    fill_unique(model)
    return model


def check_ep_fsdp(root):
    model = ep_fsdp_model()
    # 3 local experts x r 3 = 9 lora_A rows over dp_shard 2: uneven FSDP shards
    expert_a = next(
        p
        for n, p in model.named_parameters()
        if "experts" in n and "lora_A" in n and p.requires_grad
    )
    assert isinstance(expert_a, DTensor) and expert_a.device_mesh.mesh_dim_names == (
        "dp_shard",
    ), expert_a.device_mesh
    assert expert_a.shape[0] == 9, expert_a.shape
    calls, shard_keys, sharded = save_both(model, root, "ep-fsdp")
    assert calls == {"_transfer": 0, "_gather_tensor": 0}, calls
    if dist.get_rank() == 0:
        from safetensors.torch import load_file

        assert len(shard_keys) == 4, shard_keys
        exported = load_file(sharded / "adapter_model.safetensors")
        experts = {k: v.shape for k, v in exported.items() if "experts" in k}
        assert len(experts) == 8, experts
        # full EP size: lora_A [r * E, in], lora_B [out, r * E]
        assert all(shape[0] == 18 for k, shape in experts.items() if "lora_A" in k), (
            experts
        )
        assert all(shape[1] == 18 for k, shape in experts.items() if "lora_B" in k), (
            experts
        )
    dist.barrier()
    return model


def modules_to_save_model(flat, tie=False, replicated=()):
    """Tiny Llama with trained ``lm_head`` / ``embed_tokens`` copies; the copies of the
    ``replicated`` modules are left out of FSDP and held whole on every rank."""
    from peft import LoraConfig, get_peft_model
    from torch.distributed.fsdp import fully_shard
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(5)
    base = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=66,  # 17/17/17/15 rows over 4 ranks
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
            tie_word_embeddings=tie,
        )
    ).to(DEVICE)
    model = get_peft_model(
        base,
        LoraConfig(
            r=4,
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["lm_head", "embed_tokens"],
            ensure_weight_tying=tie,
        ),
    )
    ignored = {
        parameter
        for name, parameter in model.named_parameters()
        if any(f"{module}.modules_to_save." in name for module in replicated)
    }
    for layer in base.model.layers:
        fully_shard(layer, mesh=flat)
    fully_shard(model, mesh=flat, ignored_params=ignored or None)
    fill_unique(model)
    return model


EMBED = "base_model.model.model.embed_tokens.weight"
LM_HEAD = "base_model.model.lm_head.weight"


def copies(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if ".modules_to_save." in name
    }


def check_modules_to_save(flat, root):
    model = modules_to_save_model(flat)
    trained = copies(model)
    assert len(trained) == 2, list(trained)
    assert all(
        isinstance(p, DTensor) and p.placements == (Shard(0),) for p in trained.values()
    )
    calls, shard_keys, sharded = save_both(model, root, "modules-to-save")
    assert calls == {"_transfer": 0, "_gather_tensor": 0}, calls
    if dist.get_rank() == 0:
        from safetensors.torch import load_file

        assert all(EMBED in keys and LM_HEAD in keys for keys in shard_keys.values())
        exported = load_file(sharded / "adapter_model.safetensors")
        assert exported[EMBED].shape == exported[LM_HEAD].shape == (66, 16)
        assert not torch.equal(exported[EMBED], exported[LM_HEAD])
    dist.barrier()


def check_modules_to_save_tied(flat, root):
    model = modules_to_save_model(flat, tie=True)
    trained = copies(model)
    # one trained tensor reachable under both modules' names
    assert len(trained) == 2 and len({id(p) for p in trained.values()}) == 1, list(
        trained
    )
    assert isinstance(next(iter(trained.values())), DTensor)
    calls, shard_keys, sharded = save_both(model, root, "modules-to-save-tied")
    assert calls == {"_transfer": 0, "_gather_tensor": 0}, calls
    if dist.get_rank() == 0:
        from safetensors.torch import load_file

        assert len(shard_keys) == 4, shard_keys
        exported = load_file(sharded / "adapter_model.safetensors")
        assert exported[EMBED].shape == (66, 16)
        assert torch.equal(exported[EMBED], exported[LM_HEAD])
    dist.barrier()


def check_replicated_wrapper(flat, root):
    model = modules_to_save_model(flat, replicated=("lm_head",))
    trained = copies(model)
    head = next(p for n, p in trained.items() if "lm_head" in n)
    embed = next(p for n, p in trained.items() if "embed_tokens" in n)
    assert not isinstance(head, DTensor) and isinstance(embed, DTensor)
    calls, shard_keys, sharded = save_both(model, root, "replicated-wrapper")
    assert calls == {"_transfer": 0, "_gather_tensor": 0}, calls
    if dist.get_rank() == 0:
        holders = [name for name, keys in shard_keys.items() if LM_HEAD in keys]
        assert holders == [fsdp2_sharded_adapter.shard_name(0)], shard_keys
        assert sum(EMBED in keys for keys in shard_keys.values()) == 4, shard_keys
    dist.barrier()


def check_final_save(model, root, label):
    """The end-of-training export goes through the sharded writer, matches the gather path's
    bytes, and returns only once it and any earlier checkpoint are consolidated."""
    from axolotl.train import save_trained_model

    rank0 = dist.get_rank() == 0
    gathered, sharded = root / f"{label}-final-gather", root / f"{label}-final"
    run = root / f"{label}-final-run"
    if _model_needs_ownership(model):
        # what save_trained_model does for EP under the gather format
        make_trainer(model, gathered).save_model(str(gathered))
    else:
        state = full_model_state(model, adapter_only=True)
        if rank0:
            make_trainer(model, gathered)._save(str(gathered), state_dict=state)
    if rank0:
        # left behind by a run that died mid-consolidation
        stale = sharded / SHARD_DIRNAME
        stale.mkdir(parents=True)
        (stale / "manifest.json").write_text("{}")
        (stale / fsdp2_sharded_adapter.shard_name(0)).write_bytes(b"junk")
    dist.barrier()

    def slow(shard_dir, *args, **kwargs):
        time.sleep(0.5)
        return consolidate(shard_dir, *args, **kwargs)

    trainer = make_trainer(model, run, "sharded")
    checkpoint = run / "checkpoint-1"
    cfg = DictDefault(
        {
            "adapter": "lora",
            "adapter_checkpoint_format": "sharded",
            "output_dir": str(sharded),
        }
    )
    error = None
    with patch.object(fsdp2_sharded_adapter, "consolidate", slow):
        trainer.save_model(str(checkpoint), _internal_call=True)
        save_trained_model(cfg, trainer, model)
        if rank0:
            try:
                assert CONSOLIDATIONS.pending() == []
                for directory in (checkpoint, sharded):
                    assert (directory / "adapter_model.safetensors").is_file()
                    assert not (directory / SHARD_DIRNAME).exists(), directory
                assert not (sharded / "pytorch_model_fsdp.bin").exists()
            except AssertionError as exc:
                error = f"final save returned early: {exc!r}"
    _check_errors(error)
    assert trainer._axolotl_final_adapter_saved
    # the final export is never held back like a checkpoint
    assert os.path.abspath(sharded) not in trainer._axolotl_unconsolidated
    expect_identical(f"final-{label}", gathered, sharded)

    # adapter_final_save_format: gather keeps the final export off the sharded writer
    trainer = make_trainer(model, run, "sharded")
    trainer._axolotl_cfg["adapter_final_save_format"] = "gather"
    with patch.object(
        fsdp2_sharded_adapter,
        "write_adapter_shards",
        side_effect=AssertionError("final export went through the sharded writer"),
    ):
        assert not trainer.save_final_adapter(str(root / f"{label}-final-forced"))
    dist.barrier()


def main():
    root = Path(sys.argv[1])
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("cuda:nccl,cpu:gloo")
    else:
        dist.init_process_group("gloo")
    flat = init_device_mesh(DEVICE, (4,), mesh_dim_names=("flat",))
    mesh = init_device_mesh(DEVICE, (2, 2), mesh_dim_names=("dp", "ep"))
    fsdp_model = check_fsdp_only(flat, mesh, root)
    model = check_ep(mesh, root)
    check_regular_checkpoint_resume(model, root)
    check_torch_consolidation(model, root)
    check_interrupted_resume(model, root)
    ep_fsdp = check_ep_fsdp(root)
    check_final_save(fsdp_model, root, "fsdp-only")
    check_final_save(model, root, "ep")
    check_final_save(ep_fsdp, root, "ep-fsdp")
    check_modules_to_save(flat, root)
    check_modules_to_save_tied(flat, root)
    check_replicated_wrapper(flat, root)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
