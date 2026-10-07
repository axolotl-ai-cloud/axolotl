"""Distributed CPU regression worker for full EP and 8-bit checkpoints."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torchao.optim import AdamW8bit
from torchao.optim.adam import single_param_adam
from torchao.optim.subclass_8bit import OptimState8bit

from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _check_errors,
    _restore_tensor,
    full_model_state,
    full_optimizer_state,
    patch_fsdp2_full_checkpoint,
    restore_model_state,
    restore_optimizer_state,
)
from axolotl.monkeypatch.torchao_optim import patch_torchao_optim_state_8bit


class Experts(nn.Module):
    def __init__(self, ep_rank):
        super().__init__()
        self.num_local_experts = 2
        self.num_experts_global = 4
        self.local_expert_offset = ep_rank * 2
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 16, 256))


class ParamWrapper(nn.Module):
    def __init__(self, ep_rank):
        super().__init__()
        self.base_layer = Experts(ep_rank)
        self.parameter_name = "gate_up_proj"
        self._ep_lora_sharded = True
        self.lora_A = nn.ModuleDict({"default": nn.Linear(256, 32, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(32, 256, bias=False)})


class Model(nn.Module):
    def __init__(self, mesh, expert_mesh, expert_placements, dense_placements):
        super().__init__()
        self.expert = ParamWrapper(mesh.get_coordinate()[-1])
        self.dense = nn.Linear(256, 32, bias=False)
        for module in self.expert.modules():
            for name, p in list(module.named_parameters(recurse=False)):
                value = distribute_tensor(
                    torch.zeros_like(p), expert_mesh, expert_placements
                )
                setattr(module, name, nn.Parameter(value))
        self.dense.weight = nn.Parameter(
            distribute_tensor(
                torch.zeros_like(self.dense.weight), mesh, dense_placements
            )
        )
        self.register_buffer("counter", torch.tensor(3))


def check_full_parameter_save_route(mesh, root):
    from accelerate import Accelerator
    from accelerate.utils import fsdp_utils
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs

    from axolotl.core.trainers.mixins.distributed_parallel import (
        DistributedParallelMixin,
    )

    ep_rank = mesh.get_coordinate()[1]
    model = nn.Module()
    model.experts = Experts(ep_rank)
    model.experts.num_experts = 2
    model.experts.down_proj = nn.Parameter(torch.full((2, 256, 16), ep_rank + 1.0))
    with torch.no_grad():
        model.experts.gate_up_proj.fill_(ep_rank + 1.0)
    for name, parameter in list(model.experts.named_parameters()):
        setattr(
            model.experts,
            name,
            nn.Parameter(distribute_tensor(parameter, mesh["dp"], (Shard(0),))),
        )
    model.frozen = nn.Parameter(torch.tensor([5.0]), requires_grad=False)
    model.register_buffer("counter", torch.tensor(3))
    patch_fsdp2_full_checkpoint()
    accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0,
        wait_for_everyone=dist.barrier,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    accelerator.get_state_dict = lambda model, unwrap=True: Accelerator.get_state_dict(
        accelerator, model, unwrap=unwrap
    )
    trainer = object.__new__(DistributedParallelMixin)
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.axolotl_cfg = SimpleNamespace(expert_parallel_size=2, adapter=None)
    trainer.is_fsdp_enabled = True
    trainer._save_model_native = lambda *args: accelerator.get_state_dict(model)
    state = trainer.save_model()
    if dist.get_rank() == 0:
        for name, shape in (
            ("gate_up_proj", (4, 16, 256)),
            ("down_proj", (4, 256, 16)),
        ):
            value = state[f"experts.{name}"]
            assert tuple(value.shape) == shape
            torch.testing.assert_close(value[:2], torch.ones_like(value[:2]))
            torch.testing.assert_close(value[2:], torch.full_like(value[2:], 2))
        torch.testing.assert_close(state["frozen"], model.frozen)
        torch.testing.assert_close(state["counter"], model.counter)
        print("PASS ep-full-parameter-save-route", flush=True)
    else:
        assert state == {}
    plugin = accelerator.state.fsdp_plugin
    directory = root / "native-full-parameter"
    fsdp_utils.save_fsdp_model(
        plugin, accelerator, model, directory, **get_fsdp_ckpt_kwargs()
    )
    if dist.get_rank() == 0:
        saved = torch.load(directory / "pytorch_model_fsdp.bin", weights_only=True)
        assert set(saved) == set(state)
    with torch.no_grad():
        for parameter in model.parameters():
            (
                parameter.to_local() if isinstance(parameter, DTensor) else parameter
            ).zero_()
        model.counter.zero_()
    fsdp_utils.load_fsdp_model(
        plugin, accelerator, model, directory, **get_fsdp_ckpt_kwargs()
    )
    torch.testing.assert_close(model.frozen, torch.tensor([5.0]), rtol=0, atol=0)
    assert model.counter.item() == 3
    for parameter in model.experts.parameters():
        torch.testing.assert_close(
            parameter.to_local(),
            torch.full_like(parameter.to_local(), ep_rank + 1.0),
            rtol=0,
            atol=0,
        )
    if dist.get_rank() == 0:
        print("PASS native-trainer-non-peft-full-checkpoint", flush=True)


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


def check_peft_trainer_save_route(mesh, root):
    from accelerate import Accelerator
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
    from axolotl.integrations.expert_parallel.shard import shard_expert_lora
    from axolotl.train import save_trained_model
    from axolotl.utils.dict import DictDefault
    from axolotl.utils.freeze import freeze_layers_except

    def make_base():
        torch.manual_seed(456)
        base = nn.Module()
        base.experts = nn.Module()
        base.experts.num_experts = 4
        base.experts.gate_up_proj = nn.Parameter(torch.ones(4, 16, 256))
        base.experts.down_proj = nn.Parameter(torch.ones(4, 256, 16))
        base.dense = nn.Linear(256, 32, bias=False)
        base.register_buffer("counter", torch.tensor(3))
        return base

    base = make_base()
    slice_cpu_experts(base, mesh)
    model = get_peft_model(
        base,
        LoraConfig(
            target_modules=["dense"],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            r=16,
        ),
    )
    assert shard_expert_lora(model, 2) == 4
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            if parameter.requires_grad:
                setattr(
                    module,
                    name,
                    nn.Parameter(distribute_tensor(parameter, mesh["dp"], (Shard(0),))),
                )
    ep_rank = mesh.get_coordinate()[1]
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
                local = (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                )
                local.fill_(ep_rank + 0.125 if "expert" in name else 0.25)
    freeze_layers_except(
        model,
        [f"^{name}$" for name, _ in model.named_parameters() if "lora_B" in name],
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.to_local().add_(0.125)
    expected = {
        name: snapshot(parameter)
        for name, parameter in model.named_parameters()
        if "lora_" in name
    }
    accelerator = SimpleNamespace(
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
    accelerator.get_state_dict = lambda model, unwrap=True: Accelerator.get_state_dict(
        accelerator, model, unwrap=unwrap
    )
    for unwrap in (True, False):
        full = accelerator.get_state_dict(model, unwrap=unwrap)
        if dist.get_rank() == 0:
            assert set(full) == set(model.state_dict())
            assert "base_model.model.counter" in full
            assert "base_model.model.dense.base_layer.weight" in full
    if dist.get_rank() == 0:
        print("PASS full-peft-state-dict", flush=True)
    trainer = object.__new__(AxolotlTrainer)
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.is_fsdp_enabled = True
    trainer.is_deepspeed_enabled = False
    trainer.processing_class = None
    trainer.data_collator = None
    trainer.args = SimpleNamespace(
        output_dir=str(root / "sft-model-only"),
        should_save=dist.get_rank() == 0,
        save_only_model=True,
        push_to_hub=False,
    )
    trainer.save_model(_internal_call=True)
    dist.barrier()
    error = None
    if dist.get_rank() == 0:
        saved = torch.load(
            Path(trainer.args.output_dir) / "pytorch_model_fsdp.bin", weights_only=True
        )
        if set(saved) != set(expected):
            error = "Trainer checkpoint omitted frozen adapter factors"
    _check_errors(error)
    if dist.get_rank() == 0:
        directory = Path(trainer.args.output_dir)
        assert (directory / "adapter_model.safetensors").is_file()
        saved = torch.load(directory / "pytorch_model_fsdp.bin", weights_only=True)
        assert set(saved) == set(expected)
        assert any(value.shape == (64, 256) for value in saved.values())
        assert not (directory / "optimizer.bin").exists()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                (
                    parameter.to_local()
                    if isinstance(parameter, DTensor)
                    else parameter
                ).zero_()
    with patch.object(
        model, "load_adapter", side_effect=AssertionError("Used local PEFT restore")
    ):
        trainer._load_from_checkpoint(trainer.args.output_dir)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            compare(parameter, expected[name])
    if dist.get_rank() == 0:
        print("PASS sft-model-only-ep-adapter-resume", flush=True)
    regular_directory = root / "sft-regular"
    trainer.args.save_only_model = False
    trainer.save_model(str(regular_directory), _internal_call=True)
    from accelerate.utils import fsdp_utils

    fsdp_utils.save_fsdp_model(
        accelerator.state.fsdp_plugin,
        accelerator,
        model,
        regular_directory,
        adapter_only=True,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.to_local().zero_()
    with patch.object(
        model, "load_adapter", side_effect=AssertionError("Used local PEFT restore")
    ):
        trainer._load_from_checkpoint(regular_directory)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            compare(parameter, expected[name])
    if dist.get_rank() == 0:
        print("PASS sft-regular-ep-adapter-resume", flush=True)
    full_adapter = full_model_state(model, adapter_only=True)
    trainer.save_model(str(root / "trainer-final-export"))
    final_directory = root / "final-ep-adapter"
    with patch.object(
        ExpertParallelPlugin, "_resolve_ep_group", return_value=mesh["ep"].get_group()
    ):
        save_trained_model(
            DictDefault(
                {
                    "adapter": "lora",
                    "expert_parallel_size": 2,
                    "output_dir": str(final_directory),
                }
            ),
            trainer,
            model,
        )
    if dist.get_rank() == 0:
        for directory in (root / "trainer-final-export", final_directory):
            assert not (directory / "pytorch_model_fsdp.bin").exists()
            reloaded = PeftModel.from_pretrained(make_base(), directory)
            expected_adapter = get_peft_model_state_dict(model, state_dict=full_adapter)
            actual_adapter = get_peft_model_state_dict(reloaded)
            assert set(actual_adapter) == set(expected_adapter)
            for name, value in expected_adapter.items():
                torch.testing.assert_close(actual_adapter[name], value, rtol=0, atol=0)
            inputs = torch.linspace(-0.5, 0.5, 256).reshape(1, 256)
            dense = model.base_model.model.dense
            expected_output = torch.nn.functional.linear(
                inputs, dense.base_layer.weight
            )
            expected_output += (
                torch.nn.functional.linear(
                    torch.nn.functional.linear(
                        inputs,
                        full_adapter["base_model.model.dense.lora_A.default.weight"],
                    ),
                    full_adapter["base_model.model.dense.lora_B.default.weight"],
                )
                * dense.scaling["default"]
            )
            torch.testing.assert_close(
                reloaded.base_model.model.dense(inputs), expected_output, rtol=0, atol=0
            )
        print("PASS final-ep-adapter-base-layout", flush=True)
    dist.barrier()


def check_non_ep_peft_exports(root):
    from accelerate import Accelerator
    from accelerate.utils import DistributedType, fsdp_utils
    from bitsandbytes.nn.parametrize import replace_parameter_4bit
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel

    from axolotl.core.trainers.base import AxolotlTrainer
    from axolotl.utils.freeze import freeze_layers_except

    config_path = root / "base-config"
    config = GPT2Config(vocab_size=32, n_embd=8, n_head=2, n_layer=1)
    if dist.get_rank() == 0:
        config.save_pretrained(config_path)
    dist.barrier()

    def make_base():
        torch.manual_seed(71)
        base = GPT2LMHeadModel(copy.deepcopy(config))
        base.resize_token_embeddings(40)
        base.config._name_or_path = str(config_path)
        base.name_or_path = str(config_path)
        base.experts = nn.Module()
        base.experts.gate_up_proj = nn.Parameter(
            torch.randn(4, 8, 32), requires_grad=False
        )
        replace_parameter_4bit(base.experts, "gate_up_proj", compress_statistics=True)
        return base

    model = get_peft_model(
        make_base(),
        LoraConfig(target_modules=["c_attn"], modules_to_save=["ln_f"], r=2),
    )
    model.add_adapter("inactive", copy.deepcopy(model.peft_config["default"]))
    model._moe_experts_quantized = True
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.fill_(0.2 if ".inactive." in name else 0.1)
            elif "modules_to_save" in name:
                parameter.fill_(0.125 if name.endswith("bias") else 1.25)
    freeze_layers_except(
        model,
        [
            f"^{name}$"
            for name, _ in model.named_parameters()
            if "lora_B.default" in name
        ],
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.0625)
    inputs = torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]])
    expected, outputs = {}, {}
    for adapter in ("default", "inactive"):
        model.set_adapter(adapter)
        model.eval()
        expected[adapter] = {
            name: value.detach().clone()
            for name, value in get_peft_model_state_dict(
                model, adapter_name=adapter
            ).items()
        }
        outputs[adapter] = model(inputs).logits.detach()
    model.set_adapter("default")
    freeze_layers_except(
        model,
        [
            f"^{name}$"
            for name, _ in model.named_parameters()
            if "lora_B.default" in name
        ],
    )
    mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("dp",))
    replacements = {}
    for prefix, module in model.named_modules():
        for name, parameter in list(module.named_parameters(recurse=False)):
            key = prefix + "." + name
            if (
                "lora_" in key
                or "modules_to_save" in key
                or key.endswith(("transformer.wte.weight", "lm_head.weight"))
            ):
                if id(parameter) not in replacements:
                    replacements[id(parameter)] = nn.Parameter(
                        distribute_tensor(parameter.detach(), mesh, (Shard(0),)),
                        requires_grad=parameter.requires_grad,
                    )
                setattr(module, name, replacements[id(parameter)])
    # A sharded packed weight beside plain quant-state tensors crashed DCP full state dicts.
    packed = model.base_model.model.experts.parametrizations["gate_up_proj"]
    packed.original = nn.Parameter(
        distribute_tensor(packed.original.detach(), mesh, (Shard(0),)),
        requires_grad=False,
    )
    assert isinstance(packed.original, DTensor)
    snapshots = {
        key: parameter.to_local().detach().clone()
        for key, parameter in replacements.items()
    }
    accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0,
        num_processes=dist.get_world_size(),
        wait_for_everyone=dist.barrier,
        parallelism_config=None,
        distributed_type=DistributedType.FSDP,
        is_fsdp2=True,
        unwrap_model=lambda model, **kwargs: model,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    accelerator.get_state_dict = lambda model, unwrap=True: Accelerator.get_state_dict(
        accelerator, model, unwrap=unwrap
    )
    trainer = object.__new__(AxolotlTrainer)
    trainer.model, trainer.accelerator = model, accelerator
    trainer.is_fsdp_enabled, trainer.is_deepspeed_enabled = True, False
    trainer.processing_class = trainer.data_collator = None
    model._tp_size = 2
    with patch.object(
        model,
        "state_dict",
        side_effect=AssertionError("Collected frozen quantized base"),
    ):
        full_model_state(model, adapter_only=True)
    for label, internal_call, model_only in (
        ("regular", True, False),
        ("model-only", True, True),
        ("final", False, False),
    ):
        directory = root / f"non-ep-{label}"
        trainer.args = SimpleNamespace(
            output_dir=str(directory),
            should_save=dist.get_rank() == 0,
            save_only_model=model_only,
            push_to_hub=False,
        )
        trainer.save_model(_internal_call=internal_call)
        if label == "regular":
            fsdp_utils.save_fsdp_model(
                accelerator.state.fsdp_plugin,
                accelerator,
                model,
                directory,
                adapter_only=True,
            )
        error = None
        if dist.get_rank() == 0:
            try:
                reloaded = PeftModel.from_pretrained(make_base(), directory)
                reloaded.load_adapter(directory / "inactive", adapter_name="inactive")
                for adapter in ("default", "inactive"):
                    file = (
                        directory
                        / ("inactive" if adapter == "inactive" else "")
                        / "adapter_model.safetensors"
                    )
                    saved = load_file(file)
                    assert set(saved) == set(expected[adapter])
                    for name, value in expected[adapter].items():
                        compare(saved[name], value)
                    reloaded.set_adapter(adapter)
                    reloaded.eval()
                    torch.testing.assert_close(
                        reloaded(inputs).logits, outputs[adapter], rtol=0, atol=0
                    )
                assert any("wte.weight" in name for name in saved)
                assert any("ln_f" in name for name in saved)
                assert (directory / "pytorch_model_fsdp.bin").exists() == internal_call
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        _check_errors(error)
        if internal_call:
            with torch.no_grad():
                for parameter in replacements.values():
                    parameter.to_local().zero_()
            with patch.object(
                model, "load_adapter", side_effect=AssertionError("Used local restore")
            ):
                trainer._load_from_checkpoint(directory)
            for key, parameter in replacements.items():
                compare(parameter.to_local(), snapshots[key])
        if dist.get_rank() == 0:
            print(f"PASS non-ep-frozen-adapter-{label}", flush=True)
    dist.barrier()


def check_restore_collectives(mesh):
    parameter = distribute_tensor(torch.arange(16.0), mesh, (Shard(0), Shard(0)))
    state = torch.arange(16.0) if dist.get_rank() == 0 else None
    with (
        patch.object(
            dist, "all_gather_object", wraps=dist.all_gather_object
        ) as gathers,
        patch.object(
            dist, "broadcast_object_list", wraps=dist.broadcast_object_list
        ) as broadcasts,
    ):
        restored = _restore_tensor(state, parameter, None)
    torch.testing.assert_close(
        restored.to_local(), parameter.to_local(), rtol=0, atol=0
    )
    assert gathers.call_count == 3
    assert broadcasts.call_count == 1
    if dist.get_rank() == 0:
        print("PASS restore-collectives-batched", flush=True)


def check_mixtral_full_model_export(mesh, root):
    from accelerate import Accelerator
    from safetensors.torch import load_file
    from transformers import MixtralConfig, MixtralForCausalLM

    from axolotl.core.trainers.base import AxolotlTrainer

    torch.manual_seed(13)
    model = MixtralForCausalLM(
        MixtralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_implementation="eager",
            experts_implementation="eager",
        )
    )
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    slice_cpu_experts(model, mesh)
    trainer = object.__new__(AxolotlTrainer)
    trainer.model = model
    trainer.is_fsdp_enabled = True
    trainer.is_deepspeed_enabled = False
    trainer.processing_class = None
    trainer.data_collator = None
    trainer.accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0,
        wait_for_everyone=dist.barrier,
        parallelism_config=None,
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        ),
    )
    trainer.accelerator.get_state_dict = lambda model: Accelerator.get_state_dict(
        trainer.accelerator, model
    )
    trainer.args = SimpleNamespace(
        output_dir=str(root / "mixtral-model-only"),
        should_save=dist.get_rank() == 0,
        save_only_model=True,
        push_to_hub=False,
    )
    trainer.save_model(_internal_call=True)
    if dist.get_rank() == 0:
        saved = torch.load(
            Path(trainer.args.output_dir) / "pytorch_model_fsdp.bin", weights_only=True
        )
        assert set(saved) == set(expected)
        for name, value in expected.items():
            torch.testing.assert_close(saved[name], value, rtol=0, atol=0)
    directory = root / "mixtral-final-export"
    trainer.save_model(str(directory))
    if dist.get_rank() == 0:
        assert not (directory / "pytorch_model_fsdp.bin").exists()
        exported = load_file(directory / "model.safetensors")
        assert not any(
            "gate_up_proj" in name or "down_proj" in name for name in exported
        )
        assert sum(".experts." in name for name in exported) == 12
        reloaded = MixtralForCausalLM.from_pretrained(
            directory, attn_implementation="eager", experts_implementation="eager"
        )
        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        print("PASS mixtral-final-original-layout", flush=True)
    dist.barrier()


def snapshot(value):
    if isinstance(value, DTensor):
        value = value.to_local()
    if isinstance(value, OptimState8bit):
        return {a: getattr(value, a).clone() for a in value.tensor_attrs}
    return value.detach().clone()


def compare(actual, expected):
    if isinstance(actual, DTensor):
        actual = actual.to_local()
    if isinstance(expected, dict):
        assert isinstance(actual, OptimState8bit)
        for attr in actual.tensor_attrs:
            torch.testing.assert_close(
                getattr(actual, attr), expected[attr], rtol=0, atol=0
            )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def populate(model, optimizer, ep_rank, shard_rank):
    for i, (name, p) in enumerate(model.named_parameters()):
        local = p.to_local()
        with torch.no_grad():
            local.copy_(
                torch.arange(local.numel()).reshape(local.shape) / 10000
                + ep_rank
                + shard_rank * 10
                + i
            )
        values = {
            "step": torch.tensor(2.0 + (ep_rank if name.startswith("expert.") else 0))
        }
        for key, signed in [("exp_avg", True), ("exp_avg_sq", False)]:
            buffer = optimizer._new_buffer(p, signed)
            inner = buffer.to_local()
            if isinstance(inner, OptimState8bit):
                inner.codes.copy_(
                    (
                        (
                            torch.arange(inner.numel()).reshape(inner.shape)
                            + ep_rank * 19
                            + shard_rank * 7
                        )
                        % 256
                    ).to(torch.uint8)
                )
                inner.scale.copy_(
                    torch.arange(inner.scale.numel()) / 1000
                    + 1
                    + ep_rank
                    + shard_rank * 10
                )
            else:
                inner.fill_(0.1 + ep_rank + shard_rank)
            values[key] = buffer
        optimizer.state[p] = values


def roundtrip(root, label, model, optimizer, target=None):
    rank = dist.get_rank()
    expected = {
        name: dict(
            parameter=snapshot(p),
            state={k: snapshot(v) for k, v in optimizer.state[p].items()},
        )
        for name, p in model.named_parameters()
    }
    # Keep an independent pre-save reference for each EP group's complete tensors.
    full_reference = {
        name: dict(
            parameter=snapshot(p.full_tensor()),
            state={
                k: snapshot(v.full_tensor() if isinstance(v, DTensor) else v)
                for k, v in optimizer.state[p].items()
            },
        )
        for name, p in model.named_parameters()
    }
    ep_rank = (
        model.expert.base_layer.local_expert_offset // 2
        if hasattr(model, "expert")
        else 0
    )
    references = [None] * dist.get_world_size()
    dist.all_gather_object(references, (ep_rank, full_reference))
    model_state = full_model_state(model)
    optimizer_state = full_optimizer_state(model, optimizer)
    if rank == 0:
        torch.save(model_state, root / f"{label}-model.bin")
        torch.save(optimizer_state, root / f"{label}-optimizer.bin")
        if hasattr(model, "expert"):
            assert model_state["expert.base_layer.gate_up_proj"].shape == (4, 16, 256)
            assert model_state["expert.lora_A.default.weight"].shape == (64, 256)
            assert model_state["expert.lora_B.default.weight"].shape == (256, 64)
    dist.barrier()
    if target is None:
        for p in model.parameters():
            with torch.no_grad():
                p.to_local().zero_()
        optimizer.state.clear()
    else:
        model, optimizer = target()
    model_state = (
        torch.load(root / f"{label}-model.bin", weights_only=True) if rank == 0 else {}
    )
    optimizer_state = (
        torch.load(root / f"{label}-optimizer.bin", weights_only=True)
        if rank == 0
        else {}
    )
    restore_model_state(model, model_state)
    restore_optimizer_state(model, optimizer, optimizer_state)
    if target is None:
        for name, p in model.named_parameters():
            compare(p, expected[name]["parameter"])
            for key, value in optimizer.state[p].items():
                compare(value, expected[name]["state"][key])
        # One more Adam update must be identical, including the newly quantized moments.
        for name, p in model.named_parameters():
            state = optimizer.state[p]
            local = p.to_local()
            reference_p = expected[name]["parameter"].clone()
            reference_state = copy.deepcopy(expected[name]["state"])
            for key, signed in [("exp_avg", True), ("exp_avg_sq", False)]:
                if isinstance(reference_state[key], dict):
                    parts = reference_state[key]
                    reference_state[key] = OptimState8bit(
                        *(parts[a] for a in ("codes", "scale", "qmap")),
                        signed,
                        dtype=torch.float32,
                    )
            grad = torch.full_like(local, 0.125)
            for parameter, values in [(local, state), (reference_p, reference_state)]:
                values["step"] += 1
                with torch.no_grad():
                    single_param_adam(
                        parameter,
                        grad,
                        values["step"],
                        values["exp_avg"].to_local()
                        if isinstance(values["exp_avg"], DTensor)
                        else values["exp_avg"],
                        values["exp_avg_sq"].to_local()
                        if isinstance(values["exp_avg_sq"], DTensor)
                        else values["exp_avg_sq"],
                        None,
                        torch.tensor(0.001),
                        0.9,
                        0.999,
                        0.01,
                        1e-8,
                        True,
                        False,
                    )
            compare(p, reference_p)
            for key in ("exp_avg", "exp_avg_sq", "step"):
                compare(state[key], snapshot(reference_state[key]))
    else:
        ep_rank = model.expert.base_layer.local_expert_offset // 2
        reference = next(values for ep, values in references if ep == ep_rank)
        for name, p in model.named_parameters():
            compare(p.full_tensor(), reference[name]["parameter"])
            for key, value in optimizer.state[p].items():
                compare(
                    value.full_tensor() if isinstance(value, DTensor) else value,
                    reference[name]["state"][key],
                )
    assert model.counter.item() == 3
    if rank == 0:
        print(f"PASS {label}", flush=True)
    dist.barrier()


def main():
    patch_torchao_optim_state_8bit()
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    root = Path(sys.argv[1])
    root.mkdir(exist_ok=True)
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "ep"))
    check_full_parameter_save_route(mesh, root)
    check_peft_trainer_save_route(mesh, root)
    check_non_ep_peft_exports(root)
    check_restore_collectives(mesh)
    check_mixtral_full_model_export(mesh, root)
    model = Model(mesh, mesh["dp"], (Shard(0),), (Shard(0), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])
    roundtrip(root, "ep-dp", model, optimizer)
    # Recreate the source after the update performed by the first check.
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])
    permuted = DeviceMesh(
        "cpu", torch.tensor([[0, 2], [1, 3]]), mesh_dim_names=("dp", "ep")
    )

    def target():
        new_model = Model(permuted, permuted["dp"], (Shard(0),), (Shard(0), Shard(0)))
        return new_model, AdamW8bit(new_model.parameters(), lr=0.001)

    roundtrip(root, "ep-permuted-ranks", model, optimizer, target)
    cp = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("cp", "ep"))
    model = Model(cp, cp["cp"], (Replicate(),), (Replicate(), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, cp.get_coordinate()[1], 0)
    roundtrip(root, "ep-cp", model, optimizer)
    hsdp = DeviceMesh(
        "cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("replicate", "ep")
    )
    # All ranks must create subgroup meshes in the same collective order.
    expert_meshes = [
        DeviceMesh(
            "cpu",
            hsdp.mesh[:, ep].reshape(2, 1),
            mesh_dim_names=("replicate", "shard"),
        )
        for ep in range(2)
    ]
    expert_mesh = expert_meshes[hsdp.get_coordinate()[1]]
    model = Model(mesh, mesh["dp"], (Shard(0),), (Shard(0), Shard(0)))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, mesh.get_coordinate()[1], mesh.get_coordinate()[0])

    def hsdp_target():
        target_model = Model(
            hsdp, expert_mesh, (Replicate(), Shard(0)), (Replicate(), Shard(0))
        )
        return target_model, AdamW8bit(target_model.parameters(), lr=0.001)

    roundtrip(root, "ep-dp-to-hsdp", model, optimizer, hsdp_target)
    model, optimizer = hsdp_target()
    populate(model, optimizer, hsdp.get_coordinate()[1], 0)
    roundtrip(root, "ep-hsdp", model, optimizer)
    flat = DeviceMesh("cpu", torch.arange(4), mesh_dim_names=("tp",))
    model = nn.Module()
    model.weight = nn.Parameter(
        distribute_tensor(torch.zeros(128, 256), flat, (Shard(1),))
    )
    model.register_buffer("counter", torch.tensor(3))
    optimizer = AdamW8bit(model.parameters(), lr=0.001)
    populate(model, optimizer, rank, 0)
    # The wrapper cannot full_tensor() a column shard; compare its physical local payload.
    expected_p = snapshot(model.weight)
    expected_m = {k: snapshot(v) for k, v in optimizer.state[model.weight].items()}
    state = full_optimizer_state(model, optimizer)
    restore_optimizer_state(model, optimizer, state)
    compare(model.weight, expected_p)
    for k, v in optimizer.state[model.weight].items():
        compare(v, expected_m[k])
    if rank == 0:
        print("PASS tp-cross-global-blocks-same-layout", flush=True)
    small_source = nn.Module()
    small_source.weight = nn.Parameter(
        distribute_tensor(
            torch.zeros(64, 64, dtype=torch.bfloat16), flat, (Replicate(),)
        )
    )
    small_optimizer = AdamW8bit(small_source.parameters())
    populate(small_source, small_optimizer, 0, 0)
    expected_floats = {
        key: value.to_local().dequantize(output_dtype=torch.float32)
        for key, value in small_optimizer.state[small_source.weight].items()
        if key != "step"
    }
    small_state = full_optimizer_state(small_source, small_optimizer)
    small_target = nn.Module()
    small_target.weight = nn.Parameter(
        distribute_tensor(torch.zeros(64, 64, dtype=torch.bfloat16), flat, (Shard(0),))
    )
    small_target_optimizer = AdamW8bit(small_target.parameters())
    restore_optimizer_state(small_target, small_target_optimizer, small_state)
    for key, value in small_target_optimizer.state[small_target.weight].items():
        if key != "step":
            assert value.dtype == torch.float32
            assert not isinstance(value.to_local(), OptimState8bit)
            compare(value.to_local(), expected_floats[key].chunk(4, dim=0)[rank])
    if rank == 0:
        print("PASS quantized-to-small-fp32-shards", flush=True)

    from torch.distributed.tensor.placement_types import _StridedShard

    strided_model = nn.Module()
    strided_model.weight = nn.Parameter(
        distribute_tensor(
            torch.zeros(128, 256), mesh, (_StridedShard(0, split_factor=2), Shard(0))
        )
    )
    strided_optimizer = AdamW8bit(strided_model.parameters(), lr=0.001)
    populate(strided_model, strided_optimizer, rank, 0)
    reference_full = strided_model.weight.full_tensor().detach()
    saved_model = full_model_state(strided_model)
    if rank == 0:
        torch.testing.assert_close(
            saved_model["weight"], reference_full, rtol=0, atol=0
        )
    expected_strided = {
        k: snapshot(v) for k, v in strided_optimizer.state[strided_model.weight].items()
    }
    saved_strided = full_optimizer_state(strided_model, strided_optimizer)
    strided_optimizer.state.clear()
    restore_optimizer_state(strided_model, strided_optimizer, saved_strided)
    for key, value in strided_optimizer.state[strided_model.weight].items():
        compare(value, expected_strided[key])
    if rank == 0:
        print("PASS strided-dp-tp", flush=True)

    row_model = nn.Module()
    row_model.weight = nn.Parameter(
        distribute_tensor(torch.zeros(128, 256), flat, (Shard(0),))
    )
    try:
        restore_optimizer_state(row_model, AdamW8bit(row_model.parameters()), state)
    except ValueError as exc:
        assert "regroups saved quantization blocks" in str(exc)
    else:
        raise AssertionError("Incompatible regrouping was accepted")
    if rank == 0:
        print("PASS incompatible-restore-rejected", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
