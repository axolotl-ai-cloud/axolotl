"""Distributed native packed BNB checkpoint regression worker."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import bitsandbytes.functional as F
import torch
import torch.distributed as dist
import torch.nn.utils.parametrize as P
from accelerate.utils import fsdp_utils
from bitsandbytes.nn import Linear4bit, Linear8bitLt, Params4bit
from bitsandbytes.nn.parametrize import replace_parameter_4bit
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from axolotl.integrations.expert_parallel.shard import shard_expert_weights
from axolotl.monkeypatch.accelerate.fsdp2_bnb_checkpoint import (
    _encode_state,
    _quant_state,
    packed_parameters,
)
from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _local,
    patch_fsdp2_full_checkpoint,
)
from axolotl.monkeypatch.moe_quant import Bnb8bitParametrization


def clone(value):
    if torch.is_tensor(value):
        result = torch.empty(value.shape, dtype=value.dtype, device="cpu")
        result.copy_(value.detach())
        return result
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    return copy.deepcopy(value)


def compare(actual, expected, path=""):
    if torch.is_tensor(expected):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.contiguous().reshape(-1).view(torch.uint8),
        ), path
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key, item in expected.items():
            compare(actual[key], item, path + "." + key)
    else:
        assert actual == expected


def corrupt(value):
    if torch.is_tensor(value):
        value.zero_()
    elif isinstance(value, dict):
        for item in value.values():
            corrupt(item)


def make_model(mesh, rows, compressed, quant_type, device, eightbit=False, shard=True):
    torch.manual_seed(100)
    model = nn.Module()
    model.experts = nn.Module()
    model.experts.num_experts = 4
    for name in ("gate_up_proj", "down_proj"):
        if eightbit:
            parameter = torch.randint(
                -127, 127, (4, rows, 256), dtype=torch.int8, device=device
            )
            setattr(model.experts, name, nn.Parameter(parameter, requires_grad=False))
            stats = torch.rand(4 * rows, device=device) + 1
            P.register_parametrization(
                model.experts, name, Bnb8bitParametrization(stats), unsafe=True
            )
        else:
            parameter = torch.randn(4, rows, 256, dtype=torch.bfloat16, device=device)
            setattr(model.experts, name, nn.Parameter(parameter, requires_grad=False))
            replace_parameter_4bit(
                model.experts,
                name,
                compress_statistics=compressed,
                quant_type=quant_type,
            )
    assert shard_expert_weights(model, mesh["ep"].get_group()) == 1
    model.dense = Linear4bit(
        256,
        128,
        bias=False,
        quant_type=quant_type,
        quant_storage=torch.bfloat16,
        compress_statistics=True,
    )
    # CPU BNB supports quantization, but Params4bit.to() only triggers it on CUDA.
    packed, state = F.quantize_4bit(
        torch.randn(128, 256, dtype=torch.bfloat16, device=device),
        compress_statistics=True,
        quant_type=quant_type,
        quant_storage=torch.bfloat16,
    )
    model.dense.weight = Params4bit(
        packed,
        requires_grad=False,
        quant_state=state,
        bnb_quantized=True,
        quant_storage=torch.bfloat16,
        module=model.dense,
    )
    model.dense.quant_state = state
    model.frozen = nn.Parameter(torch.tensor([7.0], device=device), requires_grad=False)
    model.register_buffer("counter", torch.tensor(13, device=device))
    if shard and mesh["dp"].size() > 1:
        for name in ("gate_up_proj", "down_proj"):
            stack = model.experts.parametrizations[name]
            stack.original = nn.Parameter(
                distribute_tensor(stack.original, mesh["dp"], (Shard(0),)),
                requires_grad=False,
            )
        original = model.dense.weight
        model.dense.weight = nn.Parameter(
            distribute_tensor(original.data, mesh["dp"], (Shard(0),)),
            requires_grad=False,
        )
    return model


def snapshot(model, full=False):
    result = {}
    for name, descriptor in packed_parameters(model).items():
        metadata = (
            _encode_state(_quant_state(descriptor))
            if descriptor["mode"] == "4bit"
            else {"row_stats": descriptor["entry"].row_stats}
        )
        parameter = descriptor["parameter"]
        physical = (
            parameter.full_tensor()
            if full and isinstance(parameter, DTensor)
            else _local(parameter)
        )
        result[name] = dict(packed=clone(physical), metadata=clone(metadata))
    return result


def dequantized(model):
    result = {}
    for name, descriptor in packed_parameters(model).items():
        parameter = descriptor["parameter"]
        packed = (
            parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
        )
        if descriptor["mode"] == "4bit":
            result[name] = (
                F.dequantize_4bit(packed, _quant_state(descriptor)).detach().cpu()
            )
        else:
            result[name] = descriptor["entry"](packed).detach().cpu()
    return result


def check(
    root,
    label,
    mesh,
    rows,
    compressed,
    quant_type,
    device,
    reorder=False,
    eightbit=False,
    reshard=False,
):
    model = make_model(mesh, rows, compressed, quant_type, device, eightbit=eightbit)
    expected, outputs = snapshot(model), dequantized(model)
    full_expected = snapshot(model, full=True) if reshard else None
    accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0, wait_for_everyone=dist.barrier
    )
    plugin = SimpleNamespace(fsdp_version=2, state_dict_type="FULL_STATE_DICT")
    directory = root / label
    with (
        patch.object(
            F,
            "quantize_4bit",
            side_effect=AssertionError("Checkpoint requantized weights"),
        ),
        patch.object(
            F,
            "dequantize_4bit",
            side_effect=AssertionError("Checkpoint dequantized weights"),
        ),
    ):
        fsdp_utils.save_fsdp_model(plugin, accelerator, model, directory)
    if reorder:
        changed = DeviceMesh(
            device.type, mesh.mesh.flip(1), mesh_dim_names=("dp", "ep")
        )
        model = make_model(
            changed, rows, compressed, quant_type, device, eightbit=eightbit
        )
        model.experts.local_expert_offset = changed.get_coordinate()[1] * 2
        # Snapshot each semantic owner's reference rather than its original rank.
        references = [None] * dist.get_world_size()
        dist.all_gather_object(references, (mesh.get_coordinate(), expected, outputs))
        _, expected, outputs = next(
            item for item in references if item[0] == changed.get_coordinate()
        )
    elif reshard:
        model = make_model(
            mesh, rows, compressed, quant_type, device, eightbit=eightbit, shard=False
        )
        expected = full_expected
    aliases = {
        name: _quant_state(descriptor)
        for name, descriptor in packed_parameters(model).items()
        if descriptor["mode"] == "4bit"
    }
    with torch.no_grad():
        for parameter in model.parameters():
            _local(parameter).zero_()
        for descriptor in packed_parameters(model).values():
            if descriptor["mode"] == "4bit":
                state = _quant_state(descriptor)
                corrupt(_encode_state(state))
                state.dtype = torch.float32
            else:
                descriptor["entry"].row_stats.zero_()
        model.counter.zero_()
    with (
        patch.object(
            F,
            "quantize_4bit",
            side_effect=AssertionError("Checkpoint requantized weights"),
        ),
        patch.object(
            F,
            "dequantize_4bit",
            side_effect=AssertionError("Checkpoint dequantized weights"),
        ),
    ):
        fsdp_utils.load_fsdp_model(plugin, accelerator, model, directory)
    compare(snapshot(model), expected)
    compare(dequantized(model), outputs)
    assert model.frozen.item() == 7
    assert model.counter.item() == 13
    for name, descriptor in packed_parameters(model).items():
        if descriptor["mode"] == "4bit":
            assert _quant_state(descriptor) is aliases[name]
            assert _quant_state(descriptor).dtype == torch.bfloat16
    if dist.get_rank() == 0:
        saved = torch.load(directory / "pytorch_model_fsdp.bin", weights_only=True)
        assert saved["_axolotl_full_model_version"] == 1
        assert set(saved["state"]) == {"frozen", "counter"}
        assert (
            len(saved["quantized"]["experts.parametrizations.gate_up_proj.original"])
            == 2
        )
        print(f"PASS {label}", flush=True)
    dist.barrier()


def check_peft_routes(root, mesh, device):
    from peft import LoraConfig, get_peft_model
    from transformers.distributed.fsdp import get_fsdp_ckpt_kwargs

    base = make_model(mesh, 128, True, "nf4", device)
    base.adapter_target = nn.Linear(16, 16, bias=False, device=device)
    model = get_peft_model(base, LoraConfig(target_modules=["adapter_target"], r=4))
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.fill_(0.125)
    expected = {
        name: clone(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0, wait_for_everyone=dist.barrier
    )
    plugin = SimpleNamespace(fsdp_version=2, state_dict_type="FULL_STATE_DICT")
    directory = root / "real-peft-full"
    fsdp_utils.save_fsdp_model(plugin, accelerator, model, directory)
    with torch.no_grad():
        for parameter in model.parameters():
            _local(parameter).zero_()
        base.counter.zero_()
    fsdp_utils.load_fsdp_model(
        plugin, accelerator, model, directory, **get_fsdp_ckpt_kwargs()
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            compare(parameter, expected[name])
        else:
            assert not torch.count_nonzero(_local(parameter))
    assert base.counter.item() == 0
    adapter_directory = root / "real-peft-adapters"
    fsdp_utils.save_fsdp_model(
        plugin, accelerator, model, adapter_directory, **get_fsdp_ckpt_kwargs()
    )
    if dist.get_rank() == 0:
        saved = torch.load(
            adapter_directory / "pytorch_model_fsdp.bin", weights_only=True
        )
        assert set(saved) == set(expected)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.zero_()
    fsdp_utils.load_fsdp_model(
        plugin, accelerator, model, adapter_directory, **get_fsdp_ckpt_kwargs()
    )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            compare(parameter, expected[name])
    for label, change, message in (
        (
            "unknown-version",
            lambda saved: saved.update(_axolotl_full_model_version=2),
            "version",
        ),
        ("missing-state", lambda saved: saved.pop("state"), "ordinary model state"),
    ):
        malformed = root / label
        if dist.get_rank() == 0:
            saved = torch.load(directory / "pytorch_model_fsdp.bin", weights_only=True)
            change(saved)
            malformed.mkdir(exist_ok=True)
            torch.save(saved, malformed / "pytorch_model_fsdp.bin")
        dist.barrier()
        try:
            fsdp_utils.load_fsdp_model(
                plugin, accelerator, model, malformed, **get_fsdp_ckpt_kwargs()
            )
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError("Malformed adapter-only envelope was accepted")
    if dist.get_rank() == 0:
        print("PASS real-peft-native-routes", flush=True)


def check_rejections(root, mesh, device):
    model = make_model(mesh, 128, True, "nf4", device)
    accelerator = SimpleNamespace(
        is_main_process=dist.get_rank() == 0, wait_for_everyone=dist.barrier
    )
    plugin = SimpleNamespace(fsdp_version=2, state_dict_type="FULL_STATE_DICT")
    source = root / "nested-nf4" / "pytorch_model_fsdp.bin"
    for label, message in (
        ("missing-owner", "missing expert owners"),
        ("changed-ep-grouping", "same EP ownership ranges"),
        ("malformed-codebook", "codebook"),
        ("expert-logical-shape", "logical shape"),
        ("dense-logical-shape", "logical shape"),
        ("legacy-packed", "version"),
    ):
        directory = root / label
        if dist.get_rank() == 0:
            state = torch.load(source, weights_only=True)
            name = "experts.parametrizations.gate_up_proj.original"
            if label == "missing-owner":
                state["quantized"][name].pop()
            elif label == "changed-ep-grouping":
                records = state["quantized"][name]
                regrouped = []
                for record in records:
                    for index in range(2):
                        part = copy.deepcopy(record)
                        part["owner"]["offset"] += index
                        part["owner"]["local"] = 1
                        part["logical_shape"][0] = 1
                        part["layout"]["shape"][0] //= 2
                        part["packed"] = record["packed"].chunk(2, dim=0)[index].clone()
                        part["metadata"]["shape"][0] = 1
                        part["metadata"]["absmax"] = (
                            record["metadata"]["absmax"].chunk(2)[index].clone()
                        )
                        part["metadata"]["state2"]["absmax"] = (
                            record["metadata"]["state2"]["absmax"]
                            .chunk(2)[index]
                            .clone()
                        )
                        regrouped.append(part)
                state["quantized"][name] = regrouped
            elif label in {"expert-logical-shape", "dense-logical-shape"}:
                if label == "dense-logical-shape":
                    name = "dense.weight"
                for record in state["quantized"][name]:
                    shape = record["logical_shape"]
                    shape[-2:] = reversed(shape[-2:])
                    record["metadata"]["shape"] = list(shape)
            elif label == "malformed-codebook":
                state["quantized"][name][0]["metadata"]["code"] = torch.zeros(8)
            else:
                state = {"experts.gate_up_proj": state["quantized"][name][0]["packed"]}
            directory.mkdir(exist_ok=True)
            torch.save(state, directory / "pytorch_model_fsdp.bin")
        dist.barrier()
        with torch.no_grad():
            model.frozen.zero_()
            model.counter.zero_()
        before = snapshot(model)
        try:
            fsdp_utils.load_fsdp_model(plugin, accelerator, model, directory)
        except ValueError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError("Malformed or incompatible checkpoint was accepted")
        compare(snapshot(model), before)
        assert model.frozen.item() == 0
        assert model.counter.item() == 0
        if dist.get_rank() == 0:
            print(f"PASS rejected-{label}", flush=True)
    model.unsupported = Linear8bitLt(16, 16)
    try:
        fsdp_utils.save_fsdp_model(plugin, accelerator, model, root / "dense-int8")
    except ValueError as exc:
        assert "dense BNB Int8Params" in str(exc)
    else:
        raise AssertionError("Unsupported dense 8-bit checkpoint was accepted")
    if dist.get_rank() == 0:
        print("PASS rejected-dense-int8", flush=True)
    dist.barrier()


def check_generic_packed_state(mesh, device):
    from accelerate import Accelerator
    from peft import LoraConfig, get_peft_model

    accelerator = SimpleNamespace(
        state=SimpleNamespace(
            fsdp_plugin=SimpleNamespace(
                fsdp_version=2, state_dict_type="FULL_STATE_DICT"
            )
        )
    )
    for eightbit in (False, True):
        model = make_model(mesh, 128, True, "nf4", device, eightbit=eightbit)
        model.adapter_target = nn.Linear(16, 16, device=device)
        model = get_peft_model(
            model, LoraConfig(target_modules=["adapter_target"], r=2)
        )
        owner = mesh.get_coordinate()[1] + 1
        with torch.no_grad():
            for descriptor in packed_parameters(model).values():
                if descriptor["owner"] is None:
                    continue
                _local(descriptor["parameter"]).fill_(owner)
                if descriptor["mode"] == "4bit":
                    state = _quant_state(descriptor)
                    state.absmax.fill_(owner)
                    state.offset.zero_()
                    state.state2.absmax.fill_(owner)
                else:
                    descriptor["entry"].row_stats.fill_(owner)
        before = snapshot(model)
        for unwrap in (True, False):
            try:
                with patch.object(
                    model,
                    "state_dict",
                    side_effect=AssertionError("Ran packed state hooks"),
                ):
                    Accelerator.get_state_dict(accelerator, model, unwrap=unwrap)
            except ValueError as exc:
                assert "packed BNB expert" in str(exc), str(exc)
                assert "save_fsdp_model" in str(exc), str(exc)
            else:
                raise AssertionError(
                    "Generic full state silently accepted packed EP owners"
                )
        compare(snapshot(model), before)
        if dist.get_rank() == 0:
            print(
                f"PASS generic-packed-rejected-{'int8' if eightbit else 'nested-nf4'}",
                flush=True,
            )


def main():
    device_type, root = sys.argv[1], Path(sys.argv[2])
    if device_type == "cuda":
        import os

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device_type == "cuda"
        else torch.device("cpu")
    )
    mesh = init_device_mesh(
        device_type, (dist.get_world_size() // 2, 2), mesh_dim_names=("dp", "ep")
    )
    patch_fsdp2_full_checkpoint()
    check_generic_packed_state(mesh, device)
    for label, rows, compressed, quant_type, reorder, eightbit, reshard in (
        ("nested-nf4", 128, True, "nf4", False, False, False),
        ("nested-scale-cut", 8, True, "nf4", False, False, False),
        ("uncompressed-fp4", 8, False, "fp4", False, False, False),
        ("reordered-owners", 128, True, "nf4", True, False, False),
        ("packed-int8", 8, False, "nf4", False, True, False),
        ("dp-shards-to-replicas", 128, True, "nf4", False, False, True),
    ):
        check(
            root,
            label,
            mesh,
            rows,
            compressed,
            quant_type,
            device,
            reorder,
            eightbit,
            reshard,
        )
    check_peft_routes(root, mesh, device)
    check_rejections(root, mesh, device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
