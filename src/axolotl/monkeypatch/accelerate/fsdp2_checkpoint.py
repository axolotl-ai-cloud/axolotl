# Copyright 2026 Axolotl AI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Ownership-aware full FSDP2 model and quantized optimizer checkpoints."""

from __future__ import annotations

import copy
import inspect
import math
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.utils.fsdp_utils import fsdp2_canonicalize_names
from accelerate.utils.modeling import is_peft_model
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

_QUANTIZED = "_axolotl_optim_state_8bit"
_VERSION = "_axolotl_full_optimizer_version"
_OWNED_SCALAR = "_axolotl_ep_scalar"
_TRANSFER_BYTES = 16 * 1024 * 1024


def _parameters(model):
    return fsdp2_canonicalize_names(dict(model.named_parameters()))


def expert_ownership(model):
    """Map canonical parameter names to the experts they actually own."""
    from axolotl.integrations.expert_parallel.shard import (
        _is_param_wrapper,
        _real_experts_base,
    )

    by_id = {}
    for module in model.modules():
        base = _real_experts_base(module) if _is_param_wrapper(module) else module
        local = getattr(base, "num_local_experts", None)
        total = getattr(base, "num_experts_global", None)
        if local is None or total is None or local >= total:
            continue
        owner = dict(offset=base.local_expert_offset, local=local, total=total)
        if _is_param_wrapper(module):
            if not getattr(module, "_ep_lora_sharded", False):
                continue
            for kind in ("A", "B"):
                for adapter in getattr(module, f"lora_{kind}", {}).values():
                    by_id[id(adapter.weight)] = dict(owner, kind=kind)
        else:
            for parameter in module.parameters(recurse=False):
                by_id[id(parameter)] = dict(owner, kind="expert")
            for name in ("gate_up_proj", "down_proj"):
                parametrizations = getattr(module, "parametrizations", {})
                if name in parametrizations:
                    by_id[id(parametrizations[name].original)] = dict(
                        owner, kind="expert"
                    )
    return {
        name: by_id[id(p)] for name, p in _parameters(model).items() if id(p) in by_id
    }


def _local(value):
    return value.to_local() if isinstance(value, DTensor) else value


def _layout(value, owner=None):
    local = _local(value)
    if isinstance(value, DTensor):
        if any(p.is_partial() for p in value.placements):
            raise ValueError("Cannot checkpoint an unreduced partial optimizer tensor")
        explicit_axes = None
        if any(type(p).__name__ == "_StridedShard" for p in value.placements):
            coordinate = value.device_mesh.get_coordinate()
            explicit_axes = []
            for dim, size in enumerate(value.shape):
                axis_shape = [1] * value.ndim
                axis_shape[dim] = size
                axis = torch.arange(size).reshape(axis_shape)
                for mesh_dim, placement in enumerate(value.placements):
                    if (
                        placement.is_shard()
                        or type(placement).__name__ == "_StridedShard"
                    ) and placement.dim % value.ndim == dim:
                        chunks, _ = placement._split_tensor(
                            axis, value.device_mesh.size(mesh_dim), with_padding=False
                        )
                        axis = chunks[coordinate[mesh_dim]]
                explicit_axes.append(axis.reshape(-1).tolist())
            shape = tuple(len(axis) for axis in explicit_axes)
            offset = (0,) * value.ndim
        else:
            shape, offset = compute_local_shape_and_global_offset(
                value.shape, value.device_mesh, value.placements
            )
        if tuple(shape) != tuple(local.shape):
            raise ValueError(
                "DTensor checkpoint shape does not match its local storage"
            )
    else:
        offset = (0,) * value.ndim
        explicit_axes = None
    shape = list(value.shape)
    if owner is not None:
        dim = 1 if owner["kind"] == "B" else 0
        shape[dim] = shape[dim] * owner["total"] // owner["local"]
    return dict(
        shape=shape,
        local_shape=list(local.shape),
        offset=list(offset),
        owner=owner,
        axes=explicit_axes,
    )


def _coordinates(layout):
    axes = [
        torch.arange(o, o + s)
        for o, s in zip(layout["offset"], layout["local_shape"], strict=True)
    ]
    if layout.get("axes") is not None:
        axes = [torch.tensor(axis, dtype=torch.int64) for axis in layout["axes"]]
    owner = layout["owner"]
    if owner is not None:
        if owner["kind"] == "B":
            axes[1] = (
                axes[1] // owner["local"] * owner["total"]
                + owner["offset"]
                + axes[1] % owner["local"]
            )
        else:
            factor = (
                1 if owner["kind"] == "expert" else layout["shape"][0] // owner["total"]
            )
            axes[0] += owner["offset"] * factor
    return axes


def _index(axes):
    return tuple(torch.meshgrid(*axes, indexing="ij")) if axes else ()


def _dtype(name):
    return getattr(torch, name.removeprefix("torch."))


def _check_errors(error):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise ValueError(
            "Full checkpoint failed: " + "; ".join(str(e) for e in errors if e)
        )


def _transfer(value, shape, dtype, source, receiver=0):
    count = math.prod(shape)
    output = torch.empty(shape, dtype=dtype) if dist.get_rank() == receiver else None
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    chunk_size = max(1, _TRANSFER_BYTES // torch.empty((), dtype=dtype).element_size())
    # Keep GPU staging bounded even when the complete checkpoint lives on CPU.
    for start in range(0, count, chunk_size):
        length = min(chunk_size, count - start)
        buffer = torch.empty(length, dtype=dtype, device=device)
        if dist.get_rank() == source:
            buffer.copy_(value.detach().reshape(-1)[start : start + length])
        dist.broadcast(buffer, src=source)
        if output is not None:
            output.reshape(-1)[start : start + length].copy_(buffer)
    return output


def _quantized(value):
    return type(_local(value)).__name__ == "OptimState8bit"


def _gather_tensor(value, owner=None):
    local = _local(value)
    error = None
    try:
        layout = _layout(value, owner)
    except ValueError as exc:
        layout, error = None, str(exc)
    _check_errors(error)
    quantized = _quantized(value)
    record = dict(layout, dtype=str(local.dtype), quantized=quantized)
    if quantized:
        record.update(block_size=local.block_size, signed=local.signed)
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, record)
    chunks, seen = [], set()
    for rank, info in enumerate(records):
        identity = repr(
            (info["owner"], info["offset"], info["local_shape"], info.get("axes"))
        )
        if identity in seen:
            continue
        seen.add(identity)
        chunk = dict(info)
        if info["quantized"]:
            for attr, shape, dtype in (
                ("codes", info["local_shape"], torch.uint8),
                (
                    "scale",
                    [math.prod(info["local_shape"]) // info["block_size"]],
                    torch.float32,
                ),
                ("qmap", [256], torch.float32),
            ):
                chunk[attr] = _transfer(
                    getattr(local, attr) if dist.get_rank() == rank else None,
                    shape,
                    dtype,
                    rank,
                )
        else:
            chunk["tensor"] = _transfer(
                local if dist.get_rank() == rank else None,
                info["local_shape"],
                _dtype(info["dtype"]),
                rank,
            )
        if dist.get_rank() == 0:
            chunks.append(chunk)
    full, error = None, None
    if dist.get_rank() == 0:
        if any(c["shape"] != records[0]["shape"] for c in chunks):
            error = "Checkpoint shards disagree on the global tensor shape"
        elif any(c["quantized"] for c in chunks):
            full = {_QUANTIZED: 1, "shape": records[0]["shape"], "chunks": chunks}
        else:
            full = torch.empty(records[0]["shape"], dtype=local.dtype)
            covered = torch.zeros(records[0]["shape"], dtype=torch.bool)
            for chunk in chunks:
                indices = _index(_coordinates(chunk))
                full[indices] = chunk["tensor"]
                covered[indices] = True
            if not covered.all():
                error = "Checkpoint does not cover every expert and shard"
    _check_errors(error)
    return full


def _gather_value(value, owner):
    if torch.is_tensor(value):
        if value.ndim == 0:
            scalar = _local(value).detach().cpu()
            records = [None] * dist.get_world_size()
            dist.all_gather_object(records, (owner, scalar))
            if all(torch.equal(scalar, other) for _, other in records):
                return scalar if dist.get_rank() == 0 else None
            chunks = {}
            for saved_owner, other in records:
                identity = repr(saved_owner)
                if saved_owner is None or (
                    identity in chunks
                    and not torch.equal(chunks[identity]["value"], other)
                ):
                    raise ValueError("Replicas disagree on a checkpoint scalar")
                chunks[identity] = {"owner": saved_owner, "value": other}
            return (
                {_OWNED_SCALAR: 1, "chunks": list(chunks.values())}
                if dist.get_rank() == 0
                else None
            )
        return _gather_tensor(value, owner)
    if isinstance(value, dict):
        return {k: _gather_value(value[k], owner) for k in sorted(value)}
    return copy.deepcopy(value) if dist.get_rank() == 0 else None


def full_model_state(model, adapter_only=False):
    # TODO: stream checkpoint assembly layerwise to reduce rank-0 host memory.
    adapter_only = adapter_only and is_peft_model(model)
    parameters = _parameters(model)
    if adapter_only:
        values = {name: p for name, p in parameters.items() if p.requires_grad}
    else:
        values = fsdp2_canonicalize_names(dict(model.state_dict()))
    owners = expert_ownership(model)
    state = {
        name: _gather_value(value, owners.get(name)) for name, value in values.items()
    }
    return state if dist.get_rank() == 0 else {}


def full_optimizer_state(model, optimizer):
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    local = get_optimizer_state_dict(
        model, optimizer, options=StateDictOptions(full_state_dict=False)
    )
    signature = [(n, sorted(v)) for n, v in sorted(local["state"].items())]
    signatures = [None] * dist.get_world_size()
    dist.all_gather_object(signatures, signature)
    if any(other != signature for other in signatures):
        raise ValueError("Optimizer state keys differ across checkpoint ranks")
    owners = expert_ownership(model)
    state = {
        name: _gather_value(values, owners.get(name))
        for name, values in sorted(local["state"].items())
    }
    if dist.get_rank() != 0:
        return {}
    return {
        "state": state,
        "param_groups": copy.deepcopy(local["param_groups"]),
        _VERSION: 1,
    }


def _metadata(value):
    if isinstance(value, dict) and _QUANTIZED in value:
        return {_QUANTIZED: 1, "shape": value["shape"]}
    if _quantized(value):
        return {_QUANTIZED: 1, "shape": list(value.shape)}
    if torch.is_tensor(value) and value.ndim > 0:
        return {"_tensor": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {k: _metadata(v) for k, v in value.items()}
    return value


def _select_tensor(value, request):
    if list(value.shape) != request["shape"]:
        raise ValueError(
            f"Saved shape {list(value.shape)} differs from global target shape {request['shape']}; a legacy EP checkpoint may contain only EP group 0"
        )
    return value[_index(_coordinates(request))].contiguous()


def _select_quantized(value, request, block_size, quantize):
    if _quantized(value):
        local = _local(value)
        value = {
            _QUANTIZED: 1,
            "shape": list(local.shape),
            "chunks": [
                dict(
                    _layout(local),
                    quantized=True,
                    codes=local.codes,
                    scale=local.scale,
                    qmap=local.qmap,
                    signed=local.signed,
                    dtype=str(local.dtype),
                    block_size=local.block_size,
                )
            ],
        }
    if value["shape"] != request["shape"]:
        raise ValueError(
            f"Quantized checkpoint shape {value['shape']} differs from global target shape {request['shape']}; the checkpoint may be missing EP owners"
        )
    shape = request["local_shape"]
    if math.prod(shape) == 0:
        return torch.empty(shape, dtype=torch.float32)
    floats = torch.empty(shape, dtype=torch.float32) if not quantize else None
    codes = torch.empty(shape, dtype=torch.uint8)
    scales = torch.empty(shape, dtype=torch.float32)
    covered = torch.zeros(shape, dtype=torch.bool)
    target_axes = _coordinates(request)
    reference = None
    for chunk in value["chunks"]:
        source_axes = _coordinates(chunk)
        if any(axis.numel() == 0 for axis in source_axes):
            continue
        target_indices, source_indices = [], []
        for source_axis, target_axis in zip(source_axes, target_axes, strict=True):
            positions = torch.searchsorted(source_axis, target_axis)
            valid = positions < source_axis.numel()
            valid &= (
                source_axis[positions.clamp(max=source_axis.numel() - 1)] == target_axis
            )
            target_indices.append(torch.nonzero(valid).flatten())
            source_indices.append(positions[valid])
        if any(axis.numel() == 0 for axis in target_indices):
            continue
        src, dst = _index(source_indices), _index(target_indices)
        if not quantize:
            if chunk["quantized"]:
                values = chunk["qmap"][chunk["codes"].long()] * chunk[
                    "scale"
                ].repeat_interleave(chunk["block_size"]).reshape(chunk["local_shape"])
            else:
                values = chunk["tensor"]
            floats[dst] = values[src]
            covered[dst] = True
            continue
        if not chunk["quantized"]:
            raise ValueError(
                "New optimizer shard mixes saved floating and quantized moments; an explicit conversion is required"
            )
        if reference is not None and (
            reference["signed"] != chunk["signed"]
            or not torch.equal(reference["qmap"], chunk["qmap"])
        ):
            raise ValueError(
                "Optimizer shards use different quantization lookup tables"
            )
        reference = chunk
        src, dst = _index(source_indices), _index(target_indices)
        codes[dst] = chunk["codes"][src]
        scales[dst] = (
            chunk["scale"]
            .repeat_interleave(chunk["block_size"])
            .reshape(chunk["local_shape"])[src]
        )
        covered[dst] = True
    if not covered.all():
        raise ValueError(
            "Optimizer checkpoint does not cover every target expert and shard"
        )
    if not quantize:
        return floats
    blocks = scales.reshape(-1, block_size)
    if not torch.equal(blocks, blocks[:, :1].expand_as(blocks)):
        raise ValueError(
            "Target optimizer layout regroups saved quantization blocks; resume with a compatible mesh or explicitly convert/requantize the optimizer state"
        )
    return dict(
        codes=codes,
        scale=blocks[:, 0].contiguous(),
        qmap=reference["qmap"],
        signed=reference["signed"],
        dtype=reference["dtype"],
    )


def _restore_tensor(value, parameter, owner, quantized=False, optimizer=None):
    local = _local(parameter)
    error = None
    try:
        request = _layout(parameter, owner)
    except ValueError as exc:
        request, error = None, str(exc)
    _check_errors(error)
    requests = [None] * dist.get_world_size()
    dist.all_gather_object(requests, request)
    rank = dist.get_rank()
    result = None
    raw = getattr(optimizer, "optimizer", optimizer)
    block_size = getattr(raw, "block_size", 256)
    if quantized and not _is_8bit_optimizer(raw):
        raise ValueError(
            "Quantized optimizer checkpoint requires a torchao 8-bit optimizer"
        )

    def select(target):
        quantize = (
            quantized
            and math.prod(target["local_shape"]) >= 4096
            and math.prod(target["local_shape"]) % block_size == 0
        )
        return (
            _select_quantized(value, target, block_size, quantize)
            if quantized
            else _select_tensor(value, target)
        )

    metadata, error = [None] * len(requests), None
    if rank == 0:
        try:
            for receiver, target in enumerate(requests):
                selected = select(target)
                metadata[receiver] = (
                    {
                        "_tensor": True,
                        "shape": list(selected.shape),
                        "dtype": str(selected.dtype),
                    }
                    if torch.is_tensor(selected)
                    else _metadata(selected)
                )
                del selected
        except ValueError as exc:
            error = str(exc)
    _check_errors(error)
    dist.broadcast_object_list(metadata, src=0)
    for receiver, target in enumerate(requests):
        # Recompute shards to avoid retaining a full copy for every replica.
        selected = select(target) if rank == 0 else None
        info = metadata[receiver]
        if isinstance(info, dict) and "codes" in info:
            components = {}
            for attr in ("codes", "scale", "qmap"):
                meta = info[attr]
                components[attr] = _transfer(
                    selected[attr] if rank == 0 else None,
                    meta["shape"],
                    _dtype(meta["dtype"]),
                    0,
                    receiver,
                )
            if rank == receiver:
                from torchao.optim.subclass_8bit import OptimState8bit

                result = OptimState8bit(
                    *(
                        components[a].to(local.device)
                        for a in ("codes", "scale", "qmap")
                    ),
                    info["signed"],
                    dtype=_dtype(info["dtype"]),
                )
        else:
            meta = info
            tensor = _transfer(
                selected if rank == 0 else None,
                meta["shape"],
                _dtype(meta["dtype"]),
                0,
                receiver,
            )
            if rank == receiver:
                result = tensor.to(local.device)
    if isinstance(parameter, DTensor):
        result = DTensor.from_local(
            result,
            parameter.device_mesh,
            parameter.placements,
            shape=parameter.shape,
            stride=parameter.stride(),
            run_check=False,
        ).to(local.device)
    return result


def restore_model_state(model, state, adapter_only=False):
    adapter_only = adapter_only and is_peft_model(model)
    parameters = _parameters(model)
    targets = (
        {n: p for n, p in parameters.items() if p.requires_grad}
        if adapter_only
        else fsdp2_canonicalize_names(dict(model.state_dict()))
    )
    owners = expert_ownership(model)
    metadata = [_metadata(state) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    missing = set(targets) - set(metadata[0])
    _check_errors(f"Missing model parameters: {sorted(missing)}" if missing else None)
    with torch.no_grad():
        for name, parameter in targets.items():
            restored = _restore_tensor(
                state[name] if dist.get_rank() == 0 else None,
                parameter,
                owners.get(name),
            )
            _local(parameter).copy_(_local(restored))
    return torch.nn.modules.module._IncompatibleKeys([], [])


def _restore_value(value, meta, parameter, owner, optimizer):
    if isinstance(meta, dict) and _OWNED_SCALAR in meta:
        selected, covered, error = [], set(), None
        if owner is None:
            error = "EP optimizer counters require expert ownership on restore"
        else:
            target = set(range(owner["offset"], owner["offset"] + owner["local"]))
            for chunk in meta["chunks"]:
                source = chunk["owner"]
                overlap = target.intersection(
                    range(source["offset"], source["offset"] + source["local"])
                )
                if overlap:
                    selected.append(chunk["value"])
                    covered.update(overlap)
            if covered != target:
                error = "Optimizer counters do not cover the target experts"
            elif any(not torch.equal(selected[0], v) for v in selected[1:]):
                error = "Target EP ownership merges different optimizer step counters"
        _check_errors(error)
        return copy.deepcopy(selected[0])
    if isinstance(meta, dict) and (_QUANTIZED in meta or "_tensor" in meta):
        return _restore_tensor(
            value, parameter, owner, quantized=_QUANTIZED in meta, optimizer=optimizer
        )
    if isinstance(meta, dict):
        return {
            k: _restore_value(
                value[k] if dist.get_rank() == 0 else None,
                v,
                parameter,
                owner,
                optimizer,
            )
            for k, v in meta.items()
        }
    return copy.deepcopy(meta)


def restore_optimizer_state(model, optimizer, state):
    metadata = [_metadata(state) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    meta = metadata[0]
    if meta.get(_VERSION, 1) != 1:
        raise ValueError("Unsupported full optimizer checkpoint version")
    parameters, owners = _parameters(model), expert_ownership(model)
    missing = set(meta["state"]) - set(parameters)
    _check_errors(
        f"Missing optimizer parameters: {sorted(missing)}" if missing else None
    )
    groups = meta["param_groups"]
    names_by_id = {id(p): n for n, p in parameters.items()}
    current = [
        [names_by_id[id(p)] for p in g["params"]] for g in optimizer.param_groups
    ]
    saved = [g["params"] for g in groups]
    _check_errors(
        "Optimizer parameter groups differ from the checkpoint"
        if current != saved
        else None
    )
    loaded = {
        name: _restore_value(
            state["state"][name] if dist.get_rank() == 0 else None,
            values,
            parameters[name],
            owners.get(name),
            optimizer,
        )
        for name, values in meta["state"].items()
    }
    optimizer.load_state_dict(dict(state=loaded, param_groups=groups))
    # Quantized moments dequantized for a smaller shard must retain FP32 precision.
    for name, values in loaded.items():
        for key, value in values.items():
            if torch.is_tensor(value) and value.ndim > 0 and not _quantized(value):
                if (
                    isinstance(meta["state"][name][key], dict)
                    and _QUANTIZED in meta["state"][name][key]
                ):
                    optimizer.state[parameters[name]][key] = value


def _full(plugin):
    return plugin.fsdp_version == 2 and "FULL_STATE_DICT" in str(plugin.state_dict_type)


def _model_needs_ownership(model):
    return bool(expert_ownership(model)) or bool(
        getattr(model, "_moe_experts_quantized", False) and is_peft_model(model)
    )


def _is_8bit_optimizer(optimizer):
    raw = getattr(optimizer, "optimizer", optimizer)
    return any(
        cls.__module__.startswith("torchao.optim.")
        and cls.__name__ in {"Adam8bit", "AdamW8bit"}
        for cls in type(raw).__mro__
    )


def _optimizer_needs_ownership(model, optimizer):
    return bool(expert_ownership(model)) or _is_8bit_optimizer(optimizer)


def patch_fsdp2_full_checkpoint():
    """Keep Accelerate's full file layout while preserving every EP owner and 8-bit block."""
    import accelerate
    from accelerate.utils import fsdp_utils
    from transformers import trainer

    if getattr(fsdp_utils.save_fsdp_optimizer, "_axolotl_full_checkpoint", False):
        return
    for operation in (
        "save_fsdp_model",
        "load_fsdp_model",
        "save_fsdp_optimizer",
        "load_fsdp_optimizer",
    ):
        original = getattr(fsdp_utils, operation)

        def wrap(original, operation):
            @wraps(original)
            def checkpoint(*args, **kwargs):
                bound = inspect.signature(original).bind(*args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                plugin, accelerator = values["fsdp_plugin"], values["accelerator"]
                optimizer_operation = "optimizer" in operation
                model = values["model"]
                optimizer = values.get("optimizer")
                if not _full(plugin):
                    return original(*args, **kwargs)
                needed = (
                    _optimizer_needs_ownership(model, optimizer)
                    if optimizer_operation
                    else _model_needs_ownership(model)
                )
                if not needed:
                    return original(*args, **kwargs)
                adapter_only = values.get("adapter_only", False) and is_peft_model(
                    model
                )
                packed = {}
                if not optimizer_operation and not adapter_only:
                    from .fsdp2_bnb_checkpoint import packed_parameters

                    error = None
                    try:
                        packed = packed_parameters(model)
                    except ValueError as exc:
                        error = str(exc)
                    _check_errors(error)
                path = Path(
                    values["output_dir"] if "save" in operation else values["input_dir"]
                )
                index = values.get(
                    "optimizer_index" if optimizer_operation else "model_index", 0
                )
                basename = "optimizer" if optimizer_operation else "pytorch_model_fsdp"
                filename = path / (
                    f"{basename}_{index}.bin" if index else f"{basename}.bin"
                )
                if "save" in operation:
                    if packed:
                        from .fsdp2_bnb_checkpoint import full_packed_model_state

                        state = full_packed_model_state(model, packed)
                    else:
                        state = (
                            full_optimizer_state(model, optimizer)
                            if optimizer_operation
                            else full_model_state(model, adapter_only)
                        )
                    error = None
                    if accelerator.is_main_process:
                        try:
                            path.mkdir(parents=True, exist_ok=True)
                            torch.save(state, filename)
                        except (OSError, RuntimeError) as exc:
                            error = str(exc)
                    _check_errors(error)
                    accelerator.wait_for_everyone()
                    return None
                accelerator.wait_for_everyone()
                state, error = {}, None
                if accelerator.is_main_process:
                    try:
                        state = torch.load(
                            filename, map_location="cpu", weights_only=True
                        )
                    except Exception as exc:  # pylint: disable=broad-except
                        error = str(exc)
                _check_errors(error)
                envelope = [
                    "_axolotl_full_model_version" in state
                    if accelerator.is_main_process and not optimizer_operation
                    else False
                ]
                dist.broadcast_object_list(envelope, src=0)
                if envelope[0] and adapter_only:
                    error = None
                    if accelerator.is_main_process:
                        if state.get("_axolotl_full_model_version") != 1:
                            error = "Unsupported native packed model checkpoint version"
                        elif not isinstance(state.get("state"), dict):
                            error = "Packed checkpoint is missing ordinary model state"
                    _check_errors(error)
                    result = restore_model_state(
                        model,
                        state["state"] if accelerator.is_main_process else {},
                        adapter_only=True,
                    )
                elif packed:
                    from .fsdp2_bnb_checkpoint import restore_packed_model_state

                    result = restore_packed_model_state(model, state, packed)
                else:
                    result = (
                        restore_optimizer_state(model, optimizer, state)
                        if optimizer_operation
                        else restore_model_state(model, state, adapter_only)
                    )
                accelerator.wait_for_everyone()
                return result

            checkpoint._axolotl_full_checkpoint = True
            return checkpoint

        patched = wrap(original, operation)
        setattr(fsdp_utils, operation, patched)
        setattr(trainer, operation, patched)
        setattr(accelerate.accelerator, operation, patched)
        setattr(accelerate.utils, operation, patched)
    original_get = accelerate.Accelerator.get_state_dict

    @wraps(original_get)
    def get_state_dict(self, model, unwrap=True):
        plugin = getattr(self.state, "fsdp_plugin", None)
        if plugin is not None and _full(plugin) and _model_needs_ownership(model):
            return full_model_state(model)
        return original_get(self, model, unwrap=unwrap)

    accelerate.Accelerator.get_state_dict = get_state_dict
