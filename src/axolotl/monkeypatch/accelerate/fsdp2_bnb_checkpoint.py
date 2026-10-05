# Copyright 2026 Axolotl AI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Native full checkpoints for owner-local bitsandbytes packed parameters."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.utils.parametrize as P
from accelerate.utils.fsdp_utils import fsdp2_canonicalize_names

_VERSION = "_axolotl_full_model_version"


def _canonical(name):
    return next(iter(fsdp2_canonicalize_names({name: None})))


def packed_parameters(model):
    from .fsdp2_checkpoint import _local, expert_ownership

    owners, result, handled = expert_ownership(model), {}, set()
    for prefix, module in model.named_modules():
        prefix = prefix + "." if prefix else ""
        for name, stack in getattr(module, "parametrizations", {}).items():
            entries = [
                entry
                for entry in stack
                if type(entry).__name__
                in {"Bnb4bitParametrization", "Bnb8bitParametrization"}
            ]
            if not entries:
                continue
            if len(entries) != 1 or len(stack) != 1:
                raise ValueError(
                    "Packed checkpoint requires a single BNB parametrization"
                )
            parameter, entry = stack.original, entries[0]
            key = _canonical(f"{prefix}parametrizations.{name}.original")
            mode = "4bit" if hasattr(entry, "quant_state") else "8bit"
            result[key] = dict(
                parameter=parameter,
                owner=owners.get(key),
                mode=mode,
                module=module,
                name=name,
                entry=entry,
                clean=_canonical(prefix + name),
            )
            handled.add(id(parameter))
        for name, parameter in module.named_parameters(recurse=False):
            if id(parameter) in handled:
                continue
            local = _local(parameter)
            if type(local).__name__ == "Int8Params":
                from .fsdp2_checkpoint import _model_needs_ownership

                if owners or _model_needs_ownership(model):
                    raise ValueError(
                        "Ownership-aware full checkpoints do not support dense BNB Int8Params"
                    )
                continue
            state = getattr(local, "quant_state", None)
            if state is None and name == "weight":
                state = getattr(module, "quant_state", None)
            if state is None:
                continue
            key = _canonical(prefix + name)
            result[key] = dict(
                parameter=parameter,
                owner=owners.get(key),
                mode="4bit",
                module=module,
                name=name,
                entry=None,
                clean=key,
            )
            handled.add(id(parameter))
    return result


def _quant_state(descriptor):
    from .fsdp2_checkpoint import _local

    entry = descriptor["entry"]
    if entry is not None:
        return entry.quant_state
    local = _local(descriptor["parameter"])
    state = getattr(local, "quant_state", None)
    return state if state is not None else descriptor["module"].quant_state


def _encode_state(state):
    return dict(
        shape=list(state.shape) if state.shape is not None else None,
        dtype=str(state.dtype) if state.dtype is not None else None,
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        absmax=state.absmax,
        code=state.code,
        offset=state.offset,
        state2=_encode_state(state.state2) if state.state2 is not None else None,
    )


def _decode_state(saved, device):
    from bitsandbytes.functional import QuantState

    from .fsdp2_checkpoint import _dtype

    return QuantState(
        absmax=saved["absmax"].to(device),
        shape=torch.Size(saved["shape"]) if saved["shape"] is not None else None,
        code=saved["code"].to(device) if saved["code"] is not None else None,
        blocksize=saved["blocksize"],
        quant_type=saved["quant_type"],
        dtype=_dtype(saved["dtype"]) if saved["dtype"] is not None else None,
        offset=saved["offset"].to(device)
        if torch.is_tensor(saved["offset"])
        else saved["offset"],
        state2=_decode_state(saved["state2"], device)
        if saved["state2"] is not None
        else None,
    )


def _summary(value):
    if torch.is_tensor(value):
        return {"_tensor": True, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _summary(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_summary(item) for item in value]
    return value


def _transfer_tree(value, summary, source, receiver=0):
    from .fsdp2_checkpoint import _dtype, _transfer

    if isinstance(summary, dict) and "_tensor" in summary:
        return _transfer(
            value, summary["shape"], _dtype(summary["dtype"]), source, receiver
        )
    if isinstance(summary, dict):
        return {
            key: _transfer_tree(
                value[key] if dist.get_rank() == source else None,
                item,
                source,
                receiver,
            )
            for key, item in summary.items()
        }
    return summary


def _ordinary_state(model, descriptors):
    values = fsdp2_canonicalize_names(dict(model.state_dict()))
    for key, descriptor in descriptors.items():
        clean = descriptor["clean"]
        internal = key.removesuffix(".original")
        values = {
            name: value
            for name, value in values.items()
            if name != key
            and name != clean
            and not name.startswith(clean + ".")
            and not name.startswith(internal + ".")
        }
    return values


def _descriptor_info(descriptor):
    from .fsdp2_checkpoint import _layout, _local

    parameter = descriptor["parameter"]
    shape = (
        list(_quant_state(descriptor).shape)
        if descriptor["mode"] == "4bit"
        else list(parameter.shape)
    )
    metadata = (
        _encode_state(_quant_state(descriptor))
        if descriptor["mode"] == "4bit"
        else {"row_stats": descriptor["entry"].row_stats}
    )
    return dict(
        owner=descriptor["owner"],
        mode=descriptor["mode"],
        logical_shape=shape,
        layout=_layout(parameter),
        dtype=str(_local(parameter).dtype),
        metadata=_summary(metadata),
    ), metadata


def _validate_record(record):
    shape = record["logical_shape"]
    count = math.prod(shape)
    layout = record["layout"]
    physical = math.prod(layout["shape"])
    owner = record["owner"]
    if "packed" in record and (
        record["packed"]["shape"] != layout["shape"]
        or record["packed"]["dtype"] != record["dtype"]
    ):
        raise ValueError("Packed checkpoint tensor differs from its physical layout")
    if owner is not None and (
        owner["kind"] != "expert"
        or shape[0] != owner["local"]
        or owner["offset"] < 0
        or owner["offset"] + owner["local"] > owner["total"]
    ):
        raise ValueError("Invalid packed expert ownership or logical shape")
    if record["mode"] == "4bit":
        from .fsdp2_checkpoint import _dtype

        itemsize = torch.empty((), dtype=_dtype(record["dtype"])).element_size()
        metadata = record["metadata"]
        if physical * itemsize != (count + 1) // 2:
            raise ValueError("Packed BNB storage does not cover its logical shape")
        if metadata["shape"] != shape or not metadata["blocksize"]:
            raise ValueError("Invalid BNB QuantState shape or blocksize")
        if (
            metadata["quant_type"] not in {"nf4", "fp4"}
            or metadata["dtype"]
            not in {"torch.float32", "torch.float16", "torch.bfloat16"}
            or metadata["code"]["shape"] != [16]
            or metadata["code"]["dtype"] != "torch.float32"
            or metadata["absmax"]["dtype"]
            != ("torch.uint8" if metadata["state2"] is not None else "torch.float32")
        ):
            raise ValueError("Invalid BNB quantization format or codebook")
        scales = math.prod(metadata["absmax"]["shape"])
        if scales != math.ceil(count / metadata["blocksize"]):
            raise ValueError("BNB scales do not cover the packed weights")
        nested = metadata["state2"]
        if nested is not None and (
            metadata["offset"] is None
            or not nested["blocksize"]
            or math.prod(nested["absmax"]["shape"])
            != math.ceil(scales / nested["blocksize"])
        ):
            raise ValueError("Invalid nested BNB quantization metadata")
        if nested is not None and (
            nested["absmax"]["dtype"] != "torch.float32"
            or nested["code"]["shape"] != [256]
            or nested["code"]["dtype"] != "torch.float32"
            or nested["state2"] is not None
        ):
            raise ValueError("Invalid nested BNB scales or codebook")
    elif physical != count:
        raise ValueError("Packed 8-bit storage does not cover its logical shape")
    elif (
        record["dtype"] != "torch.int8"
        or record["metadata"]["row_stats"]["shape"] != [math.prod(shape[:-1])]
        or record["metadata"]["row_stats"]["dtype"] != "torch.float32"
    ):
        raise ValueError("Invalid packed 8-bit row statistics")


def _gather_packed(descriptor):
    from .fsdp2_checkpoint import (
        _check_errors,
        _coordinates,
        _dtype,
        _index,
        _local,
        _transfer,
    )

    info, metadata, error = None, None, None
    try:
        info, metadata = _descriptor_info(descriptor)
        _validate_record(info)
    except (ValueError, AttributeError, TypeError) as exc:
        error = str(exc)
    _check_errors(error)
    records = [None] * dist.get_world_size()
    dist.all_gather_object(records, info)
    owners = {}
    for rank, record in enumerate(records):
        identity = repr(record["owner"])
        owners.setdefault(identity, []).append((rank, record))
    saved = []
    for members in owners.values():
        source, reference = members[0]
        error = None
        if any(
            record["mode"] != reference["mode"]
            or record["logical_shape"] != reference["logical_shape"]
            or record["layout"]["shape"] != reference["layout"]["shape"]
            or record["dtype"] != reference["dtype"]
            or record["metadata"] != reference["metadata"]
            for _, record in members
        ):
            error = "Packed checkpoint replicas disagree on quantization layout"
        _check_errors(error)
        payload = _transfer_tree(metadata, reference["metadata"], source)
        packed = (
            torch.empty(reference["layout"]["shape"], dtype=_dtype(reference["dtype"]))
            if dist.get_rank() == 0
            else None
        )
        covered = (
            torch.zeros(reference["layout"]["shape"], dtype=torch.bool)
            if dist.get_rank() == 0
            else None
        )
        seen = set()
        for rank, record in members:
            layout = record["layout"]
            identity = repr(layout)
            if identity in seen:
                continue
            seen.add(identity)
            tensor = _transfer(
                _local(descriptor["parameter"]) if dist.get_rank() == rank else None,
                layout["local_shape"],
                _dtype(record["dtype"]),
                rank,
            )
            if dist.get_rank() == 0:
                indices = _index(_coordinates(layout))
                packed[indices], covered[indices] = tensor, True
        _check_errors(
            "Packed checkpoint is missing physical shards"
            if dist.get_rank() == 0 and not covered.all()
            else None
        )
        if dist.get_rank() == 0:
            saved.append(dict(reference, packed=packed, metadata=payload))
    if dist.get_rank() == 0:
        _validate_owners(saved)
    return saved


def _validate_owners(records):
    if not records:
        raise ValueError("Packed checkpoint is missing expert owners")
    first = records[0]
    owner = first["owner"]
    if owner is None:
        if len(records) != 1:
            raise ValueError("Packed dense checkpoint has multiple owners")
        return
    covered = set()
    for record in records:
        other = record["owner"]
        if (
            other is None
            or other["total"] != owner["total"]
            or record["logical_shape"][1:] != first["logical_shape"][1:]
        ):
            raise ValueError("Packed checkpoint owners disagree on global shape")
        indices = set(range(other["offset"], other["offset"] + other["local"]))
        if covered.intersection(indices):
            raise ValueError("Packed checkpoint contains overlapping expert owners")
        covered.update(indices)
    if covered != set(range(owner["total"])):
        raise ValueError("Packed checkpoint is missing expert owners")


def full_packed_model_state(model, descriptors):
    from .fsdp2_checkpoint import _check_errors, _gather_value, expert_ownership

    owners = expert_ownership(model)
    state = {
        name: _gather_value(value, owners.get(name))
        for name, value in _ordinary_state(model, descriptors).items()
    }
    quantized = {}
    for name, descriptor in descriptors.items():
        error = None
        try:
            quantized[name] = _gather_packed(descriptor)
        except ValueError as exc:
            error = str(exc)
        _check_errors(error)
    return (
        {_VERSION: 1, "state": state, "quantized": quantized}
        if dist.get_rank() == 0
        else {}
    )


def _update_state(current, restored):
    if current.state2 is not None and restored.state2 is not None:
        _update_state(current.state2, restored.state2)
        restored.state2 = current.state2
    current.__dict__.update(restored.__dict__)


def restore_packed_model_state(model, state, descriptors):
    from .fsdp2_checkpoint import (
        _check_errors,
        _local,
        _restore_tensor,
        _select_tensor,
        expert_ownership,
    )

    rank = dist.get_rank()
    metadata = [_summary(state) if rank == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    saved = metadata[0]
    error = None
    if saved.get(_VERSION) != 1:
        error = "Unsupported native packed model checkpoint version"
    elif set(saved["quantized"]) != set(descriptors):
        error = "Packed checkpoint parameters differ from the model"
    targets = _ordinary_state(model, descriptors)
    if error is None and set(saved["state"]) != set(targets):
        error = "Ordinary model checkpoint entries differ from the model"
    _check_errors(error)
    pending = []
    for name, descriptor in descriptors.items():
        request, error = None, None
        try:
            request, _ = _descriptor_info(descriptor)
            _validate_owners(saved["quantized"][name])
            for record in saved["quantized"][name]:
                _validate_record(record)
        except (ValueError, AttributeError, TypeError, KeyError, IndexError) as exc:
            error = str(exc)
        _check_errors(error)
        requests = [None] * dist.get_world_size()
        dist.all_gather_object(requests, request)
        for receiver, target in enumerate(requests):
            records = saved["quantized"][name]
            matches = [
                i
                for i, record in enumerate(records)
                if record["owner"] == target["owner"]
            ]
            error = None
            if len(matches) != 1:
                error = "Packed checkpoint requires the same EP ownership ranges; changed EP grouping needs an explicit conversion"
            else:
                index = matches[0]
                record = records[index]
                if any(record[key] != target[key] for key in ("mode", "dtype")):
                    error = "Packed checkpoint format, logical shape or storage dtype differs"
                elif record["layout"]["shape"] != target["layout"]["shape"]:
                    error = "Packed checkpoint physical shape differs"
            _check_errors(error)
            selected, error = None, None
            if rank == 0:
                try:
                    selected = _select_tensor(
                        state["quantized"][name][index]["packed"], target["layout"]
                    )
                except ValueError as exc:
                    error = str(exc)
            _check_errors(error)
            from .fsdp2_checkpoint import _dtype, _transfer

            packed = _transfer(
                selected,
                target["layout"]["local_shape"],
                _dtype(target["dtype"]),
                0,
                receiver,
            )
            quant_state = _transfer_tree(
                state["quantized"][name][index]["metadata"] if rank == 0 else None,
                record["metadata"],
                0,
                receiver,
            )
            if rank == receiver:
                pending.append((descriptor, packed, quant_state))
    owners = expert_ownership(model)
    with torch.no_grad():
        for name, parameter in targets.items():
            restored = _restore_tensor(
                state["state"][name] if rank == 0 else None, parameter, owners.get(name)
            )
            _local(parameter).copy_(_local(restored))
        for descriptor, packed, quant_state in pending:
            local = _local(descriptor["parameter"])
            local.copy_(packed)
            if descriptor["mode"] == "4bit":
                current = _quant_state(descriptor)
                # FSDP CPU offload keeps quantization metadata on the compute device.
                restored = _decode_state(quant_state, current.absmax.device)
                _update_state(current, restored)
                entry = descriptor["entry"]
                if entry is not None:
                    entry.quant_state = current
                else:
                    local.quant_state = current
                    descriptor["module"].quant_state = current
            else:
                descriptor["entry"].row_stats.copy_(
                    quant_state["row_stats"].to(local.device)
                )
            if descriptor["entry"] is not None:
                P._cache.pop((id(descriptor["module"]), descriptor["name"]), None)
    return torch.nn.modules.module._IncompatibleKeys([], [])
