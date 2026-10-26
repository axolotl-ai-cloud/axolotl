# Copyright 2026 Axolotl AI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Ownership-aware full FSDP2 model and quantized optimizer checkpoints."""

from __future__ import annotations

import contextlib
import copy
import inspect
import math
import os
import time
from functools import wraps
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate.utils.fsdp_utils import fsdp2_canonicalize_names
from accelerate.utils.modeling import is_peft_model
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

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
            for name in ("gate_up_proj", "up_proj", "down_proj"):
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


def _device_backends(group=None) -> dict[str, str]:
    """``{device type: backend name}`` registered on ``group`` (default: WORLD)."""
    try:
        config = str(dist.get_backend_config(group))
    except Exception:  # noqa: BLE001  pylint: disable=broad-except
        config = ""
    backends = dict(
        item.strip().split(":", 1) for item in config.split(",") if ":" in item
    )
    if backends:
        return {device: name.lower() for device, name in backends.items()}
    for device_type in ("cuda", "cpu"):
        try:
            pg = group or dist.distributed_c10d._get_default_group()
            backend = pg._get_backend(torch.device(device_type))
        except Exception:  # noqa: BLE001  pylint: disable=broad-except
            backend = None
        if backend is not None:
            backends[device_type] = (
                type(backend).__name__.removeprefix("ProcessGroup").lower()
            )
    return backends


def collective_device(group=None) -> torch.device:
    """The device to stage tensors on for collectives over ``group``.

    ``dist.get_backend()`` is not usable for this: a default group created implicitly by
    ``init_device_mesh`` (no backend given) reports ``"undefined"`` while registering only
    the accelerator backend (``cuda:nccl``), so CPU tensors fail with "No backend type
    associated with device type cpu".
    """
    backends = _device_backends(group)
    if not backends:
        try:
            name = str(dist.get_backend(group))
        except Exception:  # noqa: BLE001  pylint: disable=broad-except
            name = ""
        backends = {"cuda": "nccl"} if "nccl" in name else {"cpu": name or "gloo"}
    cuda = backends.get("cuda")
    # gloo registers for cuda too, but over CPU tensors it skips the device round trip
    if (
        cuda is not None
        and torch.cuda.is_available()
        and (cuda != "gloo" or "cpu" not in backends)
    ):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _transfer_device():
    return collective_device()


_STAGING: dict = {}


def _staging(dtype, count):
    buffer = _STAGING.get(dtype)
    if buffer is None or buffer.numel() < count:
        buffer = _STAGING[dtype] = torch.empty(count, dtype=dtype, pin_memory=True)
    return buffer[:count]


def _segments(shape, start, stop):
    """Indices of the row-major boxes covering flat elements ``[start, stop)`` of ``shape``."""
    if start >= stop:
        return
    if not shape:
        yield ()
        return
    inner = math.prod(shape[1:])
    row, column = divmod(start, inner)
    end_row, end_column = divmod(stop, inner)
    if row == end_row:
        for index in _segments(shape[1:], column, end_column):
            yield (row, *index)
        return
    if column:
        for index in _segments(shape[1:], column, inner):
            yield (row, *index)
        row += 1
    if end_row > row:
        yield (slice(row, end_row),)
    if end_column:
        for index in _segments(shape[1:], 0, end_column):
            yield (end_row, *index)


# measured on 32-core B200 pods: 8-16 threads copy ~14x faster than 32 (OpenMP oversubscription)
_MAX_COPY_THREADS = 8


@contextlib.contextmanager
def _copy_threads():
    """Raise torch's intra-op threads for the receiver's CPU copies, then restore them.

    torchrun starts trainers with OMP_NUM_THREADS=1, which leaves the assembly of a large
    checkpoint on one core.
    """
    before = torch.get_num_threads()
    try:
        available = len(os.sched_getaffinity(0))
    except AttributeError:
        available = os.cpu_count() or 1
    wanted = min(_MAX_COPY_THREADS, available)
    if wanted > before:
        torch.set_num_threads(wanted)
    if _TIMING is not None:
        _TIMING["threads"] = torch.get_num_threads()
    try:
        yield
    finally:
        if torch.get_num_threads() != before:
            torch.set_num_threads(before)


# AXOLOTL_CHECKPOINT_TIMING=1: rank 0 debug-logs where assembly time goes (per phase, slowest tensors)
_TIMING: dict | None = None


def _timed(phase, start, nbytes=0):
    if _TIMING is not None:
        now = time.perf_counter()
        _TIMING["seconds"][phase] = _TIMING["seconds"].get(phase, 0.0) + now - start
        _TIMING["bytes"][phase] = _TIMING["bytes"].get(phase, 0) + nbytes
        return now
    return start


def _log_timing(total):
    if _TIMING is None or dist.get_rank() != 0:
        return
    phases = ", ".join(
        f"{phase} {seconds:.1f}s/{_TIMING['bytes'][phase] / 2**30:.1f}GiB"
        for phase, seconds in sorted(_TIMING["seconds"].items(), key=lambda kv: -kv[1])
    )
    LOG.debug(
        "checkpoint assembly %.1fs over %d tensors (%d copy threads): %s",
        total,
        len(_TIMING["tensors"]),
        _TIMING.get("threads", torch.get_num_threads()),
        phases,
    )
    for row in sorted(_TIMING["tensors"], key=lambda r: -r[0])[:8]:
        LOG.debug("checkpoint assembly slowest: %.2fs %s", row[0], row[1])


def _write(out, start, data):
    offset = 0
    for index in _segments(list(out.shape), start, start + data.numel()):
        view = out[index]
        view.copy_(data[offset : offset + view.numel()].view(view.shape))
        offset += view.numel()


def _transfer(value, shape, dtype, source, receiver=0, out=None):
    """Send ``source``'s tensor to ``receiver``'s CPU, into ``out`` (any strided view) when given."""
    count = math.prod(shape)
    rank = dist.get_rank()
    if source == receiver and rank != receiver:
        return None
    if rank == receiver and out is None:
        out = torch.empty(shape, dtype=dtype)
    flat = value.detach().reshape(-1) if rank == source else None
    chunk_size = max(1, _TRANSFER_BYTES // torch.empty((), dtype=dtype).element_size())
    threads = _copy_threads() if rank == receiver else contextlib.nullcontext()
    with threads:
        _transfer_chunks(flat, out, count, chunk_size, dtype, source, receiver, rank)
    return out if rank == receiver else None


def _transfer_chunks(flat, out, count, chunk_size, dtype, source, receiver, rank):
    buffer = None
    # Keep GPU staging bounded even when the complete checkpoint lives on CPU.
    for start in range(0, count, chunk_size):
        length = min(chunk_size, count - start)
        if source == receiver:
            piece = flat[start : start + length]
        else:
            if buffer is None:
                buffer = torch.empty(
                    min(chunk_size, count), dtype=dtype, device=_transfer_device()
                )
            piece = buffer[:length]
            if rank == source:
                piece.copy_(flat[start : start + length])
            began = time.perf_counter() if _TIMING is not None else 0.0
            dist.broadcast(piece, src=source)
            if _TIMING is not None:
                if piece.device.type == "cuda":
                    torch.cuda.current_stream(piece.device).synchronize()
                _timed("broadcast", began, piece.numel() * piece.element_size())
        if rank != receiver:
            continue
        began = time.perf_counter() if _TIMING is not None else 0.0
        if piece.device.type != "cpu":
            staged = _staging(dtype, length)
            staged.copy_(piece, non_blocking=True)
            torch.cuda.current_stream(piece.device).synchronize()
            piece = staged
            began = _timed("stage", began, piece.numel() * piece.element_size())
        _write(out, start, piece)
        _timed("write", began, piece.numel() * piece.element_size())


def _quantized(value):
    return type(_local(value)).__name__ == "OptimState8bit"


def _tensor_record(value, owner):
    local = _local(value)
    record = dict(
        _layout(value, owner), dtype=str(local.dtype), quantized=_quantized(value)
    )
    if record["quantized"]:
        record.update(block_size=local.block_size, signed=local.signed)
    return record


def _gather_records(items):
    """Every rank's layout of each ``(value, owner)`` in one collective; raises on every rank
    when any rank failed."""
    records, error = None, None
    try:
        records = [_tensor_record(value, owner) for value, owner in items]
    except ValueError as exc:
        error = str(exc)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (error, records))
    errors = [error for error, _ in gathered if error]
    if errors:
        raise ValueError("Full checkpoint failed: " + "; ".join(errors))
    return [[ranks[i] for _, ranks in gathered] for i in range(len(items))]


def _box(info):
    """``(view shape, starts, lengths)`` of the dense box a chunk fills, or ``None`` when its
    coordinates do not form one (``_StridedShard``)."""
    axes = _coordinates(info)
    shape = list(info["shape"])
    owner = info["owner"]
    if owner is not None and owner["kind"] == "B":
        # experts are innermost in the [out, r * E] columns: a box of [out, r, E]
        total, local = owner["total"], owner["local"]
        columns = axes[1]
        if len(shape) < 2 or shape[1] % total or columns.numel() % local:
            return None
        rows = columns.numel() // local
        row, expert = divmod(int(columns[0]), total) if columns.numel() else (0, 0)
        rows_axis = torch.arange(row, row + rows)
        experts_axis = torch.arange(expert, expert + local)
        if not torch.equal(
            columns, (rows_axis[:, None] * total + experts_axis).flatten()
        ):
            return None
        shape[1:2] = [shape[1] // total, total]
        axes[1:2] = [rows_axis, experts_axis]
    starts = []
    for axis in axes:
        start = int(axis[0]) if axis.numel() else 0
        if not torch.equal(axis, torch.arange(start, start + axis.numel())):
            return None
        starts.append(start)
    return shape, starts, [axis.numel() for axis in axes]


def _boxes(chunks):
    """Shared view shape and per-chunk ``(starts, lengths)``, or ``None`` to fall back to index
    assembly; raises unless the boxes tile the tensor exactly once."""
    boxes = [_box(info) for _, info in chunks]
    if any(box is None or box[0] != boxes[0][0] for box in boxes):
        return None
    view = boxes[0][0]
    filled = [box for box in boxes if math.prod(box[2])]
    if any(
        start < 0 or start + length > size
        for _, starts, lengths in filled
        for start, length, size in zip(starts, lengths, view, strict=True)
    ):
        raise ValueError("Full checkpoint failed: Checkpoint shard exceeds the tensor")
    for i, (_, starts, lengths) in enumerate(filled):
        for _, other_starts, other_lengths in filled[i + 1 :]:
            if all(
                a < b + m and b < a + n
                for a, n, b, m in zip(
                    starts, lengths, other_starts, other_lengths, strict=True
                )
            ):
                raise ValueError("Full checkpoint failed: Checkpoint shards overlap")
    if sum(math.prod(lengths) for _, _, lengths in filled) != math.prod(view):
        raise ValueError(
            "Full checkpoint failed: Checkpoint does not cover every expert and shard"
        )
    return view, [(starts, lengths) for _, starts, lengths in boxes]


def _gather_tensor(value, owner=None, records=None):
    if records is None:
        (records,) = _gather_records([(value, owner)])
    local = _local(value)
    rank = dist.get_rank()
    chunks, seen = [], set()
    for source, info in enumerate(records):
        identity = repr(
            (info["owner"], info["offset"], info["local_shape"], info.get("axes"))
        )
        if identity not in seen:
            seen.add(identity)
            chunks.append((source, info))
    shape = records[0]["shape"]
    # records are identical on every rank, so every rank raises the same errors without a collective
    if any(info["shape"] != shape for _, info in chunks):
        raise ValueError(
            "Full checkpoint failed: Checkpoint shards disagree on the global tensor shape"
        )
    if any(info["quantized"] for _, info in chunks):
        return _gather_quantized(local, chunks, shape)
    plan = _boxes(chunks)
    began = time.perf_counter()
    full = torch.empty(shape, dtype=_dtype(records[0]["dtype"])) if rank == 0 else None
    _timed("alloc", began)
    if plan is None:
        return _gather_indexed(local, chunks, full)
    view, boxes = plan
    view = full.view(view) if rank == 0 else None
    destination = None
    for (source, info), (starts, lengths) in zip(chunks, boxes, strict=True):
        if not math.prod(lengths):
            continue
        destination = None
        if rank == 0:
            destination = view
            for dim, (start, length) in enumerate(zip(starts, lengths, strict=True)):
                destination = destination.narrow(dim, start, length)
        _transfer(
            local if rank == source else None,
            info["local_shape"],
            _dtype(info["dtype"]),
            source,
            out=destination,
        )
    if _TIMING is not None and rank == 0:
        _TIMING["tensors"].append(
            (
                time.perf_counter() - began,
                f"{records[0]['dtype']} {list(shape)} view {list(view.shape)} "
                f"stride {list(destination.stride()) if destination is not None else None} "
                f"from {len(chunks)} chunk(s)",
            )
        )
    return full


def _gather_indexed(local, chunks, full):
    rank = dist.get_rank()
    covered = torch.zeros(full.shape, dtype=torch.bool) if rank == 0 else None
    for source, info in chunks:
        tensor = _transfer(
            local if rank == source else None,
            info["local_shape"],
            _dtype(info["dtype"]),
            source,
        )
        if rank == 0:
            indices = _index(_coordinates(info))
            full[indices] = tensor
            covered[indices] = True
    error = None
    if rank == 0 and not covered.all():
        error = "Checkpoint does not cover every expert and shard"
    _check_errors(error)
    return full


def _gather_quantized(local, chunks, shape):
    rank = dist.get_rank()
    gathered = []
    for source, info in chunks:
        chunk = dict(info)
        if info["quantized"]:
            for attr, size, dtype in (
                ("codes", info["local_shape"], torch.uint8),
                (
                    "scale",
                    [math.prod(info["local_shape"]) // info["block_size"]],
                    torch.float32,
                ),
                ("qmap", [256], torch.float32),
            ):
                chunk[attr] = _transfer(
                    getattr(local, attr) if rank == source else None,
                    size,
                    dtype,
                    source,
                )
        else:
            chunk["tensor"] = _transfer(
                local if rank == source else None,
                info["local_shape"],
                _dtype(info["dtype"]),
                source,
            )
        gathered.append(chunk)
    if rank != 0:
        return None
    return {_QUANTIZED: 1, "shape": shape, "chunks": gathered}


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


def _adapter_values(model):
    values = {}
    for prefix, module in model.named_modules(remove_duplicate=False):
        prefix = prefix + "." if prefix else ""
        for name, value in module._parameters.items():
            if value is not None:
                values[prefix + name] = value
        for name, value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                values[prefix + name] = value
    return fsdp2_canonicalize_names(values)


def _base_config_dir(model):
    """Local directory holding the base model's config.json, without a Hub request."""
    configs = getattr(model, "peft_config", {}).values()
    model_id = next((c.base_model_name_or_path for c in configs), None)
    if not model_id:
        return None
    if os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
    except Exception:  # pylint: disable=broad-except
        return None
    return os.path.dirname(cached) if isinstance(cached, str) else None


def peft_save_embedding_layers(model):
    """Explicit PEFT ``save_embedding_layers``: ``"auto"`` asks the Hub for the base model's
    config.json on every save, on every rank, to detect a resized vocabulary."""
    from peft.tuners.trainable_tokens.layer import TrainableTokensLayer
    from peft.tuners.tuners_utils import BaseTunerLayer
    from peft.utils.other import AuxiliaryTrainingWrapper
    from peft.utils.peft_types import PeftType

    # as in PEFT's "auto": trainable tokens never resize the vocabulary
    using_trainable_tokens = any(
        config.peft_type == PeftType.TRAINABLE_TOKENS
        or getattr(config, "trainable_token_indices", None) is not None
        for config in getattr(model, "peft_config", {}).values()
    )
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            layer = getattr(model, getter)()
        except (AttributeError, NotImplementedError):
            continue
        # modules_to_save and trainable-token wrappers export their own trained copy
        if layer is None or isinstance(layer, AuxiliaryTrainingWrapper):
            continue
        if isinstance(layer, BaseTunerLayer):
            if not (isinstance(layer, TrainableTokensLayer) or using_trainable_tokens):
                return True
        elif any(p.requires_grad for p in layer.parameters()):
            return True
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    directory = _base_config_dir(model) if vocab_size else None
    if directory is None:
        return False
    try:
        base = type(model.config).from_pretrained(directory)
    except (OSError, ValueError):
        return False
    return vocab_size != getattr(base, "vocab_size", vocab_size)


def _adapter_export_names(model, values, adapter):
    """PEFT export key -> canonical parameter/buffer names it holds."""
    from peft.utils.save_and_load import get_peft_model_state_dict

    # PEFT rewrites export keys; string values retain raw names without gathering DTensors.
    exported = get_peft_model_state_dict(
        model,
        state_dict={name: name for name in values},
        adapter_name=adapter,
        unwrap_compiled=True,
        save_embedding_layers=peft_save_embedding_layers(model),
    )
    mapping = {}
    for key, value in exported.items():
        if isinstance(value, str):
            mapping[key] = (value,)
            continue
        names = tuple(name for name, parameter in values.items() if parameter is value)
        if not names:
            raise ValueError("PEFT export state is not a model parameter or buffer")
        mapping[key] = names
    return mapping


def _adapter_model_state(model, exports=None):
    """Canonical name -> parameter/buffer of every tensor the PEFT export holds; ``exports``
    (when given) receives each adapter's export-key mapping."""
    values = _adapter_values(model)
    selected, error = set(), None
    try:
        for adapter in model.peft_config:
            mapping = _adapter_export_names(model, values, adapter)
            if exports is not None:
                exports[adapter] = mapping
            for names in mapping.values():
                selected.update(names)
    except Exception as exc:  # pylint: disable=broad-except
        error = f"Adapter checkpoint selection failed: {exc}"
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (error, sorted(selected)))
    errors = [error for error, _ in gathered if error]
    if errors:
        raise ValueError("Full checkpoint failed: " + "; ".join(errors))
    if any(names != sorted(selected) for _, names in gathered):
        raise ValueError("Adapter checkpoint keys differ across ranks")
    _warn_unsaved_trainable(model, {id(values[name]) for name in selected})
    return {name: values[name] for name in sorted(selected)}


def _warn_unsaved_trainable(model, saved):
    unsaved = sorted(
        name
        for name, parameter in _parameters(model).items()
        if parameter.requires_grad and id(parameter) not in saved
    )
    if unsaved:
        LOG.warning_once(
            f"{len(unsaved)} trainable parameters are not part of the PEFT adapter and are "
            f"not saved in adapter checkpoints or exports: {unsaved[:8]}. Add their modules "
            "to lora_modules_to_save (or freeze them) to keep their updates."
        )


_ADAPTER_REFERENCE = "_axolotl_adapter_reference"
_ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def adapter_reference(model):
    """Model-only ``pytorch_model_fsdp.bin`` payload pointing at the PEFT adapter file saved
    beside it, or ``None`` when that file cannot stand in for the full state."""
    if not is_peft_model(model) or len(getattr(model, "peft_config", {})) != 1:
        return None
    return {_ADAPTER_REFERENCE: 1}


def _load_referenced_adapter(model, directory):
    """Rank 0: the referenced adapter file keyed by canonical names (other ranks: ``{}``)."""
    state, error = {}, None
    if dist.get_rank() == 0:
        try:
            if not is_peft_model(model) or len(model.peft_config) != 1:
                raise ValueError(
                    "An adapter-reference checkpoint restores a single-adapter PEFT model"
                )
            from .fsdp2_sharded_adapter import complete_interrupted_consolidation

            complete_interrupted_consolidation(directory)
            (adapter,) = model.peft_config
            # PEFT saves a non-default adapter in a subdirectory named after it
            folders = [directory] + (
                [directory / adapter] if adapter != "default" else []
            )
            path = next(
                (
                    folder / name
                    for folder in folders
                    for name in _ADAPTER_FILES
                    if (folder / name).is_file()
                ),
                None,
            )
            if path is None:
                raise FileNotFoundError(
                    f"{directory} references a PEFT adapter but has none of {_ADAPTER_FILES}"
                )
            if path.suffix == ".safetensors":
                from safetensors.torch import load_file

                saved = load_file(str(path), device="cpu")
            else:
                saved = torch.load(path, map_location="cpu", weights_only=True)
            mapping = _adapter_export_names(model, _adapter_values(model), adapter)
            unknown = sorted(set(saved) - set(mapping))
            if unknown:
                raise ValueError(f"Adapter file has unexpected keys: {unknown[:8]}")
            state = {
                name: value for key, value in saved.items() for name in mapping[key]
            }
        except Exception as exc:  # pylint: disable=broad-except
            error = f"Referenced adapter load failed: {exc}"
    _check_errors(error)
    return state


def full_model_state(model, adapter_only=False):
    # TODO: stream checkpoint assembly layerwise to reduce rank-0 host memory.
    global _TIMING
    _TIMING = (
        {"seconds": {}, "bytes": {}, "tensors": []}
        if os.environ.get("AXOLOTL_CHECKPOINT_TIMING") == "1"
        else None
    )
    began = time.perf_counter()
    adapter_only = adapter_only and is_peft_model(model)
    if adapter_only:
        values = _adapter_model_state(model)
    else:
        from .fsdp2_bnb_checkpoint import packed_parameters

        error = None
        try:
            if any(
                item["owner"] is not None for item in packed_parameters(model).values()
            ):
                error = (
                    "Generic full state dict cannot represent packed BNB expert ownership; "
                    "use save_fsdp_model for native full checkpoints or adapter-only export"
                )
        except ValueError as exc:
            error = str(exc)
        _check_errors(error)
        values = fsdp2_canonicalize_names(dict(model.state_dict()))
    owners = expert_ownership(model)
    tensors = [
        name
        for name, value in values.items()
        if torch.is_tensor(value) and value.ndim > 0
    ]
    records = dict(
        zip(
            tensors,
            _gather_records([(values[name], owners.get(name)) for name in tensors]),
            strict=True,
        )
    )
    state = {
        name: _gather_tensor(value, owners.get(name), records[name])
        if name in records
        else _gather_value(value, owners.get(name))
        for name, value in values.items()
    }
    _log_timing(time.perf_counter() - began)
    _TIMING = None
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
    targets = (
        _adapter_model_state(model)
        if adapter_only
        else fsdp2_canonicalize_names(dict(model.state_dict()))
    )
    owners = expert_ownership(model)
    metadata = [_metadata(state) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(metadata, src=0)
    missing = set(targets) - set(metadata[0])
    _check_errors(f"Missing model parameters: {sorted(missing)}" if missing else None)
    error = None
    try:
        for name, parameter in targets.items():
            expected = _layout(parameter, owners.get(name))["shape"]
            saved = metadata[0][name]
            saved_shape = (
                list(saved.shape)
                if torch.is_tensor(saved)
                else saved.get("shape")
                if isinstance(saved, dict)
                else None
            )
            if saved_shape != expected:
                guidance = ""
                if name in owners:
                    guidance = (
                        " A legacy EP checkpoint may contain only EP group 0's experts. "
                        "Start a new run from a complete final model export, or load a "
                        "complete adapter_model.safetensors with lora_model_dir "
                        "without restoring optimizer or scheduler state."
                    )
                raise ValueError(
                    f"Checkpoint parameter {name}: saved shape {saved_shape} differs "
                    f"from global target shape {expected}.{guidance}"
                )
    except ValueError as exc:
        error = str(exc)
    _check_errors(error)
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
                    reference = None
                    if not optimizer_operation and adapter_only:
                        from .fsdp2_sharded_adapter import is_sharded_export

                        # the adapter file being consolidated beside it holds this state
                        if is_sharded_export(path):
                            reference = adapter_reference(model)
                    if reference is not None:
                        state = reference
                    elif packed:
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
                    (
                        "_axolotl_full_model_version" in state,
                        _ADAPTER_REFERENCE in state,
                    )
                    if accelerator.is_main_process and not optimizer_operation
                    else (False, False)
                ]
                dist.broadcast_object_list(envelope, src=0)
                packed_envelope, referenced = envelope[0]
                if referenced:
                    _check_errors(
                        "Unsupported adapter-reference checkpoint version"
                        if accelerator.is_main_process
                        and state.get(_ADAPTER_REFERENCE) != 1
                        else None
                    )
                    result = restore_model_state(
                        model, _load_referenced_adapter(model, path), adapter_only=True
                    )
                elif packed_envelope and adapter_only:
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
            # PEFT saves select the adapter from this; never gather the frozen base to rank 0
            return full_model_state(model, adapter_only=is_peft_model(model))
        return original_get(self, model, unwrap=unwrap)

    accelerate.Accelerator.get_state_dict = get_state_dict
