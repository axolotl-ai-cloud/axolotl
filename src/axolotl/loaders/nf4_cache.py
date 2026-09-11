"""Versioned, tensor-only disk caches for CPU-staged NF4 base models."""

import copy
import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from axolotl.utils.nf4 import (
    BnbNF4Parametrization,
    TorchaoNF4Parametrization,
    checkpoint_nf4_linear,
    torchao_nf4_module,
)


def nf4_cache_path(cfg, model_config, model_kwargs, quantization, device):
    """Key caches by source identity, conversion settings, and implementation versions."""
    if not cfg.nf4_cache_dir:
        return None
    source = Path(cfg.base_model)
    if source.is_dir():
        files = sorted(
            path
            for path in source.rglob("*")
            if path.is_file()
            and path.suffix in {".safetensors", ".bin", ".json", ".py"}
        )
        identity = [
            (
                str(path.relative_to(source)),
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in files
        ]
        source_id = str(source.resolve())
    else:
        from transformers.utils.hub import cached_file, extract_commit_hash

        resolved = cached_file(
            cfg.base_model,
            "config.json",
            revision=model_kwargs.get("revision", "main"),
            subfolder=model_kwargs.get("subfolder", ""),
            token=model_kwargs.get("token"),
            cache_dir=model_kwargs.get("cache_dir"),
            local_files_only=model_kwargs.get("local_files_only", False),
        )
        identity = extract_commit_hash(resolved, None)
        if not identity:
            raise ValueError("NF4 caching requires a resolved Hub checkpoint commit")
        # Pin the weight read to the same source revision used by the cache key.
        model_kwargs["revision"] = identity
        model_kwargs["_commit_hash"] = identity
        source_id = cfg.base_model
    if model_kwargs.get("state_dict") is not None:
        raise ValueError("NF4 caching does not support an explicit state_dict")
    backend = cfg.nf4_backend or "bitsandbytes"
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "transformers",
            "bitsandbytes" if backend == "bitsandbytes" else "torchao",
        )
    }
    # These affect checkpoint conversion; placement and download options do not.
    options = {
        name: value
        for name, value in model_kwargs.items()
        if name
        not in {
            "device_map",
            "token",
            "cache_dir",
            "local_files_only",
            "force_download",
        }
    }
    settings = {
        "format": 1,
        "source": source_id,
        "identity": identity,
        "config": model_config.to_dict(),
        "options": options,
        "quantization": quantization.to_dict()
        if quantization
        else cfg.bnb_config_kwargs,
        "backend": backend,
        "dtype": cfg.torch_dtype,
        "experts": cfg.quantize_moe_experts,
        "model_type": cfg.model_config_type,
        "skips": cfg.lora_modules_to_save,
        "device": torch.device(device).type,
        "versions": versions,
    }
    digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, default=str).encode()
    ).hexdigest()
    return Path(cfg.nf4_cache_dir).expanduser() / f"{digest}.pt"


def _transform_metadata(transform):
    attributes = {"shape": transform.shape, "dtype": transform.dtype}
    if isinstance(transform, BnbNF4Parametrization):
        kind = "bitsandbytes"
        for name in (
            "blocksize",
            "quant_type",
            "nested",
            "nested_blocksize",
            "nested_dtype",
        ):
            if hasattr(transform, name):
                attributes[name] = getattr(transform, name)
    elif isinstance(transform, TorchaoNF4Parametrization):
        kind = "torchao"
        chunks = copy.deepcopy(transform.chunks)
        for _, metadata, _, _ in chunks:
            metadata["tensor_meta"] = asdict(metadata["tensor_meta"])
        attributes["chunks"] = chunks
    else:
        raise ValueError(f"Unsupported NF4 cache parametrization: {type(transform)}")
    return {
        "kind": kind,
        "attributes": attributes,
        "buffers": {
            name: None if value is None else (tuple(value.shape), value.dtype)
            for name, value in transform._buffers.items()
        },
    }


def save_nf4_cache(path, model):
    """Atomically publish packed tensors and primitive reconstruction metadata."""
    from axolotl.monkeypatch.moe_quant import _moe_load_state

    structures = []
    for module_path, module in model.named_modules():
        for name, chain in getattr(module, "parametrizations", {}).items():
            structures.append(
                (
                    module_path,
                    name,
                    tuple(chain.original.shape),
                    chain.original.dtype,
                    _transform_metadata(chain[0]),
                )
            )
    payload = {
        "structures": structures,
        "state_dict": model.state_dict(),
        "expert_param_order": _moe_load_state["expert_param_order"],
        "expert_count": _moe_load_state["count"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".nf4-", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_nf4_cache(path, factory):
    """Reconstruct a CPU base model without reading or quantizing source weights."""
    from axolotl.monkeypatch.moe_quant import _moe_load_state

    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    model = factory()
    for module_path, name, shape, dtype, metadata in payload["structures"]:
        module = model.get_submodule(module_path)
        cls = (
            BnbNF4Parametrization
            if metadata["kind"] == "bitsandbytes"
            else TorchaoNF4Parametrization
        )
        transform = cls.__new__(cls)
        nn.Module.__init__(transform)
        attributes = metadata["attributes"]
        if metadata["kind"] == "torchao":
            for _, context, _, _ in attributes["chunks"]:
                context["tensor_meta"] = torchao_nf4_module().SubclassTensorArgs(
                    **context["tensor_meta"]
                )
        for key, value in attributes.items():
            setattr(transform, key, value)
        for key, spec in metadata["buffers"].items():
            transform.register_buffer(
                key,
                None
                if spec is None
                else torch.empty(spec[0], dtype=spec[1], device="meta"),
            )
        setattr(
            module,
            name,
            nn.Parameter(
                torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False
            ),
        )
        if isinstance(module, nn.Linear) and name == "weight":
            checkpoint_nf4_linear(module)
        parametrize.register_parametrization(module, name, transform, unsafe=True)
    model.load_state_dict(payload["state_dict"], strict=True, assign=True)
    model.tie_weights()
    model.eval()
    _moe_load_state["expert_param_order"] = payload["expert_param_order"]
    _moe_load_state["count"] = payload["expert_count"]
    return model
