"""Portable, byte-exact XOR checkpoints compatible with Slime and CoreWeave."""

import json
import re
import shutil
import tempfile
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zstandard
from safetensors.numpy import save_file as save_bytes
from safetensors.torch import save_file


def validate_identity(identity: str) -> str:
    """Require one unambiguous storage path segment."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", identity):
        raise ValueError("Checkpoint identity must be a single path segment")
    return identity


def snapshot_weights(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Capture independent CPU tensors, including tied weights and scalar buffers."""
    if not weights:
        raise ValueError("Cannot publish an empty model")
    return {
        name: tensor.detach().to(device="cpu", copy=True).contiguous()
        for name, tensor in weights.items()
    }


def _bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy()


def write_checkpoint(
    weights: Mapping[str, torch.Tensor],
    destination: Path | str,
    *,
    identity: str,
    config: dict | None = None,
    base: Mapping[str, torch.Tensor] | None = None,
    base_identity: str | None = None,
    max_shard_size: int = 1 << 30,
) -> Path:
    """Atomically publish a full checkpoint or a delta against an exact CPU snapshot.

    Inputs must remain unchanged for the duration of this call. Full checkpoints
    always include an index, even with one shard. Deltas contain only changed
    tensors and inherit their shape, dtype, and model config from the base.
    """
    validate_identity(identity)
    if max_shard_size <= 0 or not weights:
        raise ValueError("Weights and a positive max_shard_size are required")
    if (base is None) != (base_identity is None):
        raise ValueError("base and base_identity must be supplied together")
    if base is not None:
        assert base_identity is not None
        validate_identity(base_identity)
        if identity == base_identity or weights.keys() != base.keys():
            raise ValueError("A delta needs a new identity and the same tensor names")
    elif config is None:
        raise ValueError("A full checkpoint requires the model config")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=destination.parent))
    try:
        weight_map: dict[str, str] = {}
        shard: dict[str, Any] = {}
        checksums: dict[str, str] = {}
        size, total_size, shard_number = 0, 0, 0

        def flush():
            nonlocal shard_number, size
            if not shard:
                return
            filename = f"model-{shard_number:05d}.safetensors"
            if base is None:
                save_file(shard, staging / filename, metadata={"format": "pt"})
            else:
                save_bytes(shard, staging / filename, metadata=checksums)
            weight_map.update(dict.fromkeys(shard, filename))
            shard.clear()
            checksums.clear()
            shard_number += 1
            size = 0

        for name, tensor in weights.items():
            tensor = tensor.detach().cpu().contiguous()
            total_size += tensor.numel() * tensor.element_size()
            digest = None
            if base is not None:
                previous = base[name]
                if tensor.shape != previous.shape or tensor.dtype != previous.dtype:
                    raise ValueError(
                        f"Tensor layout changed for {name}; publish a full checkpoint"
                    )
                current = _bytes(tensor)
                delta = np.bitwise_xor(current, _bytes(previous.cpu().contiguous()))
                if not np.any(delta):
                    continue
                digest = f"{zlib.adler32(current):08x}"
                tensor = np.frombuffer(
                    zstandard.ZstdCompressor(level=1).compress(delta), dtype=np.uint8
                )
                nbytes = tensor.nbytes
            else:
                tensor = tensor.clone()
                nbytes = tensor.numel() * tensor.element_size()
            if shard and size + nbytes > max_shard_size:
                flush()
            shard[name] = tensor
            if digest is not None:
                checksums[name] = digest
            size += nbytes
        flush()
        metadata: dict[str, str | int | None] = {"total_size": total_size}
        if base is not None:
            metadata = {
                "version": identity,
                "base_version": base_identity,
                "delta_encoding": "xor",
                "compression_format": "zstd",
                "checksum_format": "adler32",
            }
        else:
            (staging / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (staging / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": metadata, "weight_map": weight_map}),
            encoding="utf-8",
        )
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
