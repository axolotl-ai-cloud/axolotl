"""Mask+sign packing: a sparsity-adaptive ternary artifact of our own.

Every ternary tensor is stored as two bit planes and one scale:

- a zero bitmap, one bit per weight, set where the code is nonzero,
- a sign plane, one bit per *nonzero* weight, set where that code is `+1`,
- the per-tensor `s16` scale, plus a shape header.

That costs `2 - zero_fraction` bits per weight against the flat 2.0 bpw of the
packed 2-bit formats, so it pays exactly as much as the model is sparse — around
1.4 bpw at the ~60% zero fraction a healthy heal settles at, and never worse than
2.0. The price is that decoding is a prefix-sum over the mask rather than a shift,
which is why this is an archival/transport format and not a kernel format.

Single plane and symmetric only: one scale for the whole tensor, no offset. Group,
per-row and dual scale modes are rejected by the config schema before they get here.

Byte layout (little-endian), one flat `uint8` string per tensor:

| offset | field |
|---|---|
| 0 | `MASK_SIGN_MAGIC` (4 bytes) |
| 4 | `uint32` format version |
| 8 | `uint32` rows, `uint32` cols |
| 16 | `uint64` nonzero count |
| 24 | `float16` scale, then zero padding to `HEADER_BYTES` |
| 32 | zero bitmap, `ceil(rows·cols / 8)` bytes, row-major, LSB first |
| … | sign plane, `ceil(nnz / 8)` bytes, nonzeros in row-major order, LSB first |
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

from ..args import NON_PER_TENSOR_SCALE_MODES
from ..swap import SwapManifest
from . import bake

LOG = get_logger(__name__)

MASK_SIGN_MAGIC: bytes = b"MSGN"
MASK_SIGN_VERSION: int = 1
HEADER_BYTES: int = 32

# suffix of the packed tensor beside each weight in the exported checkpoint
PACKED_SUFFIX: str = "weight_mask_sign"

FORMAT_NAME: str = "mask_sign"
RECORD_FILENAME: str = "ternary_mask_sign.json"

_HEADER_STRUCT: str = "<4sIIIQ"
_SCALE_OFFSET: int = 24
_BITS_PER_BYTE: int = 8


def pack_mask_sign(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes and their per-tensor scale into the mask+sign layout.

    Args:
        codes: int8 codes in `{-1, 0, +1}` of shape `(out_features, in_features)`.
        scale: fp32 per-tensor `s16` scale (0-dim); stored as `float16`.

    Returns:
        Flat uint8 tensor: header, zero bitmap, then the sign plane.

    Raises:
        ValueError: If `codes` is not 2-D, holds a value outside `{-1, 0, +1}`, or
            `scale` is not a single value.
    """
    if codes.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(codes.shape)}")
    if scale.numel() != 1:
        raise ValueError(
            f"mask_sign stores one scale per tensor, got {scale.numel()} values"
        )
    flat = codes.reshape(-1).to(torch.int16)
    if bool(((flat < -1) | (flat > 1)).any()):
        raise ValueError("mask_sign expects ternary codes in {-1, 0, +1}")

    nonzero = flat != 0
    rows, cols = int(codes.shape[0]), int(codes.shape[1])
    header = _header(rows, cols, int(nonzero.sum()), scale).to(codes.device)
    return torch.cat(
        [header, _pack_bits(nonzero), _pack_bits(flat[nonzero] > 0)]
    ).contiguous()


def unpack_mask_sign(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_mask_sign`, returning int8 codes of `shape`.

    Raises:
        ValueError: If the header is not a `MASK_SIGN_MAGIC` header of a supported
            version, or its shape, length or nonzero count disagrees with `shape`.
    """
    _, nnz, mask_bytes, sign_bytes = _read_header(packed, shape)
    numel = int(shape[0]) * int(shape[1])

    mask = _unpack_bits(packed[HEADER_BYTES : HEADER_BYTES + mask_bytes], numel)
    if int(mask.sum()) != nnz:
        raise ValueError(
            f"the zero bitmap marks {int(mask.sum())} nonzeros, the header claims {nnz}"
        )
    signs = _unpack_bits(
        packed[HEADER_BYTES + mask_bytes : HEADER_BYTES + mask_bytes + sign_bytes], nnz
    )

    codes = torch.zeros(numel, dtype=torch.int8, device=packed.device)
    codes[mask] = signs.to(torch.int8) * 2 - 1
    return codes.reshape(shape)


def decode_mask_sign(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(codes, per-tensor scale)` read back out of mask+sign bytes."""
    scale, _, _, _ = _read_header(packed, shape)
    return unpack_mask_sign(packed, shape), scale


def bits_per_weight(codes: torch.Tensor) -> float:
    """Return the payload cost of `codes` in bits per weight — `2 - zero_fraction`.

    The header and the scale are excluded: they are per tensor, not per weight.
    """
    numel = codes.numel()
    if not numel:
        return 0.0
    return 2.0 - int((codes == 0).sum()) / numel


def export_mask_sign(
    master_dir: str | Path, output_dir: str | Path, manifest: SwapManifest
) -> Path:
    """Write a mask+sign checkpoint beside the master and return its directory.

    Every swapped tensor is replaced by its `PACKED_SUFFIX` byte string; everything
    else (embeddings, head, norms, config, tokenizer) is copied through unchanged.
    Each byte string is decoded again and gated on the master before it is written,
    and its sha256 lands in `RECORD_FILENAME` beside the shards.

    Raises:
        ValueError: If the manifest uses a scale mode the format cannot represent
            (anything but per-tensor), or the master still holds latent weights.
        KeyError: If a manifest entry has no matching weight in the master.
        RuntimeError: If a packed tensor does not round-trip back to the master.
    """
    master_dir = Path(master_dir)
    output = Path(output_dir)
    _reject_unsupported(manifest)
    if output.resolve() == master_dir.resolve():
        raise ValueError(
            "mask_sign must be written beside the master, not over it; the packed "
            "weights are not loadable as an ordinary checkpoint"
        )

    entries = {f"{entry.name}.weight": entry for entry in manifest.entries}
    remaining = set(entries)
    shards = bake.shard_paths(master_dir)
    bake.copy_aux_files(
        master_dir,
        output,
        skip={path.name for path in shards}
        | {bake.SAFETENSORS_INDEX_NAME, bake.TORCH_INDEX_NAME},
    )

    weight_map: dict[str, str] = {}
    records: dict[str, dict] = {}
    total_size = 0
    for path in shards:
        tensors, metadata = bake.load_shard(path)
        packed: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            entry = entries.get(key)
            if entry is None:
                packed[key] = tensor
                continue
            try:
                codes, scale = bake.derive_codes_and_scale(tensor)
            except ValueError as exc:
                raise ValueError(
                    f"{entry.name} still holds latent weights ({exc}); bake the master "
                    "before packing it"
                ) from exc
            payload = pack_mask_sign(codes, scale)
            packed[f"{entry.name}.{PACKED_SUFFIX}"] = payload
            records[entry.name] = _gate_and_record(payload, tensor, codes)
            remaining.discard(key)
        shard_name = Path(path).name
        for key, tensor in packed.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        bake.save_shard(packed, output / shard_name, metadata)
    if remaining:
        raise KeyError(
            f"{len(remaining)} manifest weights are missing from the master in "
            f"{master_dir}: {sorted(remaining)[:5]}"
        )

    if (master_dir / bake.SAFETENSORS_INDEX_NAME).is_file():
        (output / bake.SAFETENSORS_INDEX_NAME).write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    write_record(output, records)
    manifest.save(output)
    bake.write_quantizer_metadata(output, manifest, artifact_format=FORMAT_NAME)
    LOG.info(
        f"ternary: wrote {len(records)} mask_sign tensors to {output} at "
        f"{_summary(records)['bits_per_weight']:.3f} bits per weight"
    )
    return output


def write_record(output_dir: str | Path, records: dict[str, dict]) -> Path:
    """Write the per-tensor gate digests and bit accounting, returning the record path."""
    path = Path(output_dir) / RECORD_FILENAME
    path.write_text(
        json.dumps(
            {
                "format": FORMAT_NAME,
                "version": MASK_SIGN_VERSION,
                **_summary(records),
                "tensors": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _summary(records: dict[str, dict]) -> dict:
    weights = sum(record["weights"] for record in records.values())
    packed_bytes = sum(record["bytes"] for record in records.values())
    payload_bits = sum(
        (record["bytes"] - HEADER_BYTES) * _BITS_PER_BYTE for record in records.values()
    )
    return {
        "tensor_count": len(records),
        "weights": weights,
        "packed_bytes": packed_bytes,
        # payload only; `stored_bits_per_weight` carries the per-tensor headers too
        "bits_per_weight": payload_bits / weights if weights else 0.0,
        "stored_bits_per_weight": (
            packed_bytes * _BITS_PER_BYTE / weights if weights else 0.0
        ),
    }


def _gate_and_record(
    payload: torch.Tensor, master_weight: torch.Tensor, codes: torch.Tensor
) -> dict:
    """Round-trip the byte string on the master and return its accounting record."""
    from . import parity

    failures = parity.gate_block_bytes(FORMAT_NAME, payload, master_weight)
    if failures:
        raise RuntimeError(
            f"ternary: the mask_sign gate rejected a {tuple(master_weight.shape)} "
            "tensor: " + "; ".join(failures)
        )
    return {
        "rows": int(codes.shape[0]),
        "cols": int(codes.shape[1]),
        "weights": int(codes.numel()),
        "nonzeros": int((codes != 0).sum()),
        "bytes": int(payload.numel()),
        "bits_per_weight": bits_per_weight(codes),
        "sha256": hashlib.sha256(_payload_bytes(payload)).hexdigest(),
    }


def _reject_unsupported(manifest: SwapManifest) -> None:
    if manifest.group_size is not None or (
        manifest.weight_scale in NON_PER_TENSOR_SCALE_MODES
    ):
        raise ValueError(
            "mask_sign stores one scalar scale per tensor and cannot represent "
            f"ternary.weight_scale: {manifest.weight_scale} "
            f"(group_size={manifest.group_size}); export master_bf16 instead"
        )


def _payload_bytes(payload: torch.Tensor) -> bytes:
    return bytes(payload.detach().to(torch.uint8).contiguous().cpu().numpy().tobytes())


def _header(rows: int, cols: int, nnz: int, scale: torch.Tensor) -> torch.Tensor:
    fields = struct.pack(
        _HEADER_STRUCT, MASK_SIGN_MAGIC, MASK_SIGN_VERSION, rows, cols, nnz
    )
    header = torch.zeros(HEADER_BYTES, dtype=torch.uint8)
    header[: len(fields)] = torch.frombuffer(bytearray(fields), dtype=torch.uint8)
    header[_SCALE_OFFSET : _SCALE_OFFSET + 2] = (
        scale.detach().to(torch.float32).reshape(1).to(torch.float16).view(torch.uint8)
    )
    return header


def _read_header(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, int, int, int]:
    """Validate a packed tensor against `shape` and return `(scale, nnz, mask/sign bytes)`."""
    if packed.ndim != 1 or packed.dtype != torch.uint8:
        raise ValueError(
            f"expected a flat uint8 mask_sign tensor, got {packed.dtype} of shape "
            f"{tuple(packed.shape)}"
        )
    if packed.numel() < HEADER_BYTES:
        raise ValueError(
            f"a mask_sign tensor is at least {HEADER_BYTES} bytes, got {packed.numel()}"
        )
    head = bytes(packed[:HEADER_BYTES].detach().cpu().tolist())
    magic, version, rows, cols, nnz = struct.unpack(
        _HEADER_STRUCT, head[:_SCALE_OFFSET]
    )
    if magic != MASK_SIGN_MAGIC:
        raise ValueError(f"not a mask_sign tensor: magic {magic!r}")
    if version != MASK_SIGN_VERSION:
        raise ValueError(
            f"mask_sign version {version} is not readable by version {MASK_SIGN_VERSION}"
        )
    if (rows, cols) != (int(shape[0]), int(shape[1])):
        raise ValueError(
            f"packed tensor holds a {rows}x{cols} weight, not {tuple(shape)}"
        )
    numel = rows * cols
    if nnz > numel:
        raise ValueError(f"header claims {nnz} nonzeros in a {numel}-weight tensor")

    mask_bytes = -(-numel // _BITS_PER_BYTE)
    sign_bytes = -(-nnz // _BITS_PER_BYTE)
    expected = HEADER_BYTES + mask_bytes + sign_bytes
    if packed.numel() != expected:
        raise ValueError(
            f"packed tensor has {packed.numel()} bytes, expected {expected} for a "
            f"{rows}x{cols} weight with {nnz} nonzeros"
        )
    scale = torch.tensor(
        struct.unpack("<e", head[_SCALE_OFFSET : _SCALE_OFFSET + 2])[0],
        dtype=torch.float32,
        device=packed.device,
    )
    return scale, nnz, mask_bytes, sign_bytes


def _pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """Pack a flat boolean tensor into LSB-first uint8 bytes, zero-padding the tail."""
    flat = bits.reshape(-1).to(torch.int16)
    pad = -flat.numel() % _BITS_PER_BYTE
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    lanes = flat.reshape(-1, _BITS_PER_BYTE)
    return (lanes << _bit_shifts(flat.device)).sum(dim=-1).to(torch.uint8)


def _unpack_bits(packed: torch.Tensor, count: int) -> torch.Tensor:
    bits = (packed.to(torch.int16).unsqueeze(-1) >> _bit_shifts(packed.device)) & 1
    return bits.reshape(-1)[:count].to(torch.bool)


def _bit_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(_BITS_PER_BYTE, dtype=torch.int16, device=device)


__all__ = [
    "HEADER_BYTES",
    "MASK_SIGN_MAGIC",
    "MASK_SIGN_VERSION",
    "PACKED_SUFFIX",
    "RECORD_FILENAME",
    "bits_per_weight",
    "decode_mask_sign",
    "export_mask_sign",
    "pack_mask_sign",
    "unpack_mask_sign",
    "write_record",
]
