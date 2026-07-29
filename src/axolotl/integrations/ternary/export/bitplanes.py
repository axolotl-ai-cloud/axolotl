"""Packed containers for the two-plane grids: `fv5` (dual) and `tp9` (trit_planes).

Both modes reach quality the single-plane grid cannot, and both were master-only until
now — 16 bits per weight to ship a model whose information content is 3 to 4. These are
size-truthful containers for them, in the layout a five/nine-level kernel would want.

**`fv5`** carries the five-value `dual` grid `{0, ±s_lo, ±s_hi}` as three bit planes over
the flat row-major weight, in the Fermion FV5 shape:

| plane | meaning |
|---|---|
| `bp` | the weight is positive |
| `bn` | the weight is negative |
| `br` | the weight uses `s_hi` rather than `s_lo` |

`w = (bp - bn) · (br ? s_hi : s_lo)`, so 3 bits per weight plus two f32 per row. Two
invariants make the planes a grid rather than three independent masks, and the unpacker
enforces both: `bp & bn == 0` (a weight has one sign) and `br ⊆ (bp | bn)` (a zero
weight selects no scale). A container that violates either is rejected rather than
decoded into something plausible.

**`tp9`** carries the nine-value free sum `w = t1·s1 + t2·s2` as its two trit planes,
each packed two bits per weight through the same machinery the monitor's snapshots use,
plus two f32 per row. 4 bits per weight.

Neither is a kernel format today — nothing consumes them. They exist so a healed
two-plane model can be shipped and archived at its real size, and so the layout a future
llama.cpp fork would need already has a producer and a byte-exact spec.

The scales are *not* re-derived from the values here: packing requires the manifest's
persisted scale sidecar. A free-sum row that uses only combination states does not
contain its own scales at all, and inferring a plausible pair for a shipped artifact is
how a silently different model gets published.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

from .. import quant
from ..swap import SwapManifest
from . import bake

LOG = get_logger(__name__)

FV5_FORMAT: str = "fv5"
TP9_FORMAT: str = "tp9"

FV5_SUFFIX: str = "weight_fv5"
TP9_SUFFIX: str = "weight_tp9"
SCALES_SUFFIX: str = "scales"

FV5_RECORD_FILENAME: str = "ternary_fv5.json"
TP9_RECORD_FILENAME: str = "ternary_tp9.json"

# scale mode each container is the packed form of
FORMAT_SCALE_MODES: dict[str, str] = {FV5_FORMAT: "dual", TP9_FORMAT: "trit_planes"}

FV5_PLANES: int = 3
FV5_BITS_PER_WEIGHT: float = 3.0
TP9_BITS_PER_WEIGHT: float = 4.0

_BITS_PER_BYTE: int = 8
_SCALE_COLUMNS: int = 2
_SCALE_BYTES: int = 4


# ------------------------------------------------------------------- fv5 planes


def pack_fv5(codes: torch.Tensor) -> torch.Tensor:
    """Pack five-state `dual` codes into the `bp | bn | br` bit planes.

    Args:
        codes: int8 codes in `{-2, -1, 0, +1, +2}` of shape `(rows, cols)`, where
            `±1` selects `s_lo` and `±2` selects `s_hi`.

    Returns:
        Flat uint8 tensor of `3 · ceil(numel / 8)` bytes: `bp`, then `bn`, then `br`,
        each LSB-first over the row-major weight.

    Raises:
        ValueError: If `codes` is not 2-D or holds a state outside `{-2..2}`.
    """
    if codes.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(codes.shape)}")
    flat = codes.reshape(-1).to(torch.int16)
    if bool(((flat < -2) | (flat > 2)).any()):
        raise ValueError("fv5 expects five-state codes in {-2, -1, 0, +1, +2}")
    return torch.cat(
        [
            _pack_bits(flat > 0),
            _pack_bits(flat < 0),
            _pack_bits(flat.abs() == 2),
        ]
    ).contiguous()


def unpack_fv5(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_fv5`, enforcing the layout's two invariants.

    Raises:
        ValueError: If the byte count disagrees with `shape`, a weight is marked both
            positive and negative, or a zero weight selects a scale.
    """
    numel = int(shape[0]) * int(shape[1])
    plane_bytes = _plane_bytes(numel)
    expected = FV5_PLANES * plane_bytes
    if packed.numel() != expected:
        raise ValueError(
            f"fv5 tensor has {packed.numel()} bytes, expected {expected} for {shape}"
        )

    planes = [
        _unpack_bits(packed[index * plane_bytes : (index + 1) * plane_bytes], numel)
        for index in range(FV5_PLANES)
    ]
    positive, negative, high = planes
    if bool((positive & negative).any()):
        raise ValueError(
            "fv5 invariant violated: a weight is marked both positive and negative"
        )
    if bool((high & ~(positive | negative)).any()):
        raise ValueError("fv5 invariant violated: a zero weight selects the s_hi scale")

    level = high.to(torch.int8) + 1
    codes = (positive.to(torch.int8) - negative.to(torch.int8)) * level
    return codes.reshape(shape)


def decode_fv5(
    packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(five-state codes, (rows, 2) scales)` read back out of fv5 bytes."""
    return unpack_fv5(packed, shape), _check_scales(scales, shape, FV5_FORMAT)


# ------------------------------------------------------------------- tp9 planes


def pack_tp9(planes: torch.Tensor) -> torch.Tensor:
    """Pack the two free-sum trit planes two bits per weight, plane 0 then plane 1.

    Args:
        planes: int8 planes of shape `(2, rows, cols)`, every entry in `{-1, 0, 1}`.

    Returns:
        Flat uint8 tensor of `2 · ceil(numel / 4)` bytes.

    Raises:
        ValueError: If `planes` is not `(2, rows, cols)` or holds a non-trit.
    """
    if planes.ndim != 3 or planes.shape[0] != quant.DUAL_PLANES:
        raise ValueError(
            f"expected ({quant.DUAL_PLANES}, rows, cols) trit planes, got shape "
            f"{tuple(planes.shape)}"
        )
    if bool(((planes < -1) | (planes > 1)).any()):
        raise ValueError("tp9 expects trit planes in {-1, 0, +1}")
    return torch.cat(
        [quant.pack_codes(planes[index]) for index in range(quant.DUAL_PLANES)]
    ).contiguous()


def unpack_tp9(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_tp9`, returning int8 planes of `(2, *shape)`.

    Raises:
        ValueError: If the byte count disagrees with `shape`.
    """
    numel = int(shape[0]) * int(shape[1])
    plane_bytes = -(-numel // quant.CODES_PER_BYTE)
    expected = quant.DUAL_PLANES * plane_bytes
    if packed.numel() != expected:
        raise ValueError(
            f"tp9 tensor has {packed.numel()} bytes, expected {expected} for {shape}"
        )
    return torch.stack(
        [
            quant.unpack_codes(
                packed[index * plane_bytes : (index + 1) * plane_bytes], shape
            )
            for index in range(quant.DUAL_PLANES)
        ]
    )


def decode_tp9(
    packed: torch.Tensor, scales: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(trit planes, (rows, 2) scales)` read back out of tp9 bytes."""
    return unpack_tp9(packed, shape), _check_scales(scales, shape, TP9_FORMAT)


# ------------------------------------------------------------------- accounting


def payload_bytes(fmt: str, shape: tuple[int, int]) -> int:
    """Return the plane bytes a tensor of `shape` costs in `fmt`."""
    numel = int(shape[0]) * int(shape[1])
    if fmt == FV5_FORMAT:
        return FV5_PLANES * _plane_bytes(numel)
    return quant.DUAL_PLANES * -(-numel // quant.CODES_PER_BYTE)


def bits_per_weight(
    fmt: str, shape: tuple[int, int], with_scales: bool = True
) -> float:
    """Return the measured cost of a tensor in bits per weight.

    Args:
        fmt: `fv5` or `tp9`.
        shape: `(rows, cols)` of the weight.
        with_scales: Include the two f32 per row. The planes alone are 3.0 (fv5) or
            4.0 (tp9) bits; the scales add `64 / cols`, which is where a narrow tensor
            pays for the per-row grid.
    """
    numel = int(shape[0]) * int(shape[1])
    if not numel:
        return 0.0
    total = payload_bytes(fmt, shape)
    if with_scales:
        total += int(shape[0]) * _SCALE_COLUMNS * _SCALE_BYTES
    return total * _BITS_PER_BYTE / numel


# ---------------------------------------------------------------------- export


def export_bitplanes(
    master_dir: str | Path,
    output_dir: str | Path,
    manifest: SwapManifest,
    fmt: str = FV5_FORMAT,
) -> Path:
    """Write an `fv5` or `tp9` checkpoint beside the master and return its directory.

    Every swapped tensor becomes a byte string plus its `(rows, 2)` f32 scales;
    everything else — embeddings, head, norms, any `subln` gains — is copied through
    unchanged. Each byte string is decoded again and gated on the master before it is
    written, and its sha256 lands in the record beside the shards.

    Raises:
        ValueError: If the manifest's scale mode is not the one `fmt` packs, the
            master still holds latent weights, or it ships no persisted scales.
        KeyError: If a manifest entry has no matching weight in the master.
        RuntimeError: If a packed tensor does not round-trip back to the master.
    """
    master_dir = Path(master_dir)
    output = Path(output_dir)
    _reject_unsupported(manifest, fmt)
    if output.resolve() == master_dir.resolve():
        raise ValueError(
            f"{fmt} must be written beside the master, not over it; the packed weights "
            "are not loadable as an ordinary checkpoint"
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

    suffix = FV5_SUFFIX if fmt == FV5_FORMAT else TP9_SUFFIX
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
            scales = _require_scales(manifest, entry.name, fmt)
            try:
                codes, derived = bake.derive_codes_and_scale(
                    tensor, entry.group_size, entry.weight_scale, scales
                )
            except ValueError as exc:
                raise ValueError(
                    f"{entry.name} still holds latent weights ({exc}); bake the master "
                    "before packing it"
                ) from exc
            payload = pack_fv5(codes) if fmt == FV5_FORMAT else pack_tp9(codes)
            packed[f"{entry.name}.{suffix}"] = payload
            packed[f"{entry.name}.{SCALES_SUFFIX}"] = derived.to(torch.float32)
            records[entry.name] = _gate_and_record(
                fmt, payload, derived, tensor, entry.weight_scale
            )
            remaining.discard(key)
        shard_name = Path(path).name
        for key, value in packed.items():
            weight_map[key] = shard_name
            total_size += value.numel() * value.element_size()
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
    write_record(output, fmt, records)
    manifest.save(output)
    bake.write_quantizer_metadata(output, manifest, artifact_format=fmt)
    LOG.info(
        f"ternary: wrote {len(records)} {fmt} tensors to {output} at "
        f"{summarize(records)['bits_per_weight']:.3f} bits per weight "
        f"({summarize(records)['payload_bits_per_weight']:.3f} before scales)"
    )
    return output


def write_record(output_dir: str | Path, fmt: str, records: dict[str, dict]) -> Path:
    """Write the per-tensor gate digests and bit accounting, returning the record path."""
    name = FV5_RECORD_FILENAME if fmt == FV5_FORMAT else TP9_RECORD_FILENAME
    path = Path(output_dir) / name
    path.write_text(
        json.dumps(
            {
                "format": fmt,
                "scale_mode": FORMAT_SCALE_MODES[fmt],
                "consumed_by": "no engine yet — archival and future-kernel layout",
                **summarize(records),
                "tensors": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def summarize(records: dict[str, dict]) -> dict:
    """Aggregate the per-tensor accounting, scales included and excluded."""
    weights = sum(record["weights"] for record in records.values())
    payload = sum(record["payload_bytes"] for record in records.values())
    scale_bytes = sum(record["scale_bytes"] for record in records.values())
    return {
        "tensor_count": len(records),
        "weights": weights,
        "payload_bytes": payload,
        "scale_bytes": scale_bytes,
        "payload_bits_per_weight": (
            payload * _BITS_PER_BYTE / weights if weights else 0.0
        ),
        "bits_per_weight": (
            (payload + scale_bytes) * _BITS_PER_BYTE / weights if weights else 0.0
        ),
    }


# --------------------------------------------------------------------- internals


def _require_scales(
    manifest: SwapManifest, name: str, fmt: str
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = manifest.scales_for(name)
    if scales is None:
        raise ValueError(
            f"{fmt} packs the grid the master was baked on, and {name} ships none. "
            "Its scales live in the manifest's sidecar, written at bake time; a "
            "free-sum row that uses only combination states does not contain them, so "
            "there is nothing to recover from the values. Re-export from a master "
            "saved by a run that persisted them"
        )
    return scales


def _gate_and_record(
    fmt: str,
    payload: torch.Tensor,
    scales: torch.Tensor,
    master_weight: torch.Tensor,
    weight_scale: str,
) -> dict:
    """Round-trip the byte string on the master and return its accounting record."""
    from . import parity

    shape = (int(master_weight.shape[0]), int(master_weight.shape[1]))
    failures = parity.gate_bitplane_bytes(
        fmt, payload, scales, master_weight, weight_scale
    )
    if failures:
        raise RuntimeError(
            f"ternary: the {fmt} gate rejected a {shape} tensor: " + "; ".join(failures)
        )
    return {
        "rows": shape[0],
        "cols": shape[1],
        "weights": shape[0] * shape[1],
        "payload_bytes": int(payload.numel()),
        "scale_bytes": int(scales.numel()) * _SCALE_BYTES,
        "payload_bits_per_weight": bits_per_weight(fmt, shape, with_scales=False),
        "bits_per_weight": bits_per_weight(fmt, shape),
        "sha256": hashlib.sha256(_payload_bytes(payload)).hexdigest(),
        "scales_sha256": hashlib.sha256(_payload_bytes(scales)).hexdigest(),
    }


def _reject_unsupported(manifest: SwapManifest, fmt: str) -> None:
    if fmt not in FORMAT_SCALE_MODES:
        raise ValueError(f"unknown bit-plane format {fmt!r}")
    wanted = FORMAT_SCALE_MODES[fmt]
    if manifest.weight_scale != wanted:
        raise ValueError(
            f"{fmt} is the packed form of ternary.weight_scale: {wanted}, but this "
            f"master was trained with {manifest.weight_scale}. Its grid has a "
            "different shape, so the planes would mean something else"
        )


def _check_scales(
    scales: torch.Tensor, shape: tuple[int, int], fmt: str
) -> torch.Tensor:
    if tuple(scales.shape) != (int(shape[0]), _SCALE_COLUMNS):
        raise ValueError(
            f"{fmt} needs {(int(shape[0]), _SCALE_COLUMNS)} scales for a {shape} "
            f"tensor, got {tuple(scales.shape)}"
        )
    return scales.to(torch.float32)


def _plane_bytes(numel: int) -> int:
    return -(-numel // _BITS_PER_BYTE)


def _payload_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes()


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
    "FORMAT_SCALE_MODES",
    "FV5_BITS_PER_WEIGHT",
    "FV5_FORMAT",
    "FV5_PLANES",
    "FV5_RECORD_FILENAME",
    "FV5_SUFFIX",
    "SCALES_SUFFIX",
    "TP9_BITS_PER_WEIGHT",
    "TP9_FORMAT",
    "TP9_RECORD_FILENAME",
    "TP9_SUFFIX",
    "bits_per_weight",
    "decode_fv5",
    "decode_tp9",
    "export_bitplanes",
    "pack_fv5",
    "pack_tp9",
    "payload_bytes",
    "summarize",
    "unpack_fv5",
    "unpack_tp9",
    "write_record",
]
