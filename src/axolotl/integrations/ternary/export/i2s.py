"""bitnet.cpp I2_S writer (2.0 bpw GGUF variant)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..args import EmbeddingDtype, RouterDtype
from ..swap import SwapEntry, SwapManifest
from .gguf_tq import _scale_bytes, _ternary_to_unsigned, require_gguf, write_gguf

I2S_BLOCK_SIZE: int = 128

# per-tensor f32 scale in the first 4 bytes, appended once after every block
I2S_SCALE_TAIL_BYTES: int = 32

# ggml type id of bitnet.cpp's I2_S; upstream gguf does not define it
I2S_GGML_TYPE_ID: int = 36

_I2S_LANES: int = 4
_I2S_LANE_STRIDE: int = I2S_BLOCK_SIZE // _I2S_LANES


def pack_i2s(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes into the I2_S layout.

    Codes map `{-1, 0, +1} → {0, 1, 2}` and are serialized in 128-value blocks where
    the four values at stride 32 share a byte; a 32-byte tail carries the per-tensor
    scale as a little-endian f32 in its first four bytes.

    Args:
        codes: int8 codes of shape `(out_features, in_features)`.
        scale: fp32 per-tensor `s16` scale.

    Returns:
        Flat uint8 tensor of the packed tensor including its scale tail.

    Raises:
        ValueError: If the row length is not a multiple of `I2S_BLOCK_SIZE`.
    """
    unsigned = _ternary_to_unsigned(codes, I2S_BLOCK_SIZE)
    lanes = unsigned.reshape(
        unsigned.shape[0], unsigned.shape[1], _I2S_LANES, _I2S_LANE_STRIDE
    ).to(torch.int16)
    body = (lanes << _lane_shifts(lanes.device)[:, None]).sum(dim=-2).to(torch.uint8)
    tail = torch.zeros(I2S_SCALE_TAIL_BYTES, dtype=torch.uint8, device=codes.device)
    tail[:4] = _scale_bytes(scale, "<f4", codes.device)
    return torch.cat([body.reshape(-1), tail])


def unpack_i2s(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_i2s`, returning int8 codes of `shape` (scale tail dropped)."""
    rows, cols = shape
    if cols % I2S_BLOCK_SIZE:
        raise ValueError(
            f"in_features {cols} is not a multiple of the {I2S_BLOCK_SIZE}-weight block size"
        )
    expected = rows * cols // _I2S_LANES + I2S_SCALE_TAIL_BYTES
    if packed.numel() != expected:
        raise ValueError(
            f"packed tensor has {packed.numel()} bytes, expected {expected} for {shape}"
        )
    body = packed[: rows * cols // _I2S_LANES].reshape(
        rows, cols // I2S_BLOCK_SIZE, 1, _I2S_LANE_STRIDE
    )
    lanes = (body.to(torch.int16) >> _lane_shifts(packed.device)[:, None]) & 3
    return (lanes.reshape(rows, cols) - 1).to(torch.int8)


def decode_i2s(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(codes, per-tensor scale)` read back out of I2_S bytes."""
    rows, cols = shape
    codes = unpack_i2s(packed, shape)
    tail = packed[rows * cols // _I2S_LANES :][:4]
    scale = tail.contiguous().cpu().numpy().view("<f4").astype(np.float32)
    return codes, torch.from_numpy(scale)


def export_i2s(
    master_dir: str | Path,
    output_dir: str | Path,
    manifest: SwapManifest,
    gate: bool = True,
    embedding_dtype: EmbeddingDtype = "bf16",
    router_dtype: RouterDtype = "bf16",
) -> Path:
    """Write a bitnet.cpp-compatible I2_S GGUF file and return its path.

    Args:
        master_dir: Directory holding the baked master.
        output_dir: Where the GGUF file is written.
        manifest: The swap manifest naming the ternary tensors.
        gate: Round-trip every packed tensor's bytes back to codes before they reach
            the container writer.
        embedding_dtype: `int8` packs the token embeddings and output head as Q8_0.
        router_dtype: `int8` packs the kept-FP MoE router tensors as Q8_0.

    Raises:
        ImportError: If the `gguf` package is not installed.
        ValueError: If the manifest scale mode or architecture is unsupported.
        RuntimeError: If `gate` is set and a tensor's blocks do not round-trip.
    """
    gguf = require_gguf()
    ggml_type = getattr(gguf.GGMLQuantizationType, "I2_S", I2S_GGML_TYPE_ID)

    def pack_entry(
        codes: torch.Tensor, scale: torch.Tensor, entry: SwapEntry
    ) -> tuple[np.ndarray, tuple[int, int] | None]:
        packed = pack_i2s(codes, scale).numpy()
        # gguf re-derives a uint8 tensor's shape from block metadata that I2_S,
        # being a bitnet.cpp type, is not registered with; an int8 view is the same
        # bytes on disk and keeps the declared shape.
        return packed.view(np.int8), (entry.out_features, entry.in_features)

    return write_gguf(
        master_dir=master_dir,
        output_dir=output_dir,
        manifest=manifest,
        pack_entry=pack_entry,
        ggml_type=ggml_type,
        filename="ggml-model-i2_s.gguf",
        gate_format="i2_s" if gate else None,
        embedding_dtype=embedding_dtype,
        router_dtype=router_dtype,
    )


def _lane_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(_I2S_LANES - 1, -1, -1, dtype=torch.int16, device=device) * 2


__all__ = [
    "I2S_BLOCK_SIZE",
    "I2S_GGML_TYPE_ID",
    "I2S_SCALE_TAIL_BYTES",
    "export_i2s",
    "pack_i2s",
    "unpack_i2s",
]
