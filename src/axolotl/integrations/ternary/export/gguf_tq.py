"""GGUF TQ2_0 / TQ1_0 writers (llama.cpp ternary block formats).

The `gguf` package is an optional dependency; every entry point here raises a clear
`ImportError` when it is missing rather than failing at import time. The block packers
themselves are pure torch and always available.

Block layouts (256 weights per block, one f16 scale each):

- **TQ2_0**, 66 B/block (2.0625 bpw): `qs[64]` then the f16 scale. A block is two
  128-element halves; inside a half the four elements 32 apart share a byte at bit
  shifts 0/2/4/6, storing `code + 1`.
- **TQ1_0**, 54 B/block (1.6875 bpw): `qs[48]`, `qh[4]`, then the f16 scale. Bytes hold
  five base-3 trits, most significant first, rescaled by `ceil(v * 256 / 243)` so a
  decoder recovers trit `n` with the uint8 arithmetic `((byte * 3**n & 0xFF) * 3) >> 8`.
  `qs[0:32]` covers elements 0..159 (stride 32), `qs[32:48]` elements 160..239
  (stride 16), and the four `qh` bytes carry four trits each for elements 240..255
  (stride 4), placed in the top four of the five trit slots.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from axolotl.utils.logging import get_logger

from .. import aux_modules, quant
from ..args import NON_PER_TENSOR_SCALE_MODES, EmbeddingDtype, RouterDtype
from ..swap import SwapEntry, SwapManifest
from .bake import derive_codes_and_scale, load_master

LOG = get_logger(__name__)

GGUFTernaryType = Literal["tq2_0", "tq1_0"]

GATE_RECORD_FILENAME: str = "ternary_block_gate.json"

# Q8_0 (llama.cpp): 32 weights per block, an f16 scale then 32 int8 codes
Q8_0_BLOCK_SIZE: int = 32
Q8_0_BLOCK_BYTES: int = Q8_0_BLOCK_SIZE + 2
Q8_0_QMAX: int = 127

# smallest positive f16 subnormal, the floor a Q8_0 block scale can be stored at
F16_MIN_POSITIVE: float = 2.0**-24

# GGUF tensor names the embedding dtype applies to; with tied embeddings only the
# first exists and llama.cpp derives the output head from it
EMBEDDING_TENSOR_NAMES: frozenset[str] = frozenset(
    {"token_embd.weight", "output.weight"}
)

# the aux-module family `router_dtype` applies to
ROUTER_FAMILY_KEY: str = "routers"

# parameter suffixes a state-dict key carries over its module name
_PARAM_SUFFIXES: tuple[str, ...] = (".weight", ".bias")

_MAX_REPORTED_ROUTERS: int = 3

# tensors spelled out per family before the refusal truncates
_MAX_REPORTED_TENSORS: int = 5

QK_K: int = 256

TQ2_0_QS_BYTES: int = QK_K // 4
TQ2_0_BLOCK_BYTES: int = TQ2_0_QS_BYTES + 2

TQ1_0_QS_BYTES: int = (QK_K - 4 * QK_K // 64) // 5
TQ1_0_QH_BYTES: int = QK_K // 64
TQ1_0_BLOCK_BYTES: int = TQ1_0_QS_BYTES + TQ1_0_QH_BYTES + 2

_TQ2_0_HALVES: int = 2
_TQ2_0_LANES: int = 4
_TQ2_0_LANE_STRIDE: int = QK_K // (_TQ2_0_HALVES * _TQ2_0_LANES)

_TQ1_0_TRITS: int = 5
_TQ1_0_BASE: int = 3**_TQ1_0_TRITS
# (bytes, trits per byte) groups, in element order: qs low, qs high, qh
_TQ1_0_GROUPS: tuple[tuple[int, int], ...] = ((32, 5), (16, 5), (TQ1_0_QH_BYTES, 4))

_GGUF_ARCH: dict[str, str] = {"llama": "LLAMA", "qwen3": "QWEN3"}

_PackEntry = Callable[
    [torch.Tensor, torch.Tensor, SwapEntry], "tuple[np.ndarray, tuple[int, int] | None]"
]


def gguf_available() -> bool:
    """Whether the optional `gguf` package can be imported."""
    return importlib.util.find_spec("gguf") is not None


def pack_tq2_0(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes into TQ2_0 blocks (2.0625 bpw, f16 block scale).

    Args:
        codes: int8 codes of shape `(out_features, in_features)`.
        scale: fp32 `s16` scale; per tensor, broadcast into every block scale.

    Returns:
        Flat uint8 tensor holding the block-serialized rows.

    Raises:
        ValueError: If `in_features` is not a multiple of the TQ2_0 block size.
    """
    unsigned = _ternary_to_unsigned(codes, QK_K)
    rows, n_blocks = unsigned.shape[0], unsigned.shape[1]
    lanes = unsigned.reshape(
        rows, n_blocks, _TQ2_0_HALVES, _TQ2_0_LANES, _TQ2_0_LANE_STRIDE
    ).to(torch.int16)
    shifts = _lane_shifts(lanes.device)
    qs = (lanes << shifts[:, None]).sum(dim=-2).to(torch.uint8)
    blocks = torch.cat(
        [
            qs.reshape(rows, n_blocks, TQ2_0_QS_BYTES),
            _scale_bytes(scale, "<f2", qs.device).expand(rows, n_blocks, 2),
        ],
        dim=-1,
    )
    return blocks.reshape(-1)


def unpack_tq2_0(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_tq2_0`, returning int8 codes of `shape` (block scales dropped)."""
    rows, cols = _check_packed(packed, shape, TQ2_0_BLOCK_BYTES)
    blocks = packed.reshape(rows, cols // QK_K, TQ2_0_BLOCK_BYTES)
    qs = blocks[..., :TQ2_0_QS_BYTES].to(torch.int16)
    qs = qs.reshape(rows, -1, _TQ2_0_HALVES, 1, _TQ2_0_LANE_STRIDE)
    lanes = (qs >> _lane_shifts(packed.device)[:, None]) & 3
    return (lanes.reshape(rows, cols) - 1).to(torch.int8)


def quantize_q8_0(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive the Q8_0 codes and per-block scales of a weight — the packer's oracle.

    Per 32-weight block: `d = f16_round(amax / 127)` and `q = round_half_even(w / d)`
    clipped to `[-127, 127]`, so `w ≈ q · d` with a half-step bound. The scale is
    rounded through f16 *before* the codes are derived, which is what makes `q · d`
    reproducible from the stored bytes with no drift.

    Args:
        weight: Embedding or head weight of shape `(rows, cols)`.

    Returns:
        `(codes, scales)` — int8 codes shaped like `weight`, and fp32 scales of shape
        `(rows, cols // Q8_0_BLOCK_SIZE)`. An all-zero block yields a zero scale.

    Raises:
        ValueError: If `weight` is not 2-D or its row length is not a multiple of 32.
    """
    blocks = _q8_0_blocks(weight)
    amax = blocks.abs().amax(dim=-1)
    scale = quant.f16_round_scale(amax / Q8_0_QMAX)
    # an f16-subnormal scale would underflow to zero and take the whole block with it
    scale = torch.where(amax > 0, scale.clamp_min(F16_MIN_POSITIVE), scale)
    divisor = torch.where(scale > 0, scale, torch.ones_like(scale)).unsqueeze(-1)
    codes = torch.round(blocks / divisor).clamp_(-Q8_0_QMAX, Q8_0_QMAX)
    return codes.to(torch.int8).reshape(weight.shape), scale


def pack_q8_0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Pack Q8_0 codes and block scales into llama.cpp's 34-byte blocks.

    Each block is the f16 scale (little-endian) followed by its 32 int8 codes.

    Args:
        codes: int8 codes of shape `(rows, cols)`.
        scales: fp32 scales of shape `(rows, cols // Q8_0_BLOCK_SIZE)`.

    Returns:
        Flat uint8 tensor of `rows * n_blocks * Q8_0_BLOCK_BYTES` bytes.

    Raises:
        ValueError: If the shapes disagree or the row length is not a multiple of 32.
    """
    blocks = _q8_0_blocks(codes)
    rows, n_blocks = blocks.shape[0], blocks.shape[1]
    if tuple(scales.shape) != (rows, n_blocks):
        raise ValueError(
            f"expected {(rows, n_blocks)} block scales for {tuple(codes.shape)} codes, "
            f"got {tuple(scales.shape)}"
        )
    scale_bytes = (
        torch.from_numpy(
            scales.detach().to(torch.float32).cpu().numpy().astype("<f2").view(np.uint8)
        )
        .to(codes.device)
        .reshape(rows, n_blocks, 2)
    )
    payload = blocks.to(torch.uint8).reshape(rows, n_blocks, Q8_0_BLOCK_SIZE)
    return torch.cat([scale_bytes, payload], dim=-1).reshape(-1)


def decode_q8_0(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of `pack_q8_0`: return `(int8 codes, fp32 block scales)`.

    Raises:
        ValueError: If the byte count does not match `shape`.
    """
    rows, cols = shape
    _check_q8_0_cols(cols)
    n_blocks = cols // Q8_0_BLOCK_SIZE
    expected = rows * n_blocks * Q8_0_BLOCK_BYTES
    if packed.numel() != expected:
        raise ValueError(
            f"packed tensor has {packed.numel()} bytes, expected {expected} for {shape}"
        )
    blocks = packed.reshape(rows * n_blocks, Q8_0_BLOCK_BYTES)
    scales = _decode_f16(blocks[:, :2]).reshape(rows, n_blocks)
    codes = blocks[:, 2:].reshape(rows, cols).view(torch.int8)
    return codes, scales


def dequantize_q8_0(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Return `codes * scales` in fp32, broadcasting each scale over its block."""
    blocks = _q8_0_blocks(codes).to(torch.float32)
    return (blocks * scales.to(torch.float32).unsqueeze(-1)).reshape(codes.shape)


def _q8_0_blocks(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(
            f"expected a 2D (rows, cols) tensor, got shape {tuple(tensor.shape)}"
        )
    rows, cols = tensor.shape
    _check_q8_0_cols(cols)
    return tensor.reshape(rows, cols // Q8_0_BLOCK_SIZE, Q8_0_BLOCK_SIZE)


def _check_q8_0_cols(cols: int) -> None:
    if cols % Q8_0_BLOCK_SIZE:
        raise ValueError(
            f"Q8_0 needs a row length that is a multiple of {Q8_0_BLOCK_SIZE}, got "
            f"{cols}; llama.cpp cannot load a padded row"
        )


def decode_tq2_0(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(codes, block scales)` read back out of TQ2_0 bytes."""
    rows, cols = _check_packed(packed, shape, TQ2_0_BLOCK_BYTES)
    blocks = packed.reshape(rows * (cols // QK_K), TQ2_0_BLOCK_BYTES)
    return unpack_tq2_0(packed, shape), _decode_f16(blocks[:, TQ2_0_QS_BYTES:])


def pack_tq1_0(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pack ternary codes into TQ1_0 blocks (1.6875 bpw, base-3 packed).

    Same arguments and errors as `pack_tq2_0`.
    """
    unsigned = _ternary_to_unsigned(codes, QK_K)
    rows, n_blocks = unsigned.shape[0], unsigned.shape[1]
    parts: list[torch.Tensor] = []
    start = 0
    for n_bytes, n_trits in _TQ1_0_GROUPS:
        span = n_bytes * n_trits
        parts.append(_pack_trits(unsigned[..., start : start + span], n_bytes, n_trits))
        start += span
    parts.append(_scale_bytes(scale, "<f2", unsigned.device).expand(rows, n_blocks, 2))
    return torch.cat(parts, dim=-1).reshape(-1)


def unpack_tq1_0(packed: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Inverse of `pack_tq1_0`, using the runtime's uint8 trit extraction."""
    rows, cols = _check_packed(packed, shape, TQ1_0_BLOCK_BYTES)
    blocks = packed.reshape(rows, cols // QK_K, TQ1_0_BLOCK_BYTES)
    parts: list[torch.Tensor] = []
    start = 0
    for n_bytes, n_trits in _TQ1_0_GROUPS:
        digits = _unpack_trits(blocks[..., start : start + n_bytes], n_trits)
        parts.append(digits.reshape(rows, -1, n_bytes * n_trits))
        start += n_bytes
    return (torch.cat(parts, dim=-1).reshape(rows, cols) - 1).to(torch.int8)


def decode_tq1_0(
    packed: torch.Tensor, shape: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(codes, block scales)` read back out of TQ1_0 bytes."""
    rows, cols = _check_packed(packed, shape, TQ1_0_BLOCK_BYTES)
    blocks = packed.reshape(rows * (cols // QK_K), TQ1_0_BLOCK_BYTES)
    scales = _decode_f16(blocks[:, TQ1_0_QS_BYTES + TQ1_0_QH_BYTES :])
    return unpack_tq1_0(packed, shape), scales


def _decode_f16(pairs: torch.Tensor) -> torch.Tensor:
    """Decode `(n, 2)` little-endian f16 bytes into an fp32 tensor of `n` scales."""
    raw = pairs.contiguous().cpu().numpy().reshape(-1).view("<f2")
    return torch.from_numpy(raw.astype(np.float32))


def export_gguf_tq(
    master_dir: str | Path,
    output_dir: str | Path,
    manifest: SwapManifest,
    quant_type: GGUFTernaryType = "tq2_0",
    gate: bool = True,
    embedding_dtype: EmbeddingDtype = "bf16",
    router_dtype: RouterDtype = "bf16",
) -> Path:
    """Write a GGUF file whose ternary tensors use TQ2_0/TQ1_0 and return its path.

    Embeddings and the output head are written as Q6_K/F16; norms stay F32.

    Args:
        master_dir: Directory holding the baked master.
        output_dir: Where the GGUF file is written.
        manifest: The swap manifest naming the ternary tensors.
        quant_type: `tq2_0` or `tq1_0`.
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
    if quant_type not in ("tq2_0", "tq1_0"):
        raise ValueError(
            f"unknown gguf ternary type {quant_type!r}; use tq2_0 or tq1_0"
        )
    ggml_type = getattr(gguf.GGMLQuantizationType, quant_type.upper(), None)
    if ggml_type is None:
        raise ImportError(
            f"the installed gguf package has no {quant_type.upper()} type; "
            "upgrade it (pip install -U gguf)"
        )
    packer = pack_tq2_0 if quant_type == "tq2_0" else pack_tq1_0

    def pack_entry(
        codes: torch.Tensor, scale: torch.Tensor, entry: SwapEntry
    ) -> tuple[np.ndarray, tuple[int, int] | None]:
        packed = packer(codes, scale).reshape(entry.out_features, -1)
        return packed.numpy(), None

    return write_gguf(
        master_dir=master_dir,
        output_dir=output_dir,
        manifest=manifest,
        pack_entry=pack_entry,
        ggml_type=ggml_type,
        filename=f"ggml-model-{quant_type}.gguf",
        file_type=getattr(gguf.LlamaFileType, f"MOSTLY_{quant_type.upper()}", None),
        gate_format=f"gguf_{quant_type}" if gate else None,
        embedding_dtype=embedding_dtype,
        router_dtype=router_dtype,
    )


def require_gguf() -> Any:
    """Import and return the optional `gguf` package.

    Raises:
        ImportError: If it is not installed.
    """
    try:
        return importlib.import_module("gguf")
    except ImportError as exc:
        raise ImportError(
            "GGUF export needs the optional `gguf` package: pip install gguf"
        ) from exc


def write_gguf(
    *,
    master_dir: str | Path,
    output_dir: str | Path,
    manifest: SwapManifest,
    pack_entry: _PackEntry,
    ggml_type: Any,
    filename: str,
    file_type: Any = None,
    gate_format: str | None = None,
    embedding_dtype: EmbeddingDtype = "bf16",
    router_dtype: RouterDtype = "bf16",
) -> Path:
    """Write one GGUF file from a baked master, packing ternary tensors with `pack_entry`.

    Shared by the TQ2_0/TQ1_0 and I2_S writers: everything except the block layout
    (tensor naming, dtypes of the kept-FP tensors, architecture metadata) is identical.

    Args:
        gate_format: Parity format name; when set, every packed tensor's serialized
            bytes are decoded again and gated on the master before the container
            writer sees them, and their digests are recorded beside the file.
        embedding_dtype: `int8` packs the token embeddings and the output head as
            Q8_0 instead of F16; with tied embeddings only one tensor exists and
            llama.cpp derives the head from it.
        router_dtype: `int8` packs the 2-D weights of kept-FP router modules as Q8_0
            instead of F16. Router biases stay F32 with the other 1-D tensors.

    Raises:
        ValueError: If the manifest scale mode or architecture is unsupported, the
            master carries tensors outside the text stack (see
            `assert_text_stack_only`), or a tensor selected for Q8_0 has a row length
            Q8_0 cannot block.
        RuntimeError: If a gated tensor's blocks do not round-trip to the master.
    """
    gguf = require_gguf()
    master_dir, output_dir = Path(master_dir), Path(output_dir)
    _check_per_tensor_scale(manifest)

    arch_key = _GGUF_ARCH.get(manifest.model_type)
    if arch_key is None:
        raise ValueError(
            f"gguf export does not support model_type {manifest.model_type!r} "
            f"(supported: {sorted(_GGUF_ARCH)})"
        )
    arch = getattr(gguf.MODEL_ARCH, arch_key)

    config = json.loads((master_dir / "config.json").read_text(encoding="utf-8"))
    state_dict, _ = load_master(master_dir)
    ternary = {f"{entry.name}.weight": entry for entry in manifest.entries}
    name_map = gguf.get_tensor_name_map(arch, int(config["num_hidden_layers"]))
    assert_text_stack_only(state_dict, name_map, manifest.model_type)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    digests: dict[str, str] = {}
    quantized_routers: list[str] = []
    writer = gguf.GGUFWriter(path, gguf.MODEL_ARCH_NAMES[arch])
    try:
        _add_metadata(writer, config, master_dir, file_type)
        for key, tensor in state_dict.items():
            if key.endswith(".inv_freq"):
                continue
            name = name_map.get_name(key, try_suffixes=_PARAM_SUFFIXES)
            entry = ternary.get(key)
            if entry is None:
                knob = _q8_0_knob(key, name, tensor, embedding_dtype, router_dtype)
                if knob is not None:
                    data = _pack_q8_0_tensor(key, knob, tensor)
                    if knob == "router_dtype":
                        quantized_routers.append(key)
                    if gate_format is not None:
                        digests[name] = gate_packed_bytes("q8_0", data, tensor)
                    writer.add_tensor(
                        name,
                        data,
                        raw_shape=tuple(tensor.shape),
                        raw_dtype=gguf.GGMLQuantizationType.Q8_0,
                    )
                    continue
                # 1-D tensors are norms; llama.cpp keeps them f32
                dtype = torch.float32 if tensor.ndim == 1 else torch.float16
                writer.add_tensor(name, tensor.to(dtype).contiguous().numpy())
                continue
            codes, scale = derive_codes_and_scale(tensor, manifest.group_size)
            data, raw_shape = pack_entry(codes, scale, entry)
            if gate_format is not None:
                digests[entry.name] = gate_packed_bytes(gate_format, data, tensor)
            writer.add_tensor(name, data, raw_shape=raw_shape, raw_dtype=ggml_type)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()

    if router_dtype == "int8":
        _report_quantized_routers(quantized_routers)
    if gate_format is not None:
        write_gate_record(output_dir, gate_format, path, digests)
    LOG.info(f"ternary: wrote {path}")
    return path


def assert_text_stack_only(
    state_dict: dict[str, torch.Tensor], name_map: Any, model_type: str
) -> None:
    """Refuse a master carrying tensors no GGUF tensor name fits.

    A GGUF tensor name map describes one text transformer stack. Anything bolted onto
    it — a vision or audio tower, its projector, an MTP or draft head — maps to
    nothing, and the writer would otherwise produce a file that loads, answers, and
    is missing a component the checkpoint was trained with.

    Args:
        state_dict: The baked master's tensors.
        name_map: The `gguf` tensor name map for this architecture.
        model_type: The manifest's model type, named in the error.

    Raises:
        ValueError: If any tensor has no GGUF name.
    """
    unnamed = [
        key
        for key in state_dict
        if not key.endswith(".inv_freq")
        and name_map.get_name(key, try_suffixes=_PARAM_SUFFIXES) is None
    ]
    if unnamed:
        raise ValueError(_unrepresentable_message(unnamed, model_type))


def _unrepresentable_message(keys: list[str], model_type: str) -> str:
    grouped: dict[str, list[str]] = {}
    unclaimed: list[str] = []
    for key in keys:
        family = aux_modules.classify(_module_name(key))
        if family is None:
            unclaimed.append(key)
        else:
            grouped.setdefault(family.key, []).append(key)

    plural = "" if len(keys) == 1 else "s"
    lines = [
        f"ternary: {len(keys)} tensor{plural} of this master have no gguf tensor name "
        f"under architecture {model_type!r}. A GGUF file describes one text "
        "transformer stack, so writing it would silently drop them and ship the "
        "language model alone:"
    ]
    for family_key, family in aux_modules.AUX_MODULE_FAMILIES.items():
        if family_key in grouped:
            lines.append(f"  {family.label}: {_listed(grouped[family_key])}")
    if unclaimed:
        lines.append(f"  claimed by no known family: {_listed(unclaimed)}")
    lines.append(
        "Export master_bf16 (the re-finetunable interchange artifact) or hf_bitnet "
        "(which keeps the source architecture, towers included), or point the GGUF "
        "export at a text-only checkpoint."
    )
    return "\n".join(lines)


def _listed(names: list[str]) -> str:
    head = ", ".join(names[:_MAX_REPORTED_TENSORS])
    extra = len(names) - _MAX_REPORTED_TENSORS
    return f"{head} (+{extra} more)" if extra > 0 else head


def _module_name(key: str) -> str:
    """Return the module name behind a state-dict key, for the aux-module registry."""
    for suffix in _PARAM_SUFFIXES:
        key = key.removesuffix(suffix)
    return key


def is_router_tensor(key: str) -> bool:
    """Whether a state-dict key names a parameter of an MoE router module."""
    module = key
    for suffix in _PARAM_SUFFIXES:
        module = module.removesuffix(suffix)
    family = aux_modules.classify(module)
    return family is not None and family.key == ROUTER_FAMILY_KEY


def _q8_0_knob(
    key: str,
    name: str,
    tensor: torch.Tensor,
    embedding_dtype: EmbeddingDtype,
    router_dtype: RouterDtype,
) -> str | None:
    """Return the export knob asking for this kept-FP tensor as Q8_0, or `None`."""
    if embedding_dtype == "int8" and name in EMBEDDING_TENSOR_NAMES:
        return "embedding_dtype"
    if router_dtype == "int8" and tensor.ndim == 2 and is_router_tensor(key):
        return "router_dtype"
    return None


def _pack_q8_0_tensor(key: str, knob: str, tensor: torch.Tensor) -> np.ndarray:
    try:
        return pack_q8_0(*quantize_q8_0(tensor)).numpy()
    except ValueError as exc:
        raise ValueError(
            f"ternary.export.{knob}: int8 cannot pack {key} "
            f"{tuple(tensor.shape)}: {exc}"
        ) from exc


def _report_quantized_routers(names: list[str]) -> None:
    if not names:
        LOG.warning(
            "ternary: export.router_dtype: int8 matched no router module in this "
            "model; every tensor was written at its default dtype"
        )
        return
    listed = ", ".join(names[:_MAX_REPORTED_ROUTERS])
    if len(names) > _MAX_REPORTED_ROUTERS:
        listed += f" (+{len(names) - _MAX_REPORTED_ROUTERS} more)"
    LOG.warning(
        f"ternary: packed {len(names)} router tensors as Q8_0: {listed}. "
        "llama.cpp's own quantizer never quantizes ffn_gate_inp, so this file is "
        "outside what upstream produces: check the runtime loads it and re-measure "
        "expert utilization before shipping it"
    )


def gate_packed_bytes(fmt: str, data: np.ndarray, master_weight: torch.Tensor) -> str:
    """Gate exactly the byte string the container writer will serialize.

    Args:
        fmt: Parity format name, or `q8_0` for a quantized embedding tensor.
        data: The array handed to `GGUFWriter.add_tensor`.
        master_weight: The baked master tensor it was packed from.

    Returns:
        The sha256 of the verified bytes, which are byte-for-byte the ones written.

    Raises:
        RuntimeError: If the bytes do not decode back to the master's codes and s16.
    """
    from . import parity

    payload = np.ascontiguousarray(data).tobytes()
    verified = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
    if fmt == "q8_0":
        failures = parity.gate_q8_0_bytes(verified, master_weight)
    else:
        failures = parity.gate_block_bytes(fmt, verified, master_weight)
    if failures:
        raise RuntimeError(
            f"ternary: the {fmt} block gate rejected a "
            f"{tuple(master_weight.shape)} tensor: " + "; ".join(failures)
        )
    return hashlib.sha256(payload).hexdigest()


def write_gate_record(
    output_dir: str | Path, fmt: str, artifact: Path, digests: dict[str, str]
) -> Path:
    """Record the gated tensor digests beside the artifact and return the record path."""
    path = Path(output_dir) / GATE_RECORD_FILENAME
    path.write_text(
        json.dumps(
            {"format": fmt, "artifact": artifact.name, "tensors": digests},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    LOG.info(f"ternary: {fmt} block gate passed for {len(digests)} tensors")
    return path


def _lane_shifts(device: torch.device) -> torch.Tensor:
    return torch.arange(_TQ2_0_LANES, dtype=torch.int16, device=device) * 2


def _ternary_to_unsigned(codes: torch.Tensor, block_size: int) -> torch.Tensor:
    """Validate codes and return them as `(rows, n_blocks, block_size)` uint8 in {0,1,2}."""
    if codes.ndim != 2:
        raise ValueError(
            f"expected (out_features, in_features) codes, got shape {tuple(codes.shape)}"
        )
    rows, cols = codes.shape
    if cols % block_size:
        raise ValueError(
            f"in_features {cols} is not a multiple of the {block_size}-weight block size"
        )
    unsigned = codes.to(torch.int16) + 1
    if unsigned.numel() and (int(unsigned.min()) < 0 or int(unsigned.max()) > 2):
        raise ValueError("ternary codes must all be -1, 0 or +1")
    return unsigned.to(torch.uint8).reshape(rows, cols // block_size, block_size)


def _scale_bytes(
    scale: torch.Tensor, dtype: str, device: torch.device | None = None
) -> torch.Tensor:
    """Return the little-endian byte encoding of a per-tensor scale."""
    if scale.numel() != 1:
        raise ValueError(
            f"expected a per-tensor scale, got {scale.numel()} values; group scales "
            "are not representable in this format"
        )
    raw = np.asarray([float(scale)], dtype=dtype).view(np.uint8).copy()
    return torch.from_numpy(raw).to(device)


def _check_packed(
    packed: torch.Tensor, shape: tuple[int, int], block_bytes: int
) -> tuple[int, int]:
    rows, cols = shape
    if cols % QK_K:
        raise ValueError(
            f"in_features {cols} is not a multiple of the {QK_K}-weight block size"
        )
    expected = rows * (cols // QK_K) * block_bytes
    if packed.numel() != expected:
        raise ValueError(
            f"packed tensor has {packed.numel()} bytes, expected {expected} for {shape}"
        )
    return rows, cols


def _pack_trits(unsigned: torch.Tensor, n_bytes: int, n_trits: int) -> torch.Tensor:
    """Pack `n_trits` stride-`n_bytes` trits per byte, base 3 rescaled by 256/243."""
    digits = unsigned.reshape(*unsigned.shape[:-1], n_trits, n_bytes).to(torch.int32)
    place = torch.tensor(
        [3 ** (_TQ1_0_TRITS - 1 - n) for n in range(n_trits)],
        dtype=torch.int32,
        device=unsigned.device,
    )
    value = (digits * place[:, None]).sum(dim=-2)
    return ((value * 256 + _TQ1_0_BASE - 1) // _TQ1_0_BASE).to(torch.uint8)


def _unpack_trits(packed: torch.Tensor, n_trits: int) -> torch.Tensor:
    """Recover trits with the runtime's uint8 extraction: `((b * 3**n & 0xFF) * 3) >> 8`."""
    scaled = packed.to(torch.int32)
    digits = [(((scaled * 3**n) & 0xFF) * 3) >> 8 for n in range(n_trits)]
    return torch.stack(digits, dim=-2)


def _check_per_tensor_scale(manifest: SwapManifest) -> None:
    if (
        manifest.group_size is not None
        or manifest.weight_scale in NON_PER_TENSOR_SCALE_MODES
    ):
        raise ValueError(
            "GGUF ternary formats carry one f16 scale per 256-weight block derived from "
            f"a single per-tensor scale; ternary.weight_scale: {manifest.weight_scale} "
            f"(group_size {manifest.group_size}) cannot be represented"
        )


def _add_metadata(writer: Any, config: dict, master_dir: Path, file_type: Any) -> None:
    n_heads = int(config["num_attention_heads"])
    hidden = int(config["hidden_size"])
    head_dim = int(config.get("head_dim") or hidden // n_heads)

    writer.add_name(master_dir.name)
    writer.add_block_count(int(config["num_hidden_layers"]))
    writer.add_context_length(int(config.get("max_position_embeddings", 2048)))
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(int(config["intermediate_size"]))
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(int(config.get("num_key_value_heads", n_heads)))
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_dimension_count(head_dim)
    writer.add_rope_freq_base(float(config.get("rope_theta", 10000.0)))
    writer.add_layer_norm_rms_eps(float(config.get("rms_norm_eps", 1e-5)))
    if file_type is not None:
        writer.add_file_type(file_type)

    try:
        _add_vocab(writer, config, master_dir)
    except Exception as exc:  # vocab is best-effort metadata, tensors are the product
        LOG.warning(
            f"ternary: could not write a tokenizer into the GGUF ({exc}); merge one "
            "from convert_hf_to_gguf.py before running the file"
        )


def _add_vocab(writer: Any, config: dict, master_dir: Path) -> None:
    """Write the BPE vocab from `tokenizer.json`, plus special tokens via gguf."""
    gguf = require_gguf()
    tokenizer_path = master_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"no tokenizer.json in {master_dir}")
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = tokenizer.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"unsupported tokenizer model {model.get('type')!r}; BPE only")

    vocab: dict[str, int] = model.get("vocab", {})
    by_id = {index: token for token, index in vocab.items()}
    added = {int(item["id"]): item for item in tokenizer.get("added_tokens", [])}
    tokens: list[str] = []
    types: list[int] = []
    for index in range(int(config.get("vocab_size") or len(vocab))):
        if index in added:
            tokens.append(added[index]["content"])
            types.append(
                gguf.TokenType.CONTROL
                if added[index].get("special")
                else gguf.TokenType.USER_DEFINED
            )
        elif index in by_id:
            tokens.append(by_id[index])
            types.append(gguf.TokenType.NORMAL)
        else:
            tokens.append(f"[PAD{index}]")
            types.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    gguf.SpecialVocab(master_dir, load_merges=True).add_to_gguf(writer)
    LOG.warning(
        "ternary: the GGUF carries no tokenizer.ggml.pre entry; llama.cpp will fall "
        "back to the default pre-tokenizer"
    )


__all__ = [
    "EMBEDDING_TENSOR_NAMES",
    "GATE_RECORD_FILENAME",
    "GGUFTernaryType",
    "QK_K",
    "Q8_0_BLOCK_BYTES",
    "Q8_0_BLOCK_SIZE",
    "Q8_0_QMAX",
    "ROUTER_FAMILY_KEY",
    "TQ1_0_BLOCK_BYTES",
    "TQ1_0_QH_BYTES",
    "TQ1_0_QS_BYTES",
    "TQ2_0_BLOCK_BYTES",
    "TQ2_0_QS_BYTES",
    "assert_text_stack_only",
    "decode_q8_0",
    "dequantize_q8_0",
    "export_gguf_tq",
    "gate_packed_bytes",
    "gguf_available",
    "is_router_tensor",
    "pack_q8_0",
    "pack_tq1_0",
    "pack_tq2_0",
    "quantize_q8_0",
    "require_gguf",
    "unpack_tq1_0",
    "unpack_tq2_0",
    "write_gguf",
]
