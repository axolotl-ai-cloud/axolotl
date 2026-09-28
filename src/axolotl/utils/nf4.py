"""Bounded 4-bit conversion and serializable frozen-weight parametrizations."""

import functools
import math
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bitsandbytes.functional import QuantState

import torch
from torch import nn


def torchao_nf4_module():
    """Resolve the NF4 implementation before and after torchao's 0.18 module move."""
    from importlib import import_module

    try:
        return import_module("torchao.dtypes.nf4tensor")
    except ModuleNotFoundError as exc:
        if exc.name != "torchao.dtypes.nf4tensor":
            raise
        return import_module("torchao.quantization.quantize_.workflows.nf4.nf4_tensor")


def quantize_bnb_4bit(
    value: torch.Tensor,
    *,
    blocksize: int = 64,
    compress_statistics: bool = True,
    quant_type: str = "nf4",
    quant_storage: torch.dtype = torch.uint8,
    chunk_size: int = 2**26,
    device: torch.device | str | None = None,
    storage_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, "QuantState"]:
    """Quantize independent blocks, then compress the complete scale vector once.

    Args:
        value: Floating-point weights in their training dtype.
        blocksize: Number of weights sharing a quantization scale.
        compress_statistics: Double-quantize the complete scale vector.
        quant_type: Bitsandbytes codebook (NF4 or FP4).
        quant_storage: Dtype used to view the packed bytes.
        chunk_size: Maximum elements per native kernel call; rounded to whole blocks.
        device: Device used for quantization, defaulting to the input device.
        storage_device: Device for packed weights and metadata, defaulting to the input device.

    Returns:
        Packed weights and a standard bitsandbytes quantization state.
    """
    import bitsandbytes.functional as F

    chunk_size = chunk_size // blocksize * blocksize
    if not 0 < chunk_size < 2**31:
        raise ValueError(
            "quantization chunk size must be block-aligned and below 2**31"
        )
    device = device or value.device
    storage_device = storage_device or value.device
    flat = value.reshape(-1)
    packed = torch.empty(
        (flat.numel() + 1) // 2, dtype=torch.uint8, device=storage_device
    )
    scales = torch.empty(
        math.ceil(flat.numel() / blocksize), device=device, dtype=torch.float32
    )
    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        data, state = F.quantize_4bit(
            flat[start:end].to(device),
            blocksize=blocksize,
            compress_statistics=False,
            quant_type=quant_type,
        )
        packed[start // 2 : (end + 1) // 2].copy_(data.flatten())
        scales[start // blocksize : math.ceil(end / blocksize)].copy_(state.absmax)
    offset = state2 = None
    if compress_statistics:
        offset = scales.mean()
        scales, state2 = F.quantize_blockwise(scales - offset, blocksize=256)
    state = F.QuantState(
        absmax=scales,
        shape=value.shape,
        dtype=value.dtype,
        blocksize=blocksize,
        code=F.get_4bit_type(quant_type, device=device),
        quant_type=quant_type,
        offset=offset,
        state2=state2,
    )
    state.to(storage_device)
    return packed.view(quant_storage).reshape(-1, 1), state


def dequantize_bnb_4bit(
    data: torch.Tensor,
    state: "QuantState",
    *,
    out: torch.Tensor | None = None,
    chunk_size: int = 2**26,
) -> torch.Tensor:
    """Dequantize with bounded kernel element counts and a single output allocation."""
    import bitsandbytes.functional as F

    chunk_size = chunk_size // state.blocksize * state.blocksize
    if not 0 < chunk_size < 2**31:
        raise ValueError(
            "dequantization chunk size must be block-aligned and below 2**31"
        )
    scales = state.absmax
    if state.nested:
        scales = F.dequantize_blockwise(scales, state.state2) + state.offset
    if out is None:
        out = torch.empty(state.shape, device=data.device, dtype=state.dtype)
    if tuple(out.shape) != tuple(state.shape) or not out.is_contiguous():
        raise ValueError(
            "dequantization output must be contiguous and match the original shape"
        )
    raw = data.view(torch.uint8).reshape(-1)
    flat = out.view(-1)
    for start in range(0, flat.numel(), chunk_size):
        end = min(start + chunk_size, flat.numel())
        chunk_state = F.QuantState(
            absmax=scales[start // state.blocksize : math.ceil(end / state.blocksize)],
            shape=(end - start,),
            dtype=state.dtype,
            blocksize=state.blocksize,
            code=state.code,
            quant_type=state.quant_type,
        )
        chunk_data = raw[start // 2 : (end + 1) // 2].reshape(-1, 1)
        # the kernel can only write into an output on its own device, so a CPU
        # destination for accelerator-resident data goes through a chunk-sized copy
        if data.device.type == "cpu" or out.device != data.device:
            flat[start:end].copy_(
                F.dequantize_4bit(chunk_data, chunk_state).reshape(-1)
            )
        else:
            F.dequantize_4bit(chunk_data, chunk_state, out=flat[start:end])
    return out


class BnbNF4Parametrization(nn.Module):
    """Keep quantization metadata in buffers so device moves and FSDP loading include it."""

    def __init__(self, state):
        super().__init__()
        self.shape = tuple(state.shape)
        self.dtype = state.dtype
        self.blocksize = state.blocksize
        self.quant_type = state.quant_type
        self.register_buffer("absmax", state.absmax)
        self.register_buffer("code", state.code)
        self.register_buffer("offset", state.offset)
        self.nested = state.nested
        if state.nested:
            self.nested_blocksize = state.state2.blocksize
            self.nested_dtype = state.state2.dtype
            self.register_buffer("nested_absmax", state.state2.absmax)
            self.register_buffer("nested_code", state.state2.code)

    @property
    def quant_state(self):
        from bitsandbytes.functional import QuantState

        nested_state = None
        if self.nested:
            nested_state = QuantState(
                absmax=self.nested_absmax,
                code=self.nested_code,
                blocksize=self.nested_blocksize,
                dtype=self.nested_dtype,
            )
        return QuantState(
            absmax=self.absmax,
            code=self.code,
            shape=torch.Size(self.shape),
            dtype=self.dtype,
            blocksize=self.blocksize,
            quant_type=self.quant_type,
            offset=self.offset,
            state2=nested_state,
        )

    @torch.no_grad()
    def forward(self, data):
        if data.is_meta:
            return torch.empty(self.shape, dtype=self.dtype, device="meta")
        return dequantize_bnb_4bit(data, self.quant_state)


def prequantized_bnb_4bit(
    value: torch.Tensor,
) -> tuple[torch.Tensor, BnbNF4Parametrization]:
    """Adopt a checkpoint's 4-bit payload in the layout ``quantize_bnb_4bit`` produces.

    Args:
        value: A bitsandbytes ``Params4bit`` restored from serialized components.

    Returns:
        Packed weights and the parametrization that dequantizes them.
    """
    state = value.quant_state
    data = value.detach().as_subclass(torch.Tensor).view(torch.uint8).reshape(-1, 1)
    packed = (math.prod(state.shape) + 1) // 2
    if data.numel() != packed:
        raise ValueError(
            f"prequantized weight holds {data.numel()} packed bytes for a "
            f"{tuple(state.shape)} tensor that needs {packed}; floating-point "
            "bnb_4bit_quant_storage is reinterpreted when the checkpoint is cast to "
            "another compute dtype, so train in the checkpoint's storage dtype"
        )
    return data, BnbNF4Parametrization(state)


class TorchaoNF4Parametrization(nn.Module):
    """Reconstruct torchao NF4Tensor chunks from FSDP-compatible packed storage."""

    def __init__(self, shape, dtype):
        super().__init__()
        self.shape = tuple(shape)
        self.dtype = dtype
        self.chunks = []

    def add_chunk(self, tensor, numel):
        names, metadata = tensor.__tensor_flatten__()
        index = len(self.chunks)
        for name in names:
            if name != "quantized_data":
                self.register_buffer(f"chunk_{index}_{name}", getattr(tensor, name))
        self.chunks.append((names, metadata, numel, tensor.quantized_data.numel()))

    @torch.no_grad()
    def forward(self, data, active_experts=None):
        from dataclasses import replace

        NF4Tensor = torchao_nf4_module().NF4Tensor

        if data.is_meta:
            return torch.empty(self.shape, dtype=self.dtype, device="meta")
        output_shape = (
            self.shape
            if active_experts is None
            else (len(active_experts), *self.shape[1:])
        )
        result = torch.empty(output_shape, dtype=self.dtype, device=data.device)
        expert_size = math.prod(self.shape[1:])
        selected = None if active_experts is None else active_experts.tolist()
        source_offset = target_offset = 0
        for index, (names, metadata, numel, packed_size) in enumerate(self.chunks):
            overlaps = (
                None
                if selected is None
                else [
                    (
                        row,
                        max(expert * expert_size, target_offset),
                        min((expert + 1) * expert_size, target_offset + numel),
                        expert,
                    )
                    for row, expert in enumerate(selected)
                    if expert * expert_size < target_offset + numel
                    and (expert + 1) * expert_size > target_offset
                ]
            )
            if overlaps == []:
                source_offset += packed_size
                target_offset += numel
                continue
            inner = {
                name: getattr(self, f"chunk_{index}_{name}")
                for name in names
                if name != "quantized_data"
            }
            inner["quantized_data"] = data.flatten()[
                source_offset : source_offset + packed_size
            ]
            context = dict(metadata)
            context["tensor_meta"] = replace(context["tensor_meta"], device=data.device)
            tensor = NF4Tensor.__tensor_unflatten__(inner, context, None, None)
            dense = tensor.get_original_weight().flatten()[:numel]
            if selected is None:
                result.view(-1)[target_offset : target_offset + numel].copy_(dense)
            else:
                for row, start, end, expert in overlaps:
                    destination = row * expert_size + start - expert * expert_size
                    result.view(-1)[destination : destination + end - start].copy_(
                        dense[start - target_offset : end - target_offset]
                    )
            source_offset += packed_size
            target_offset += numel
        return result


def quantize_torchao_nf4(
    value: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    storage_device: torch.device | str | None = None,
    chunk_size: int = 2**20,
) -> tuple[torch.Tensor, TorchaoNF4Parametrization]:
    """Quantize bounded chunks with independent double-quantized scales.

    Args:
        value: Floating-point weights in their training dtype, of any shape.
        device: Quantization device, defaulting to the input device.
        storage_device: Packed storage device, defaulting to the input device.
        chunk_size: Maximum elements per chunk, rounded to 64 * 256 alignment.
            Loading and merging must use the same chunk size.

    Returns:
        Packed bytes and the parametrization containing per-chunk metadata.
    """
    to_nf4 = torchao_nf4_module().to_nf4

    device = device or value.device
    storage_device = storage_device or value.device
    alignment = 64 * 256
    chunk_size = chunk_size // alignment * alignment
    if chunk_size <= 0:
        raise ValueError("torchao chunk size must be at least 16384")
    flat = value.reshape(-1)
    total = math.ceil(flat.numel() / alignment) * alignment
    packed = torch.empty(total // 2, dtype=torch.uint8, device=storage_device)
    parametrization = TorchaoNF4Parametrization(value.shape, value.dtype)
    offset = 0
    for start in range(0, flat.numel(), chunk_size):
        chunk = flat[start : start + chunk_size].to(device)
        numel = chunk.numel()
        padding = (-numel) % alignment
        if padding:
            chunk = torch.nn.functional.pad(chunk, (0, padding))
        tensor = to_nf4(chunk.reshape(-1, 64))
        size = tensor.quantized_data.numel()
        packed[offset : offset + size].copy_(tensor.quantized_data)
        parametrization.add_chunk(tensor.to(storage_device), numel)
        offset += size
    return packed.reshape(-1, 1), parametrization


def _checkpointed_linear_forward(self: nn.Linear, *args, **kwargs) -> torch.Tensor:
    from torch.utils.checkpoint import checkpoint

    if not torch.is_grad_enabled():
        return self._nf4_original_forward(*args, **kwargs)
    return checkpoint(self._nf4_original_forward, *args, use_reentrant=False, **kwargs)


def checkpoint_nf4_linear(module: nn.Linear) -> None:
    """Recompute frozen dense weights during backward instead of retaining them.

    Checkpoint only the base linear operation, preserving the module's original
    forward implementation and PEFT's separate adapter gradients.
    """
    from types import MethodType

    if hasattr(module, "_nf4_original_forward"):
        return
    module._nf4_original_forward = module.forward
    module.forward = MethodType(_checkpointed_linear_forward, module)


def nf4_skip_modules(model_type: str | None, quantization: dict) -> set[str]:
    """Resolve user and architecture exclusions shared by loading and merging.

    Entries are matched by ``nf4_skip_matches``: plain module names or dotted
    paths by name, entries with a leading or trailing ``.`` as substrings, and
    entries containing regex metacharacters as regular expressions.
    """
    skips = {"lm_head", "embed_out"}
    skips.update(quantization.get("llm_int8_skip_modules") or [])
    if model_type == "falcon_h1":
        skips.add("out_proj")
    for key in skips:
        _skip_matcher(key)
    return skips


_REGEX_METACHARACTERS = frozenset("*+?[](){}^$|\\")


def nf4_skip_tier(key: str) -> str:
    """Name the rule ``nf4_skip_matches`` applies to one exclusion entry."""
    # regex first: an escaped dot at either end would otherwise read as substring
    if _REGEX_METACHARACTERS & set(key):
        return "regex"
    if key.startswith(".") or key.endswith("."):
        return "substring"
    return "name"


@functools.lru_cache(maxsize=None)
def _skip_matcher(key: str):
    tier = nf4_skip_tier(key)
    if tier == "regex":
        try:
            pattern = re.compile(key)
        except re.error as exc:
            raise ValueError(
                f"Invalid regex in llm_int8_skip_modules: {key!r}"
            ) from exc
        return lambda name: pattern.search(name) is not None
    if tier == "substring":
        return lambda name: key in name
    prefix = key + "."
    return lambda name: name == key or name.startswith(prefix) or key in name.split(".")


def nf4_skip_matches(name: str, key: str) -> bool:
    """Match one exclusion entry against a full parameter path.

    - ``lm_head`` or ``model.layers.0.mlp``: the full path, a dotted prefix of it
      (the whole subtree), or any single path component (that module name at
      every depth). ``proj`` does not match ``q_proj``.
    - ``_proj.`` or ``.experts``: a leading or trailing dot requests a plain
      substring match against the full parameter path.
    - ``.*_proj$`` or ``layers\\.[0-3]\\.``: regex metacharacters request
      ``re.search`` over the full path.
    """
    return _skip_matcher(key)(name)


def nf4_should_quantize(
    name: str, *, linear: bool, expert: bool, skips: set[str]
) -> bool:
    """Decide whether a Linear weight or fused expert tensor is quantized."""
    return (linear or expert) and not any(nf4_skip_matches(name, key) for key in skips)
