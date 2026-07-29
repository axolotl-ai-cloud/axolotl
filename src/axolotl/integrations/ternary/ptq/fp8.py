"""Reading fp8 source checkpoints, so a fit can start from one.

An fp8 checkpoint stores each quantized weight as `float8_e4m3` payload plus a
companion scale tensor, and often a second scale describing the activations a static
quantizer expects. A ternary fit wants neither: it wants the float weight back, and
the scales must not reach the master — a leftover `weight_scale` beside a ternary
tensor tells every downstream loader to dequantize values that are already dequantized.

Naming is not standardized. `mistralai/Mistral-Medium-3.5-128B` alone ships two
layouts of the same weights: the transformers one (`weight_scale_inv` /
`activation_scale`) and the Mistral consolidated one (`qscale_weight` / `qscale_act`),
neither of which is the `weight_scale` / `input_scale` pair that compressed-tensors
and vLLM write. All of them are recognized here.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# every fp8 payload dtype a checkpoint might store weights in
FP8_DTYPES: frozenset[torch.dtype] = frozenset(
    dtype
    for name in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
    if (dtype := getattr(torch, name, None)) is not None
)

# suffixes naming the scale that dequantizes a weight. `weight_scale_inv` is the
# transformers/DeepSeek spelling, `qscale_weight` the Mistral one, `weight_scale` the
# compressed-tensors one; all three multiply the payload.
WEIGHT_SCALE_SUFFIXES: tuple[str, ...] = (
    "weight_scale_inv",
    "weight_scale",
    "qscale_weight",
    "scale_weight",
)

# suffixes naming a static-activation scale, which describes the *inputs* a quantized
# kernel expects and has no meaning once the weight is float again
ACT_SCALE_SUFFIXES: tuple[str, ...] = (
    "activation_scale",
    "input_scale",
    "qscale_act",
    "act_scale",
    "input_scale_ub",
)

# the master is bf16, so that is where a dequantized weight lands
OUTPUT_DTYPE: torch.dtype = torch.bfloat16

_WEIGHT_SUFFIX: str = ".weight"


def is_fp8(dtype: torch.dtype) -> bool:
    """Whether `dtype` is an fp8 payload type."""
    return dtype in FP8_DTYPES


def dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Return `weight · scale` as bf16.

    Through fp32 and rounded exactly once: every e4m3 value is exactly representable
    in fp32, so the product is computed without error and only the final cast rounds.
    Going via the weight's own dtype instead would round twice.

    Args:
        weight: The fp8 payload.
        scale: Its companion scale — a scalar, or anything that broadcasts against
            `weight` (a per-output-channel column).

    Returns:
        The dequantized weight in `OUTPUT_DTYPE`.

    Raises:
        ValueError: If `scale` neither is a scalar nor broadcasts against `weight` —
            block-wise fp8 (`weight_block_size` set) needs an expansion this does not
            do, and silently mis-broadcasting it would corrupt the master.
    """
    values = weight.to(torch.float32)
    factor = scale.to(torch.float32)
    if factor.numel() != 1:
        factor = _broadcastable(factor, weight)
    return (values * factor).to(OUTPUT_DTYPE)


def _broadcastable(scale: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Return `scale` shaped so it broadcasts over `weight`, or raise."""
    if scale.ndim == 1 and scale.numel() == weight.shape[0]:
        return scale.reshape(-1, 1)
    try:
        torch.broadcast_shapes(scale.shape, weight.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"an fp8 scale of shape {tuple(scale.shape)} does not broadcast over a "
            f"{tuple(weight.shape)} weight. Block-wise fp8 (a non-null "
            "weight_block_size) is not supported: its scale grid has to be expanded "
            "per block, and broadcasting it would silently rescale the wrong values"
        ) from exc
    return scale


def scale_keys(keys: Iterable[str]) -> dict[str, str]:
    """Map each fp8 weight key to the companion scale key that dequantizes it.

    Only pairs whose weight actually exists are returned, so a tensor that merely
    happens to end in one of the suffixes is never mistaken for a companion.

    Args:
        keys: Every key in one shard.

    Returns:
        `{weight_key: weight_scale_key}`.
    """
    present = set(keys)
    found: dict[str, str] = {}
    for key in present:
        if not key.endswith(_WEIGHT_SUFFIX):
            continue
        module = key[: -len(_WEIGHT_SUFFIX)]
        for suffix in WEIGHT_SCALE_SUFFIXES:
            candidate = f"{module}.{suffix}"
            if candidate in present:
                found[key] = candidate
                break
    return found


def companion_keys(keys: Iterable[str]) -> set[str]:
    """Return every scale key that belongs to an fp8 weight and must be dropped.

    Both halves matter: the weight scale is consumed by the dequantization, and the
    activation scale describes a quantized kernel's inputs, which the master has none
    of. Either one surviving into the output would be read as a live quantizer.
    """
    present = set(keys)
    drop = set(scale_keys(present).values())
    for key in present:
        if not key.endswith(_WEIGHT_SUFFIX):
            continue
        module = key[: -len(_WEIGHT_SUFFIX)]
        drop.update(
            candidate
            for suffix in ACT_SCALE_SUFFIXES
            if (candidate := f"{module}.{suffix}") in present
        )
    return drop


def dequantized_items(
    tensors: Mapping[str, torch.Tensor],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield `tensors` with fp8 weights dequantized and their scales dropped.

    Anything that is not fp8 passes through untouched — an fp8 checkpoint's shards are
    mixed, with embeddings, norms and any module the quantizer skipped still bf16.
    """
    drop = companion_keys(tensors)
    scales = scale_keys(tensors)
    for key, tensor in tensors.items():
        if key in drop:
            continue
        scale_key = scales.get(key)
        if scale_key is not None and is_fp8(tensor.dtype):
            yield key, dequantize(tensor, tensors[scale_key])
            continue
        yield key, tensor


def assert_no_orphan_fp8(key: str, dtype: torch.dtype) -> None:
    """Raise if an fp8 tensor reached a consumer with no scale to dequantize it.

    Reading an e4m3 payload as though it were a float weight would fit the ternary
    grid to the raw exponent-mantissa bytes, which is silent, plausible-looking
    garbage — so this is an error, never a warning.
    """
    if is_fp8(dtype):
        raise ValueError(
            f"{key} holds {dtype} but no companion scale was found beside it "
            f"(looked for {', '.join(WEIGHT_SCALE_SUFFIXES)}). Fitting the raw fp8 "
            "payload would quantize the storage bytes rather than the weights"
        )


def strip_quantization_config(config_path: str | Path) -> bool:
    """Remove a source `quantization_config` from an emitted `config.json`.

    The master holds bf16 ternary latents. A `quant_method: fp8` inherited from the
    source would tell transformers to dequantize them a second time, against scales
    that are no longer there.

    Args:
        config_path: The output `config.json`.

    Returns:
        Whether a config was stripped.
    """
    path = Path(config_path)
    if not path.is_file():
        return False
    config = json.loads(path.read_text())
    removed = config.pop("quantization_config", None)
    if removed is None:
        return False
    method = removed.get("quant_method") if isinstance(removed, dict) else None
    LOG.warning(
        f"ternary: dropped the source quantization_config (quant_method={method!r}) "
        f"from {path.name}; the master holds bf16 latents, not a {method} payload"
    )
    path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n")
    return True


__all__ = [
    "ACT_SCALE_SUFFIXES",
    "FP8_DTYPES",
    "OUTPUT_DTYPE",
    "WEIGHT_SCALE_SUFFIXES",
    "assert_no_orphan_fp8",
    "companion_keys",
    "dequantize",
    "dequantized_items",
    "is_fp8",
    "scale_keys",
    "strip_quantization_config",
]
