"""Bridge to axolotl's `use_onebitllms` finetuning path.

`onebitllms`' `BitNetLinear` is a QAT layer, not a loader: it keeps a full-precision
latent and re-derives its own b1.58 grid on every forward,

    scale = 1 / max(mean|w|, 1e-5),   codes = round(w · scale).clamp(±1),
    w_eff = codes / scale

so what a checkpoint stores is a *latent*, not a grid. Handing it our baked master
therefore does not preserve the master's function. The master holds `code · s`, whose
absmean is `s · (1 - z)` for a zero fraction `z`, so their scale comes out
`1 / (s · (1 - z))`: the codes survive, but every weight dequantizes to
`code · s · (1 - z)` — a uniform ~0.67x shrink of every linear at the ~33% density a
healed model settles at, which destroys the function outright.

The fix is closed-form. Store the latent *inflated* by the density:

    w = code · s / (1 - z)

Then `mean|w| = s` exactly, so their scale is `1 / s`; their codes are
`round(code / (1 - z)).clamp(±1)`, and since `1 / (1 - z) >= 1` for every `z` in
`[0, 1)` that is `code` for all three states; and their dequant is `code · s`. The
trained function is an exact fixed point of their quantizer.

Ties never arise, which matters because the package rounds two different ways: the
reference path in `utils/quantization_utils.py` uses `torch.round` (half to even) and
the Triton kernel uses `floor(x + 0.5)` (half away from zero). A tie needs
`|code| / (1 - z) = 0.5`, impossible when the ratio is either 0 or at least 1, so
both paths agree on every weight this format writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from axolotl.utils.logging import get_logger

from ..swap import SwapManifest
from . import bake

LOG = get_logger(__name__)

RECORD_FILENAME: str = "ternary_onebitllms.json"
VARIANT: str = "onebitllms_bf16"

# their floor on the absmean, and the dtype the checkpoint is written in
ABSMEAN_EPS: float = 1e-5
EXPORT_DTYPE: torch.dtype = torch.bfloat16

# scale modes whose grid `BitNetLinear` can re-derive: one scale, one plane, no offset
SUPPORTED_SCALE_MODES: frozenset[str] = frozenset({"absmean", "learnable"})

# `replace_linear_with_bitnet_linear` swaps every `nn.Linear` but `lm_head`, so a
# module we deliberately kept full precision would be quantized anyway
ALWAYS_KEPT_BY_ONEBITLLMS: tuple[str, ...] = ("lm_head", "embed_tokens", "norm")


def inflation_factor(codes: torch.Tensor) -> tuple[float, float]:
    """Return `(zero_fraction, 1 / (1 - zero_fraction))` for a tensor's codes.

    An all-zero tensor has no density to invert; it inflates by 1.0 and stays zero,
    which their quantizer maps to zero as well.
    """
    total = codes.numel()
    if not total:
        return 0.0, 1.0
    zeros = int((codes == 0).sum())
    if zeros == total:
        return 1.0, 1.0
    zero_fraction = zeros / total
    return zero_fraction, 1.0 / (1.0 - zero_fraction)


def inflate_master_weight(
    weight: torch.Tensor, scale: torch.Tensor, codes: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    """Return `(code · s / (1 - z), zero_fraction, factor)` in the export dtype."""
    zero_fraction, factor = inflation_factor(codes)
    values = codes.to(torch.float32) * scale.to(torch.float32).reshape(()) * factor
    return values.to(EXPORT_DTYPE), zero_fraction, factor


def onebitllms_quantize(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Clean-room reference for `BitNetLinear`'s weight quantizer.

    Reimplements `scale = 1 / max(mean|w|, 1e-5)`, `codes = round(w · scale).clamp(±1)`
    from the documented behaviour, in fp32. The rounding mode is left as
    `torch.round`; see the module docstring for why the package's two rounding paths
    cannot disagree on anything this format writes.

    Args:
        weight: A stored latent, any floating dtype.

    Returns:
        `(codes, scale)` — int8 codes shaped like `weight`, and the 0-dim fp32
        *dequantization* scale `mean|w|` (their `1 / scale`).
    """
    values = weight.detach().to(torch.float32)
    absmean = values.abs().mean().clamp_min(ABSMEAN_EPS)
    codes = torch.round(values / absmean).clamp_(-1.0, 1.0)
    return codes.to(torch.int8), absmean


def export_onebitllms(
    master_dir: str | Path, output_dir: str | Path, manifest: SwapManifest
) -> Path:
    """Write a `use_onebitllms`-ready checkpoint beside the master and return its dir.

    Every swapped tensor is stored inflated by its density so `BitNetLinear` re-derives
    the master's own grid; everything else — embeddings, head, norms, config,
    tokenizer — is copied through unchanged. The per-tensor factors land in
    `RECORD_FILENAME` beside the shards.

    Raises:
        ValueError: If the manifest uses a scale mode `BitNetLinear` cannot re-derive,
            inserts sub-norms, or the master still holds latent weights.
        KeyError: If a manifest entry has no matching weight in the master.
    """
    master_dir = Path(master_dir)
    output = Path(output_dir)
    _reject_unsupported(manifest)
    if output.resolve() == master_dir.resolve():
        raise ValueError(
            "onebitllms_bf16 must be written beside the master, not over it; its "
            "latents are inflated and a ternary reload would read them as a "
            "different grid"
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
    factors: dict[str, dict[str, float]] = {}
    total_size = 0
    for path in shards:
        tensors, metadata = bake.load_shard(path)
        written: dict[str, torch.Tensor] = {}
        for key, tensor in tensors.items():
            entry = entries.get(key)
            if entry is None:
                written[key] = tensor
                continue
            try:
                codes, scale = bake.derive_codes_and_scale(
                    tensor, entry.group_size, entry.weight_scale
                )
            except ValueError as exc:
                raise ValueError(
                    f"{entry.name} still holds latent weights ({exc}); bake the master "
                    "before inflating it"
                ) from exc
            inflated, zero_fraction, factor = inflate_master_weight(
                tensor, scale, codes
            )
            written[key] = inflated
            factors[entry.name] = {
                "zero_fraction": zero_fraction,
                "inflation": factor,
                "scale": float(scale.reshape(())),
            }
            remaining.discard(key)
        shard_name = Path(path).name
        for key, tensor in written.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()
        bake.save_shard(written, output / shard_name, metadata)
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
    _write_record(output, factors)
    _warn_about_extra_quantization(manifest)
    manifest.save(output)
    bake.write_quantizer_metadata(output, manifest, artifact_format=VARIANT)
    LOG.info(
        f"ternary: wrote {len(factors)} density-inflated tensors to {output} for "
        "`use_onebitllms: true`"
    )
    return output


def _write_record(output_dir: Path, factors: dict[str, dict[str, float]]) -> None:
    path = Path(output_dir) / RECORD_FILENAME
    path.write_text(
        json.dumps(
            {
                "variant": VARIANT,
                "quantizer": "b1.58 absmean, scale = 1 / max(mean|w|, 1e-5)",
                "latent": "code * scale / (1 - zero_fraction)",
                "tensors": factors,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _reject_unsupported(manifest: SwapManifest) -> None:
    if manifest.weight_scale not in SUPPORTED_SCALE_MODES or manifest.group_size:
        raise ValueError(
            f"onebitllms_bf16 cannot represent ternary.weight_scale: "
            f"{manifest.weight_scale} (group_size={manifest.group_size}). BitNetLinear "
            "re-derives one absmean scale for the whole tensor on every forward, so a "
            "per-group, per-row or two-plane grid has nowhere to live and would be "
            f"replaced by a single scale. Use {sorted(SUPPORTED_SCALE_MODES)}"
        )
    if manifest.subln:
        raise ValueError(
            "onebitllms_bf16 cannot carry ternary.subln: `replace_linear_with_bitnet_"
            "linear` swaps stock `nn.Linear` modules and has no slot for the sub-norm "
            "the swap inserted, so the exported model would drop it silently"
        )


def _warn_about_extra_quantization(manifest: SwapManifest) -> None:
    """Flag Linears we kept full precision that onebitllms will quantize anyway."""
    surprises = [
        name
        for name in manifest.kept_fp
        if not any(keep in name for keep in ALWAYS_KEPT_BY_ONEBITLLMS)
    ]
    if surprises:
        LOG.warning(
            f"ternary: {len(surprises)} Linear(s) kept full precision by this run "
            f"({', '.join(sorted(surprises)[:3])}...) will still be quantized by "
            "`replace_linear_with_bitnet_linear`, which swaps every nn.Linear but "
            "lm_head; the finetune will not match the exported function on those"
        )


__all__ = [
    "ABSMEAN_EPS",
    "EXPORT_DTYPE",
    "RECORD_FILENAME",
    "SUPPORTED_SCALE_MODES",
    "VARIANT",
    "export_onebitllms",
    "inflate_master_weight",
    "inflation_factor",
    "onebitllms_quantize",
]
