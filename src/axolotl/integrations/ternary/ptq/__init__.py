"""Analytic initialization of the ternary latents — the `ternary.init` modes.

`absmean` is a no-op: the swap keeps the FP weights and the quantizer rounds them on
the first forward. The fitting modes instead rewrite each latent so the *quantized*
weight is already the best ternary approximation of the original tensor when healing
starts, which is where a PTQ solver buys its head start.

Only symmetric fits (offset μ ≡ 0) are supported: an asymmetric grid needs a
per-row offset that no deployment format in `export/` can carry.

What lands in the latent is the fit's own reconstruction, `codes · f16(scale)` — the
same `{-s, 0, +s}` tensor a bake writes — so a fitted module is indistinguishable
from a reloaded master and is flagged `baked` accordingly. That is what keeps the
init idempotent: refitting a tensor that already holds the quantizer's output would
fit the quantizer, not the model.

The reconstruction is a lossy carrier for the scale: `absmean(codes · s)` is
`s · nonzero_fraction`, not `s`. A fitting init therefore only works in a scale mode
that *stores* its scale (`learnable`, `learnable_row`, `dual`), where
`refresh_scale_from_weight` parks the solved value in a parameter; the schema refuses
the statistic modes, which would silently fall back to the shrunken absmean on the
first optimizer step.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from ..args import resolve_ternary_config
from ..modules import TernaryLinear
from .ternary_fit import dequantize_fit, fit_ternary

if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ..swap import SwapEntry, SwapManifest

LOG = get_logger(__name__)


def initialize_model_latents(
    model: "PreTrainedModel", manifest: "SwapManifest", cfg: DictDefault
) -> None:
    """Rewrite the latent weight of every swapped module with the configured initializer.

    Runs inside `swap.convert_model`, right after the modules are replaced and before
    any parallelism wrapping, so the weights are still whole and on one device. It
    also runs on the reload path, so modules whose weights are already baked
    (`TernaryLinear.baked`) must be left untouched: refitting a tensor that is already
    `{-s, 0, +s}` fits the quantizer's own output, not the original model.

    A rank holding meta parameters has nothing to solve — under FSDP2's
    `cpu_ram_efficient_loading` only local rank 0 materializes the checkpoint and its
    weights are broadcast at wrap time, so rank 0's fit is what every rank ends up
    with. Such a rank skips the whole pass rather than fitting part of a model.

    Args:
        model: The converted model, its `TernaryLinear` modules already in place.
        manifest: The swap manifest for `model` (which modules were swapped, the
            scale mode each was built with, and the init mode being applied).
        cfg: The axolotl config; `ternary.*` is read through `resolve_ternary_config`
            and `ternary.init_calibration` supplies the calibration corpus.

    Raises:
        NotImplementedError: For an init mode the plugin accepts but cannot run yet.
    """
    ternary_cfg = resolve_ternary_config(cfg)
    if ternary_cfg.init == "absmean":
        return
    if ternary_cfg.init not in ("ternary_fit", "ternary_fit_calibrated"):
        raise NotImplementedError(
            f"ternary.init: {ternary_cfg.init} is accepted by the schema but not "
            "implemented yet; use `init: absmean`"
        )
    if has_meta_latents(model, manifest):
        LOG.warning(
            f"ternary: skipping init: {ternary_cfg.init} on this rank; its latents are "
            "on the meta device (FSDP2 cpu_ram_efficient_loading materializes the "
            "checkpoint on local rank 0 only and broadcasts it, so rank 0's fit is the "
            "one this rank will train from)"
        )
        return
    if ternary_cfg.init == "ternary_fit_calibrated":
        from .calibrate import calibrate_model_latents

        calibrate_model_latents(model, manifest, cfg)
        return

    fitted = 0
    for entry, module in iter_swapped(model, manifest):
        if module.is_baked():
            continue
        fit_module_latent(module, entry)
        fitted += 1
    LOG.info(
        f"ternary: fitted {fitted} of {len(manifest.entries)} latents with "
        f"init: {ternary_cfg.init}"
    )


def iter_swapped(
    model: "PreTrainedModel", manifest: "SwapManifest"
) -> Iterator[tuple["SwapEntry", TernaryLinear]]:
    """Yield `(entry, module)` for every manifest entry that resolves to a ternary module."""
    for entry in manifest.entries:
        module = model.get_submodule(entry.name)
        if isinstance(module, TernaryLinear):
            yield entry, module


def has_meta_latents(model: "PreTrainedModel", manifest: "SwapManifest") -> bool:
    """Whether any swapped module on this rank still holds a meta-device latent."""
    return any(
        module.weight.device.type == "meta"
        for _, module in iter_swapped(model, manifest)
    )


def fit_module_latent(module: TernaryLinear, entry: "SwapEntry") -> None:
    """Run the data-free fit for `entry`'s scale mode and write it into `module`."""
    codes, scale = fit_ternary(
        module.weight.detach(),
        scale_mode=entry.weight_scale,  # type: ignore[arg-type]
        group_size=entry.group_size,
    )
    write_latent(module, codes, scale)


def write_latent(
    module: TernaryLinear, codes: torch.Tensor, scale: torch.Tensor
) -> None:
    """Write a fit into `module`: the reconstruction as latent, the scale as parameter.

    Args:
        module: The swapped module to initialize.
        codes: int8 codes from `fit_ternary`.
        scale: fp32 scales from `fit_ternary`, in that function's layout.
    """
    with torch.no_grad():
        module.weight.copy_(dequantize_fit(codes, scale, module.weight.dtype))
    module._detect_baked()  # pylint: disable=protected-access
    # the reconstruction is on the fit's own grid, so the module re-derives exactly
    # the fit's scales from it — in every layout, including the dual pair
    module.refresh_scale_from_weight()


__all__ = [
    "fit_module_latent",
    "has_meta_latents",
    "initialize_model_latents",
    "iter_swapped",
    "write_latent",
]
