"""CLI for SAR (Subspace-Aligned Rewiring) post-processing of trained models."""

from pathlib import Path

import click
import yaml

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


@click.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--base-model",
    default=None,
    help="Spectral reference model (local dir or HF id). Default: `sar.base_model`, then top-level `base_model`.",
)
@click.option(
    "--trained-model",
    default=None,
    help="Post-trained model providing the weight delta. Default: `sar.trained_model`, then top-level `output_dir`.",
)
@click.option(
    "--merge-target",
    default=None,
    help="Expert model to merge the projected delta into instead of the base model.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Where to write the projected model. Default: `sar.output_dir`, then `{output_dir}/sar`.",
)
@click.option(
    "--rank-ratio",
    multiple=True,
    type=float,
    help="Rank ratio in (0, 1]; repeat for a sweep with one output per ratio.",
)
@click.option(
    "--scale",
    default=None,
    type=float,
    help="Coefficient applied to the projected delta.",
)
@click.option(
    "--save-dtype",
    default=None,
    type=click.Choice(["float16", "bfloat16", "float32"]),
    help="Dtype for saved tensors.",
)
@click.option(
    "--svd-device",
    default=None,
    type=click.Choice(["auto", "cuda", "cpu"]),
    help="Device for SVD computation.",
)
@click.option(
    "--save-rewiring-matrix",
    is_flag=True,
    default=None,
    help="Also save the compact per-layer rewiring matrices under `{output_dir}/rewiring/`.",
)
def sar(
    config: str,
    base_model: str | None,
    trained_model: str | None,
    merge_target: str | None,
    output_dir: str | None,
    rank_ratio: tuple[float, ...],
    scale: float | None,
    save_dtype: str | None,
    svd_device: str | None,
    save_rewiring_matrix: bool | None,
):
    """Project a trained model's weight delta onto the base model's spectral subspace.

    Settings resolve as: CLI options > the config's `sar:` block > defaults derived
    from the config (base <- `base_model`, trained <- `output_dir`, output <-
    `{output_dir}/sar`). The merged settings are always validated against the SAR
    schema before running, even for configs that fail full training validation
    (e.g. a standalone config without `datasets:`).
    """
    from pydantic import ValidationError

    from axolotl.cli.config import load_cfg
    from axolotl.integrations.sar.args import SARConfig

    with open(config, encoding="utf-8") as file:
        raw_config = yaml.safe_load(file) or {}
    if not isinstance(raw_config, dict):
        raise click.UsageError("the config file must contain a YAML mapping")

    try:
        cfg = load_cfg(config)
    except ValueError as err:
        LOG.warning(
            "Config failed full validation (%s); reading SAR settings from the raw YAML",
            err,
        )
        cfg = DictDefault(raw_config)

    raw_sar = cfg.get("sar")
    sar_present = raw_sar is not None or "sar" in raw_config
    if raw_sar is None:
        raw_sar = raw_config.get("sar") or {}
    if not isinstance(raw_sar, dict):
        raise click.UsageError("`sar:` in the config must be a mapping")

    overrides = {
        "base_model": base_model,
        "trained_model": trained_model,
        "merge_target": merge_target,
        "output_dir": output_dir,
        "rank_ratio": list(rank_ratio) if rank_ratio else None,
        "scale": scale,
        "save_dtype": save_dtype,
        "svd_device": svd_device,
        "save_rewiring_matrix": save_rewiring_matrix,
    }
    applied = {key: value for key, value in overrides.items() if value is not None}
    if not sar_present and not applied:
        raise click.UsageError(
            "no `sar:` block found in the config and no CLI overrides were given; "
            "add a `sar:` block or pass overrides such as --trained-model / --rank-ratio"
        )

    try:
        sar_cfg = SARConfig.model_validate({**raw_sar, **applied})
    except ValidationError as err:
        raise click.UsageError(f"invalid SAR settings:\n{err}") from err

    resolved_base = sar_cfg.base_model or cfg.get("base_model")
    resolved_trained = sar_cfg.trained_model or cfg.get("output_dir")
    resolved_output = sar_cfg.output_dir
    if not resolved_output and cfg.get("output_dir"):
        resolved_output = str(Path(cfg["output_dir"]) / "sar")

    missing = [
        flag
        for flag, value in (
            ("--base-model", resolved_base),
            ("--trained-model", resolved_trained),
            ("--output-dir", resolved_output),
        )
        if not value
    ]
    if missing:
        raise click.UsageError(
            f"could not resolve {', '.join(missing)}; set the corresponding `sar:` "
            "fields, top-level `base_model`/`output_dir`, or pass the option(s)"
        )

    rank_ratios = sar_cfg.rank_ratio
    if not isinstance(rank_ratios, list):
        rank_ratios = [rank_ratios]

    base_revision = sar_cfg.base_model_revision
    if base_revision is None and not sar_cfg.base_model:
        base_revision = cfg.get("revision_of_model")

    from axolotl.integrations.sar.core import run_sar

    result = run_sar(
        str(resolved_base),
        str(resolved_trained),
        str(resolved_output),
        merge_target=sar_cfg.merge_target,
        rank_ratios=rank_ratios,
        delta_rank_ratio=sar_cfg.delta_rank_ratio,
        projection=sar_cfg.projection,
        rewiring=sar_cfg.rewiring,
        scale=sar_cfg.scale,
        target_modules=sar_cfg.target_modules,
        exclude_modules=sar_cfg.exclude_modules,
        svd_device=sar_cfg.svd_device,
        save_dtype=sar_cfg.save_dtype,
        save_rewiring_matrix=sar_cfg.save_rewiring_matrix,
        base_model_revision=base_revision,
        trained_model_revision=sar_cfg.trained_model_revision,
        merge_target_revision=sar_cfg.merge_target_revision,
    )

    for ratio, ratio_output_dir in sorted(result.outputs.items()):
        LOG.info("SAR output (rank_ratio=%s): %s", ratio, ratio_output_dir)
