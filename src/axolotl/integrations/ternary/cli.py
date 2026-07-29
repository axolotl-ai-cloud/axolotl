"""axolotl CLI for ternary conversion artifacts."""

from __future__ import annotations

import json
from typing import cast, get_args

import click

from .args import ExportFormat
from .ptq.damage import DEFAULT_SEQUENCE_LEN, DEFAULT_SEQUENCES

PLUGIN_PATH = "axolotl.integrations.ternary.TernaryPlugin"


@click.group(name="ternary")
def ternary():
    """convert and export ternary (1.58-bit) models"""


def _load_plugin_cfg(config: str):
    from axolotl.cli.config import load_cfg

    cfg = load_cfg(config)
    if PLUGIN_PATH not in (cfg.plugins or []):
        # without the plugin the `ternary:` block never reaches the validated config
        raise click.ClickException(
            f"{config} does not enable the ternary plugin; add `{PLUGIN_PATH}` to `plugins:`"
        )
    return cfg


@ternary.command(name="export")
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--format",
    "formats",
    multiple=True,
    type=click.Choice(list(get_args(ExportFormat))),
    help="Export format; repeatable. Defaults to `ternary.export.formats`.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=str),
    help="Where to write the artifacts. Defaults to the config's `output_dir`.",
)
def export(config: str, formats: tuple[str, ...], output_dir: str | None) -> None:
    """pack a trained ternary master into deployment formats"""
    from .export import run_export

    cfg = _load_plugin_cfg(config)
    requested = cast("list[ExportFormat]", list(formats))
    artifacts = run_export(cfg, formats=requested or None, output_dir=output_dir)
    for fmt, path in artifacts.items():
        click.echo(f"{fmt}: {path}")


@ternary.command(name="fit-stream")
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--input-dir",
    default=None,
    type=click.Path(path_type=str),
    help="Checkpoint to fit. Defaults to the config's `base_model`.",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(path_type=str),
    help="Where the fitted checkpoint is written. Defaults to the config's `output_dir`.",
)
@click.option(
    "--device",
    default="cpu",
    show_default=True,
    type=str,
    help="Device the per-tensor solve runs on.",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    show_default=True,
    help="Skip shards a previous pass already finished.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the report as JSON instead of a summary line.",
)
def fit_stream_command(
    config: str,
    input_dir: str | None,
    output_dir: str | None,
    device: str,
    resume: bool,
    as_json: bool,
) -> None:
    """fit ternary codes shard by shard, without loading the model"""
    from .args import assert_stream_supported, resolve_ternary_config
    from .ptq.stream import stream_fit, summarize_families

    cfg = _load_plugin_cfg(config)
    try:
        assert_stream_supported(resolve_ternary_config(cfg))
    except ValueError as error:
        raise click.ClickException(str(error)) from error

    report = stream_fit(
        input_dir or cfg.base_model,
        output_dir or cfg.output_dir,
        cfg,
        device=device,
        resume=resume,
    )
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return
    click.echo(
        f"{report.output_dir}: {report.tensors_fitted} tensors fitted over "
        f"{report.shards_fitted}/{report.shards} shards "
        f"({report.shards_skipped} skipped)"
    )
    click.echo(
        f"{report.weights_fitted} weights at {report.bits_per_weight:.3f} bits per "
        f"weight, {report.zero_fraction:.1%} zero, "
        f"{report.weights_per_second / 1e6:.1f}M weights/s"
    )
    if report.families:
        click.echo(f"per family: {summarize_families(report.families)}")


@ternary.command(name="damage-map")
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--num-sequences",
    default=DEFAULT_SEQUENCES,
    show_default=True,
    type=click.IntRange(min=1),
    help="Sequences evaluated per pass.",
)
@click.option(
    "--sequence-len",
    default=DEFAULT_SEQUENCE_LEN,
    show_default=True,
    type=click.IntRange(min=2),
    help="Tokens per evaluation sequence.",
)
@click.option(
    "--dataset",
    default=None,
    type=str,
    help="Dataset to evaluate on. Defaults to the config's training dataset.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the report as JSON instead of a table.",
)
def damage_map_command(
    config: str,
    num_sequences: int,
    sequence_len: int,
    dataset: str | None,
    as_json: bool,
) -> None:
    """measure what each part of the quantizer costs, without training"""
    from .ptq.damage import damage_map

    cfg = _load_plugin_cfg(config)
    report = damage_map(
        cfg,
        num_sequences=num_sequences,
        sequence_len=sequence_len,
        dataset=dataset,
    )
    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return
    click.echo(report.to_table())
