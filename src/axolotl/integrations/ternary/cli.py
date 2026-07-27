"""axolotl CLI for ternary conversion artifacts."""

from __future__ import annotations

from typing import cast, get_args

import click

from .args import ExportFormat

PLUGIN_PATH = "axolotl.integrations.ternary.TernaryPlugin"


@click.group(name="ternary")
def ternary():
    """convert and export ternary (1.58-bit) models"""


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
    from axolotl.cli.config import load_cfg

    from .export import run_export

    cfg = load_cfg(config)
    if PLUGIN_PATH not in (cfg.plugins or []):
        # without the plugin the `ternary:` block never reaches the validated config
        raise click.ClickException(
            f"{config} does not enable the ternary plugin; add `{PLUGIN_PATH}` to `plugins:`"
        )

    requested = cast("list[ExportFormat]", list(formats))
    artifacts = run_export(cfg, formats=requested or None, output_dir=output_dir)
    for fmt, path in artifacts.items():
        click.echo(f"{fmt}: {path}")
