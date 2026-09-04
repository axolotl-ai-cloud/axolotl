"""CLI to export a trained model to a deployment format."""

from pathlib import Path
from typing import Any, Union

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.utils.gguf import export_gguf
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.export import ExportConfig

LOG = get_logger(__name__)


def resolve_model_dir(cfg: DictDefault, model_dir: str | None = None) -> Path:
    """Pick the checkpoint to export: an explicit dir, else the merged/trained output."""
    if model_dir:
        return Path(model_dir)

    merged = Path(cfg.output_dir) / "merged"
    if merged.is_dir():
        return merged
    if cfg.adapter:
        raise ValueError(
            f"{cfg.output_dir} holds a {cfg.adapter} adapter, not a full model. Run "
            f"`axolotl merge-lora` first, or pass --model-dir."
        )

    return Path(cfg.output_dir)


def do_export(config: Union[Path, str], cli_args: dict[str, Any]) -> list[Path]:
    """
    Exports a trained model to a deployment format.

    Args:
        config: The path to the config file.
        cli_args: Additional command-line arguments, overriding the config's `export` block.

    Returns:
        Paths of the written files.
    """
    cfg = load_cfg(str(config))

    overrides = {
        key: value
        for key, value in cli_args.items()
        if key in ExportConfig.model_fields
    }
    export_cfg = ExportConfig(**{**(cfg.export or {}), **overrides})

    model_dir = resolve_model_dir(cfg, cli_args.get("model_dir"))
    output_dir = Path(export_cfg.output_dir or Path(cfg.output_dir) / export_cfg.format)

    outputs = export_gguf(
        model_dir,
        output_dir,
        name=Path(cfg.output_dir).name,
        outtype=export_cfg.outtype,
        quantize=export_cfg.quantize,
        llama_cpp_dir=export_cfg.llama_cpp_dir,
    )
    LOG.info(f"Exported {len(outputs)} file(s) to {output_dir}.")

    return outputs
