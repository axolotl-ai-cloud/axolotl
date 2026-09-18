"""CLI to export a trained model to a deployment format."""

from pathlib import Path
from typing import Any, Union

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.utils.gguf import ADAPTER_CONFIG, export_gguf, export_lora_gguf
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.export import ExportConfig, validate_lora_export

LOG = get_logger(__name__)


def is_adapter_dir(model_dir: Path) -> bool:
    """Whether a directory holds a PEFT adapter rather than a full checkpoint."""
    return (model_dir / ADAPTER_CONFIG).is_file()


def resolve_model_dir(
    cfg: DictDefault, model_dir: str | None = None, lora: bool | None = None
) -> Path:
    """Pick the checkpoint to export: an explicit dir, else the merged/trained output."""
    if model_dir:
        return Path(model_dir)

    output_dir = Path(cfg.output_dir)
    if lora:
        if not is_adapter_dir(output_dir):
            raise ValueError(
                f"`export.lora` is set but {output_dir} holds no {ADAPTER_CONFIG}."
            )
        return output_dir

    merged = output_dir / "merged"
    if merged.is_dir():
        return merged
    if lora is None and is_adapter_dir(output_dir):
        return output_dir
    if cfg.adapter:
        raise ValueError(
            f"{cfg.output_dir} holds a {cfg.adapter} adapter, not a full model. Run "
            f"`axolotl merge-lora` first, or pass --model-dir."
        )

    return output_dir


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

    model_dir = resolve_model_dir(cfg, cli_args.get("model_dir"), export_cfg.lora)
    is_lora = (
        export_cfg.lora if export_cfg.lora is not None else is_adapter_dir(model_dir)
    )
    outtype = export_cfg.resolved_outtype(is_lora)

    run_dir = Path(cfg.output_dir)
    stem = f"{run_dir.name}-lora" if is_lora else run_dir.name
    outfile = export_cfg.outfile or str(
        run_dir / export_cfg.format / f"{stem}-{{ftype}}.gguf"
    )

    if is_lora:
        validate_lora_export(outtype, export_cfg.quantize)
        outputs = export_lora_gguf(
            model_dir,
            outfile,
            outtype=outtype,
            base_model=cfg.base_model,
            llama_cpp_dir=export_cfg.llama_cpp_dir,
        )
    else:
        outputs = export_gguf(
            model_dir,
            outfile,
            outtype=outtype,
            quantize=export_cfg.quantize,
            llama_cpp_dir=export_cfg.llama_cpp_dir,
        )
    LOG.info(f"Exported {len(outputs)} file(s) to {Path(outfile).parent}.")

    return outputs
