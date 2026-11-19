"""CLI to export a trained model to a deployment format."""
from pathlib import Path
from typing import Any, Union

from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from axolotl.utils.gguf import export_gguf, export_lora_gguf, is_adapter_dir
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.export import ExportConfig

LOG = get_logger(__name__)


def resolve_model_dir(
    cfg: DictDefault, model_dir: str | None = None, lora: bool | None = None
) -> tuple[Path, bool]:
    """Pick the checkpoint to export: an explicit dir, else the merged/trained output.

    Returns:
        (path, is_lora) — *is_lora* is True when the resolved directory holds a
        PEFT adapter that should be exported via convert_lora_to_gguf.py.
    """
    if model_dir:
        path = Path(model_dir)
        return path, bool(lora) if lora is not None else is_adapter_dir(path)

    merged = Path(cfg.output_dir) / "merged"
    if merged.is_dir():
        if lora:
            # User explicitly wants the adapter even though merged/ exists.
            return Path(cfg.output_dir), True
        return merged, False

    output_dir = Path(cfg.output_dir)
    if cfg.adapter:
        if lora is not None and not lora:
            raise ValueError(
                f"{cfg.output_dir} holds a {cfg.adapter} adapter, not a full model. Run "
                f"`axolotl merge-lora` first, or pass --model-dir."
            )
        # Auto-detect: adapter_config.json present means this is an adapter dir.
        return output_dir, True

    return output_dir, False


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

    model_dir, is_lora = resolve_model_dir(
        cfg, cli_args.get("model_dir"), cli_args.get("lora")
    )

    # If export.lora is set but the CLI --lora flag wasn't, use the config value.
    # This handles the case where lora=True is set in the export config block
    # but the model dir wasn't explicitly provided.
    if not is_lora and export_cfg.lora and cli_args.get("model_dir") is None:
        is_lora = is_adapter_dir(model_dir)
    run_dir = Path(cfg.output_dir)
    outfile = export_cfg.outfile or str(
        run_dir / export_cfg.format / f"{run_dir.name}-{{ftype}}.gguf"
    )

    if is_lora:
        # For LoRA exports, {ftype} is replaced by the outtype directly.
        lora_outfile = outfile.replace("{ftype}", export_cfg.outtype)
        outputs = [
            export_lora_gguf(
                model_dir,
                lora_outfile,
                outtype=export_cfg.outtype,
                base_model=cfg.base_model,
                llama_cpp_dir=export_cfg.llama_cpp_dir,
            )
        ]
    else:
        outputs = export_gguf(
            model_dir,
            outfile,
            outtype=export_cfg.outtype,
            quantize=export_cfg.quantize,
            llama_cpp_dir=export_cfg.llama_cpp_dir,
        )
    LOG.info(f"Exported {len(outputs)} file(s) to {Path(outfile).parent}.")

    return outputs