"""Click CLI definitions for various axolotl commands."""

import os
import subprocess  # nosec B404
from pathlib import Path
from typing import Literal, Optional

import click
from dotenv import load_dotenv

import axolotl
from axolotl.cli.args import (
    EvaluateCliArgs,
    PreprocessCliArgs,
    QuantizeCliArgs,
    TrainerCliArgs,
    VllmServeCliArgs,
)
from axolotl.cli.art import print_axolotl_text_art
from axolotl.cli.config_options import AXOLOTL_CONFIG_CLI_OPTIONS
from axolotl.cli.launchers.resolve import resolve_launch
from axolotl.cli.plugins import PluginCommandGroup
from axolotl.cli.utils import (
    add_options_from_config_options,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
    filter_none_kwargs,
    generate_config_files,
    launch_training,
)
from axolotl.utils import set_misc_env, set_pytorch_cuda_alloc_conf
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

LAUNCHER_COMMAND_MAPPING = {
    "accelerate": ["accelerate", "launch"],
    "torchrun": ["torchrun"],
}


@click.group(cls=PluginCommandGroup)
@click.version_option(version=axolotl.__version__, prog_name="axolotl")
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""
    print_axolotl_text_art()
    load_dotenv()
    set_pytorch_cuda_alloc_conf()
    set_misc_env()


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(PreprocessCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
def preprocess(config: str, cloud: Optional[str] = None, **kwargs):
    """
    Preprocess datasets before training.

    Args:
        config: Path to `axolotl` config YAML file.
        cloud: Path to a cloud accelerator configuration file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """

    if cloud:
        from axolotl.cli.cloud import do_cli_preprocess

        do_cli_preprocess(cloud_config=cloud, config=config)
    else:
        from axolotl.cli.preprocess import do_cli

        do_cli(config=config, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "ray", "python"]),
    default=None,
    show_default="accelerate",
    help="Launcher to use for multi-GPU training",
)
@click.option(
    "--runtime",
    default=None,
    type=click.Path(exists=True, path_type=str),
    help="Path to a runtime/cluster config YAML (launcher + per-launcher settings)",
)
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@click.option(
    "--sweep",
    type=click.Path(exists=True, path_type=str),
    help="YAML config for sweeping hyperparameters",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
@click.pass_context
def train(
    ctx: click.Context,
    config: str,
    launcher: Literal["accelerate", "torchrun", "ray", "python"] | None = None,
    runtime: str | None = None,
    cloud: str | None = None,
    sweep: str | None = None,
    **kwargs,
):
    """
    Train or fine-tune a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU training ("accelerate", "torchrun",
            "ray", or "python"). Defaults to "accelerate" unless a --runtime file or
            the training config selects otherwise.
        runtime: Path to a runtime/cluster config YAML file.
        cloud: Path to a cloud accelerator configuration file
        sweep: Path to YAML config for sweeping hyperparameters.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    resolved = resolve_launch(config, runtime, launcher, kwargs, launcher_args)

    # Process each configuration
    for cfg_file, is_group in generate_config_files(config, sweep):
        try:
            use_exec = is_group is not True
            launch_training(
                cfg_file,
                resolved.launcher,
                cloud,
                kwargs,
                resolved.launcher_args,
                use_exec,
                env=resolved.env,
                runtime=resolved.runtime,
            )
        except subprocess.CalledProcessError as exc:
            LOG.error(f"Failed to train/fine-tune config '{cfg_file}': {exc}")
            if not sweep:
                raise exc
        finally:
            # Only delete temp files, not the original config
            if cfg_file != config:
                os.unlink(cfg_file)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default=None,
    show_default="accelerate",
    help="Launcher to use for multi-GPU evaluation",
)
@click.option(
    "--runtime",
    default=None,
    type=click.Path(exists=True, path_type=str),
    help="Path to a runtime/cluster config YAML (launcher + per-launcher settings)",
)
@add_options_from_dataclass(EvaluateCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
@click.pass_context
def evaluate(
    ctx: click.Context,
    config: str,
    launcher: Literal["accelerate", "torchrun", "python"] | None = None,
    runtime: str | None = None,
    **kwargs,
):
    """
    Evaluate a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU evaluation ("accelerate", "torchrun", or "python").
        runtime: Path to a runtime/cluster config YAML file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    resolved = resolve_launch(
        config, runtime, launcher, kwargs, launcher_args, supports_ray=False
    )

    if resolved.launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[resolved.launcher]
            + resolved.launcher_args
            + ["-m", "axolotl.cli.evaluate"]
        )
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        run_kwargs: dict = {"check": True}
        if resolved.env:
            run_kwargs["env"] = {**os.environ, **resolved.env}
        subprocess.run(cmd, **run_kwargs)  # nosec B603
    else:
        from axolotl.cli.evaluate import do_cli

        do_cli(config=config, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU inference",
)
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@click.option(
    "--chat", is_flag=True, help="Launch interactive multi-turn chat interface"
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
@click.pass_context
def inference(
    ctx: click.Context, config: str, launcher: str, gradio: bool, chat: bool, **kwargs
):
    """
    Run inference with a trained model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU inference ("accelerate", "torchrun", or "python").
        gradio: Whether to use Gradio browser interface or command line for inference.
        chat: Whether to use the interactive multi-turn chat interface.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    if gradio and chat:
        raise click.UsageError("--gradio and --chat are mutually exclusive.")

    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.inference"]
        )
        if config:
            base_cmd.append(config)
        if gradio:
            base_cmd.append("--gradio")
        if chat:
            base_cmd.append("--chat")
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, gradio=gradio, chat=chat, **kwargs)


@cli.command(
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True}
)
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for weight merging",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
@click.pass_context
def merge_sharded_fsdp_weights(
    ctx: click.Context, config: str, launcher: str, **kwargs
):
    """
    Merge sharded FSDP model weights.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for weight merging ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.merge_sharded_fsdp_weights"]
        )
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.merge_sharded_fsdp_weights import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--dequant",
    is_flag=True,
    default=False,
    help="Dequantize a quantized base to bf16 in the merged checkpoint instead of "
    "re-quantizing to the base's format.",
)
@click.option(
    "--override-quantizer",
    is_flag=True,
    default=False,
    help="Merge a merge-aware adapter despite a quantizer-identity mismatch "
    "(e.g. different torchao version than trained with).",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config_options(AXOLOTL_CONFIG_CLI_OPTIONS)
@filter_none_kwargs
def merge_lora(config: str, **kwargs):
    """
    Merge trained LoRA adapters into a base model.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    from axolotl.cli.merge_lora import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument(
    "directory", type=click.Choice(["examples", "deepspeed_configs", "docs"])
)
@click.option("--dest", help="Destination directory")
def fetch(directory: str, dest: Optional[str]):
    """
    Fetch example configs or other resources.

    Available directories:
    - examples: Example configuration files
    - deepspeed_configs: DeepSpeed configuration files
    - docs: Full documentation (Quarto markdown files)

    Args:
        directory: One of `examples`, `deepspeed_configs`, `docs`.
        dest: Optional destination directory.
    """
    fetch_from_github(f"{directory}/", dest)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(VllmServeCliArgs)
@filter_none_kwargs
def vllm_serve(config: str, **cli_args: VllmServeCliArgs):
    from axolotl.cli.vllm_serve import do_vllm_serve

    do_vllm_serve(config, cli_args)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(QuantizeCliArgs)
@filter_none_kwargs
def quantize(config: str, **cli_args: QuantizeCliArgs):
    from axolotl.cli.quantize import do_quantize

    do_quantize(config, cli_args)


@cli.command()
@click.argument("model", type=click.Path(exists=True, path_type=str))
@click.argument("output", type=click.Path(exists=False, path_type=str))
def delinearize_llama4(model: str, output: str):
    from axolotl.cli.delinearize_llama4 import do_cli as do_delinearize_llama4

    do_delinearize_llama4(model, output)


@cli.command("generate-cli-config-options")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("src/axolotl/cli/config_options.py"),
    show_default=True,
    help="Path to write generated option metadata.",
)
@click.option("--check", is_flag=True, help="Fail if generated metadata is stale.")
def generate_cli_config_options(output: Path, check: bool):
    """Regenerate CLI config override option metadata."""
    from axolotl.cli.generate_config_options import write_options

    write_options(output, check=check)


@cli.command("agent-docs")
@click.argument("topic", required=False, default=None)
@click.option("--list", "list_topics", is_flag=True, help="List available topics")
def agent_docs(topic: Optional[str], list_topics: bool):
    """Show agent-optimized documentation.

    Prints reference docs designed for AI coding agents.
    These docs are bundled with the package — no network access needed.

    \b
    Examples:
        axolotl agent-docs              # overview (start here)
        axolotl agent-docs grpo         # GRPO reference
        axolotl agent-docs sft          # SFT reference
        axolotl agent-docs --list       # list all topics
    """
    from axolotl.cli.agent_docs import get_doc, list_topics as _list_topics

    if list_topics:
        for name, title in _list_topics().items():
            click.echo(f"  {name:25s} {title}")
        return

    if topic is None:
        topic = "overview"

    try:
        click.echo(get_doc(topic))
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc)) from exc


@cli.command("config-schema")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format (default: json)",
)
@click.option("--field", help="Show schema for a specific field only")
@click.option(
    "--runtime",
    "runtime_schema",
    is_flag=True,
    help="Dump the runtime (--runtime file) schema instead of the training config schema",
)
def config_schema(output_format: str, field: Optional[str], runtime_schema: bool):
    """Dump the full config JSON schema.

    Useful for AI agents and tooling to discover all available config options,
    their types, defaults, and descriptions.

    \b
    Examples:
        axolotl config-schema                    # full JSON schema
        axolotl config-schema --format yaml      # YAML format
        axolotl config-schema --field adapter     # single field
        axolotl config-schema --runtime           # --runtime file schema
    """
    import json

    from pydantic import BaseModel

    schema_model: type[BaseModel]
    if runtime_schema:
        from axolotl.utils.schemas.runtime import RuntimeConfig

        schema_model = RuntimeConfig
    else:
        from axolotl.utils.schemas.config import AxolotlInputConfig

        schema_model = AxolotlInputConfig

    try:
        schema = schema_model.model_json_schema()
    except (TypeError, ValueError, AttributeError) as exc:
        # Fallback: dump field names, types, and defaults when full schema
        # generation fails (e.g. torch.dtype not JSON-serializable)
        LOG.warning(
            "Full JSON schema generation failed, using simplified fallback: %s", exc
        )
        fields = {}
        for name, field_info in schema_model.model_fields.items():
            entry = {}
            if field_info.description:
                entry["description"] = field_info.description
            if field_info.default is not None:
                try:
                    json.dumps(field_info.default)
                    entry["default"] = field_info.default
                except (TypeError, ValueError):
                    entry["default"] = str(field_info.default)
            annotation = field_info.annotation
            if annotation is not None:
                entry["type"] = str(annotation)
            fields[name] = entry
        schema = {
            "properties": fields,
            "_note": "simplified schema (full generation failed)",
        }

    if field:
        props = schema.get("properties", {})
        if field not in props:
            # Try case-insensitive match
            matches = [k for k in props if k.lower() == field.lower()]
            if matches:
                field = matches[0]
            else:
                raise click.BadParameter(
                    f"Unknown field: {field!r}. "
                    f"Omit --field to dump the full schema, "
                    f"or pipe to jq: axolotl config-schema | jq '.properties | keys'"
                )
        schema = {field: props[field]}

    if output_format == "yaml":
        import yaml  # pylint: disable=import-outside-toplevel

        click.echo(yaml.dump(schema, default_flow_style=False, sort_keys=False))
    else:
        click.echo(json.dumps(schema, indent=2))


@cli.group()
def ray():
    """Manage a Ray cluster for axolotl training (`up`, `down`, `status`)."""


@ray.command(name="up")
@click.option(
    "--runtime",
    default=None,
    type=click.Path(exists=True, path_type=str),
    help="Runtime YAML whose ray.cluster block describes the cluster",
)
@click.option(
    "--hostfile",
    default=None,
    type=click.Path(exists=True, path_type=str),
    help="Hostfile of worker nodes to join over ssh (head is this machine)",
)
@click.option("--port", default=None, type=int, help="Ray head port (default 6379)")
@click.option(
    "--dashboard-port",
    default=None,
    type=int,
    help="Dashboard/Jobs API port (default 8265)",
)
@click.option("--ssh-user", default=None, help="SSH user for worker nodes")
@click.option(
    "--ssh-key",
    default=None,
    type=click.Path(exists=True, path_type=str),
    help="SSH identity file for worker nodes",
)
def ray_up(
    runtime: str | None,
    hostfile: str | None,
    port: int | None,
    dashboard_port: int | None,
    ssh_user: str | None,
    ssh_key: str | None,
):
    """Start a Ray head on this machine and join hostfile workers over ssh."""
    from axolotl.cli.launchers.ray_cluster import cluster_up

    runtime_cfg = None
    if runtime:
        from axolotl.cli.launchers.resolve import _load_runtime_config

        runtime_cfg = _load_runtime_config(runtime)
    cluster_up(
        hostfile=hostfile,
        port=port,
        dashboard_port=dashboard_port,
        ssh_user=ssh_user,
        ssh_key=ssh_key,
        runtime=runtime_cfg,
    )


@ray.command(name="down")
@click.option(
    "--force",
    is_flag=True,
    help="pkill Ray daemons by their unique temp-dir tag instead of `ray stop`",
)
def ray_down(force: bool):
    """Stop the cluster started by `axolotl ray up`."""
    from axolotl.cli.launchers.ray_cluster import cluster_down

    cluster_down(force=force)


@ray.command(name="status")
@click.option(
    "--address",
    default=None,
    help="Ray address to query (defaults to the cluster recorded by `axolotl ray up`)",
)
def ray_status(address: str | None):
    """Show Ray cluster status."""
    from axolotl.cli.launchers.ray_cluster import cluster_status

    cluster_status(address=address)


def main():
    cli()


if __name__ == "__main__":
    main()
