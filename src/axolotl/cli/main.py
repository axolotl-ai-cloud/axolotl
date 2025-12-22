"""Click CLI definitions for various axolotl commands."""

import os
import subprocess  # nosec B404
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
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
    filter_none_kwargs,
    generate_config_files,
    launch_training,
)
from axolotl.integrations.lm_eval.cli import lm_eval
from axolotl.utils import set_misc_env, set_pytorch_cuda_alloc_conf
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.config import AxolotlInputConfig

LOG = get_logger(__name__)

LAUNCHER_COMMAND_MAPPING = {
    "accelerate": ["accelerate", "launch"],
    "torchrun": ["torchrun"],
}


@click.group()
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
@add_options_from_config(AxolotlInputConfig)
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
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU training",
)
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@click.option(
    "--sweep",
    type=click.Path(exists=True, path_type=str),
    help="YAML config for sweeping hyperparameters",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def train(
    ctx: click.Context,
    config: str,
    launcher: Literal["accelerate", "torchrun", "python"] = "accelerate",
    cloud: str | None = None,
    sweep: str | None = None,
    **kwargs,
):
    """
    Train or fine-tune a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU training ("accelerate", "torchrun", or "python").
        cloud: Path to a cloud accelerator configuration file
        sweep: Path to YAML config for sweeping hyperparameters.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    # Handle Ray launcher override
    _launcher = None if kwargs.get("use_ray") else launcher

    # Process each configuration
    for cfg_file, is_group in generate_config_files(config, sweep):
        try:
            use_exec = is_group is not True
            launch_training(cfg_file, _launcher, cloud, kwargs, launcher_args, use_exec)
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
    default="accelerate",
    help="Launcher to use for multi-GPU evaluation",
)
@add_options_from_dataclass(EvaluateCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def evaluate(ctx: click.Context, config: str, launcher: str, **kwargs):
    """
    Evaluate a model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU evaluation ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Extract launcher args from extra args (after --)
    launcher_args = ctx.args if ctx.args else []

    if launcher in LAUNCHER_COMMAND_MAPPING:
        base_cmd = (
            LAUNCHER_COMMAND_MAPPING[launcher]
            + launcher_args
            + ["-m", "axolotl.cli.evaluate"]
        )
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
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
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
@click.pass_context
def inference(ctx: click.Context, config: str, launcher: str, gradio: bool, **kwargs):
    """
    Run inference with a trained model.

    Args:
        ctx: Click context for extra args.
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU inference ("accelerate", "torchrun", or "python").
        gradio: Whether to use Gradio browser interface or command line for inference.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, gradio=gradio, **kwargs)


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
@add_options_from_config(AxolotlInputConfig)
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
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
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
@click.argument("directory", type=click.Choice(["examples", "deepspeed_configs"]))
@click.option("--dest", help="Destination directory")
def fetch(directory: str, dest: Optional[str]):
    """
    Fetch example configs or other resources.

    Available directories:
    - examples: Example configuration files
    - deepspeed_configs: DeepSpeed configuration files

    Args:
        directory: One of `examples`, `deepspeed_configs`.
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


cli.add_command(lm_eval)


def main():
    cli()


if __name__ == "__main__":
    main()
