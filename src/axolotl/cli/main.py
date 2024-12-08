"""CLI definition for various axolotl commands."""
# pylint: disable=redefined-outer-name
import subprocess  # nosec B404
from typing import Optional

import click

import axolotl
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
)
from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig


@click.group()
@click.version_option(version=axolotl.__version__, prog_name="axolotl")
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(PreprocessCliArgs)
@add_options_from_config(AxolotlInputConfig)
def preprocess(config: str, **kwargs):
    """Preprocess datasets before training."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    from axolotl.cli.preprocess import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for multi-GPU training",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def train(config: str, accelerate: bool, **kwargs):
    """Train or fine-tune a model."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.train"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.train import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for multi-GPU inference",
)
@click.option(
    "--lora-model-dir",
    type=click.Path(exists=True, path_type=str),
    help="Directory containing LoRA model",
)
@click.option(
    "--base-model",
    type=click.Path(exists=True, path_type=str),
    help="Path to base model for non-LoRA models",
)
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@click.option("--load-in-8bit", is_flag=True, help="Load model in 8-bit mode")
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def inference(
    config: str,
    accelerate: bool,
    lora_model_dir: Optional[str] = None,
    base_model: Optional[str] = None,
    **kwargs,
):
    """Run inference with a trained model."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    del kwargs["inference"]  # interferes with inference.do_cli

    if lora_model_dir:
        kwargs["lora_model_dir"] = lora_model_dir
    if base_model:
        kwargs["output_dir"] = base_model

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.inference"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=False,
    help="Use accelerate launch for multi-GPU operations",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=str),
    help="Directory containing model weights to shard",
)
@click.option(
    "--save-dir",
    type=click.Path(path_type=str),
    help="Directory to save sharded weights",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def shard(config: str, accelerate: bool, **kwargs):
    """Shard model weights."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.shard"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.shard import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for weight merging",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=str),
    help="Directory containing sharded weights",
)
@click.option(
    "--save-path", type=click.Path(path_type=str), help="Path to save merged weights"
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def merge_sharded_fsdp_weights(config: str, accelerate: bool, **kwargs):
    """Merge sharded FSDP model weights."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if accelerate:
        base_cmd = [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.merge_sharded_fsdp_weights",
        ]
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
    "--lora-model-dir",
    type=click.Path(exists=True, path_type=str),
    help="Directory containing the LoRA model to merge",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=str),
    help="Directory to save the merged model",
)
def merge_lora(
    config: str,
    lora_model_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    """Merge a trained LoRA into a base model"""
    kwargs = {}
    if lora_model_dir:
        kwargs["lora_model_dir"] = lora_model_dir
    if output_dir:
        kwargs["output_dir"] = output_dir

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
    """
    fetch_from_github(f"{directory}/", dest)


def main():
    cli()


if __name__ == "__main__":
    main()
