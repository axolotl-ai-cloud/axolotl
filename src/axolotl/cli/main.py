"""
CLI definition for various axolotl commands
"""
# pylint: disable=redefined-outer-name
import dataclasses
import subprocess  # nosec B404
from types import NoneType
from typing import Any, Optional, Type, Union, get_args, get_origin

import click
from pydantic import BaseModel

from axolotl.cli.utils import build_command, fetch_from_github
from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig


@click.group()
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""


def add_options_from_dataclass(config_class: Type[Any]):
    """Create Click options from the fields of a dataclass."""

    def decorator(function):
        for field in reversed(dataclasses.fields(config_class)):
            field_type = field.type

            if get_origin(field_type) is Union and type(None) in get_args(field_type):
                field_type = next(
                    t for t in get_args(field_type) if not isinstance(t, NoneType)
                )

            if field_type == bool:
                field_name = field.name.replace("_", "-")
                option_name = f"--{field_name}/--no-{field_name}"
                function = click.option(
                    option_name,
                    default=field.default,
                    help=field.metadata.get("description"),
                )(function)
            else:
                option_name = f"--{field.name.replace('_', '-')}"
                function = click.option(
                    option_name,
                    type=field_type,
                    default=field.default,
                    help=field.metadata.get("description"),
                )(function)
        return function

    return decorator


def add_options_from_config(config_class: Type[BaseModel]):
    """Create Click options from the fields of a Pydantic model."""

    def decorator(function):
        for name, field in reversed(config_class.model_fields.items()):
            if field.annotation == bool:
                field_name = name.replace("_", "-")
                option_name = f"--{field_name}/--no-{field_name}"
                function = click.option(
                    option_name, default=None, help=field.description
                )(function)
            else:
                option_name = f"--{name.replace('_', '-')}"
                function = click.option(
                    option_name, default=None, help=field.description
                )(function)
        return function

    return decorator


@cli.command()
@click.argument("config", type=str)
@add_options_from_dataclass(PreprocessCliArgs)
@add_options_from_config(AxolotlInputConfig)
def preprocess(config: str, **kwargs):
    """Preprocess datasets before training."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    from axolotl.cli.preprocess import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=str)
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
@click.argument("config", type=str)
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for multi-GPU inference",
)
@click.option("--lora-model-dir", help="Directory containing LoRA model")
@click.option("--base-model", help="Path to base model for non-LoRA models")
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@click.option("--load-in-8bit", is_flag=True, help="Load model in 8-bit mode")
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def inference(config: str, accelerate: bool, **kwargs):
    """Run inference with a trained model."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

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
@click.argument("config", type=str)
@click.option(
    "--accelerate/--no-accelerate",
    default=False,
    help="Use accelerate launch for multi-GPU operations",
)
@click.option("--model-dir", help="Directory containing model weights to shard")
@click.option("--save-dir", help="Directory to save sharded weights")
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
@click.argument("config", type=str)
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for weight merging",
)
@click.option("--model-dir", help="Directory containing sharded weights")
@click.option("--save-path", help="Path to save merged weights")
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
