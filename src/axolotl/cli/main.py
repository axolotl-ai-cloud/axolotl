"""Click CLI definitions for various axolotl commands."""
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
from axolotl.common.cli import EvaluateCliArgs, PreprocessCliArgs, TrainerCliArgs
from axolotl.utils import set_pytorch_cuda_alloc_conf
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
    """
    Preprocess datasets before training.

    Args:
        config: Path to `axolotl` config YAML file.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
def train(config: str, accelerate: bool, **kwargs) -> None:
    """
    Train or fine-tune a model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # Enable expandable segments for cuda allocation to improve VRAM usage
    set_pytorch_cuda_alloc_conf()

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
    help="Use accelerate launch for multi-GPU training",
)
@add_options_from_dataclass(EvaluateCliArgs)
@add_options_from_config(AxolotlInputConfig)
def evaluate(config: str, accelerate: bool, **kwargs) -> None:
    """
    Evaluate a model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.evaluate"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.evaluate import do_cli

        do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=False,
    help="Use accelerate launch for multi-GPU inference",
)
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def inference(config: str, accelerate: bool, gradio: bool, **kwargs) -> None:
    """
    Run inference with a trained model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        gradio: Whether to use Gradio browser interface or command line for inference.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.inference"]
        if config:
            base_cmd.append(config)
        if gradio:
            base_cmd.append("--gradio")
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, gradio=gradio, **kwargs)


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=False,
    help="Use accelerate launch for multi-GPU operations",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def shard(config: str, accelerate: bool, **kwargs) -> None:
    """
    Shard model weights into chunks.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def merge_sharded_fsdp_weights(config: str, accelerate: bool, **kwargs) -> None:
    """
    Merge sharded FSDP model weights.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
def merge_lora(config: str, **kwargs) -> None:
    """
    Merge trained LoRA adapters into a base model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    from axolotl.cli.merge_lora import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("directory", type=click.Choice(["examples", "deepspeed_configs"]))
@click.option("--dest", help="Destination directory")
def fetch(directory: str, dest: Optional[str]) -> None:
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


def main():
    cli()


if __name__ == "__main__":
    main()
