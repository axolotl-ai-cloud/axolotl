"""Click CLI definitions for various axolotl commands."""

# pylint: disable=redefined-outer-name

import subprocess  # nosec B404
import tempfile
from pathlib import Path
from typing import Optional

import click
import yaml
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
from axolotl.cli.sweeps import generate_sweep_configs
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    execute_training,
    fetch_from_github,
    filter_none_kwargs,
)
from axolotl.integrations.lm_eval.cli import lm_eval
from axolotl.utils import patch_optimized_env
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.config import AxolotlInputConfig

LOG = get_logger(__name__)


@click.group()
@click.version_option(version=axolotl.__version__, prog_name="axolotl")
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""
    print_axolotl_text_art()
    load_dotenv()
    patch_optimized_env()


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option("--cloud", default=None, type=click.Path(exists=True, path_type=str))
@add_options_from_dataclass(PreprocessCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def preprocess(config: str, cloud: Optional[str] = None, **kwargs) -> None:
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


@cli.command()
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
def train(
    config: str,
    launcher: str,
    cloud: str | None = None,
    sweep: str | None = None,
    **kwargs,
) -> None:
    """
    Train or fine-tune a model.

    Args:
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU training ("accelerate", "torchrun", or "python").
        cloud: Path to a cloud accelerator configuration file
        sweep: Path to YAML config for sweeping hyperparameters.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Handle Ray launcher override
    effective_launcher = None if kwargs.get("use_ray") else launcher

    # Generate configuration files to process
    config_files = _generate_config_files(config, sweep)

    # Process each configuration
    for cfg_file in config_files:
        try:
            execute_training(cfg_file, effective_launcher, cloud, kwargs)
        except subprocess.CalledProcessError as exc:
            LOG.error(f"Failed to train/fine-tune config '{cfg_file}': {exc}")
            if not sweep:
                raise exc


def _generate_config_files(config: str, sweep: str | None) -> list[str]:
    """Generate list of configuration files to process."""
    if not sweep:
        return [config]

    # Load sweep and base configurations
    with open(sweep, "r", encoding="utf-8") as fin:
        sweep_config: dict[str, list] = yaml.safe_load(fin)
    with open(config, "r", encoding="utf-8") as fin:
        base_config: dict[str, list] = yaml.safe_load(fin)

    # Generate all possible configurations
    permutations = generate_sweep_configs(base_config, sweep_config)

    config_files = []
    for perm in permutations:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config_path = Path(temp_dir) / "config.yaml"
            with open(temp_config_path, "w", encoding="utf-8") as fout:
                yaml.dump(perm, fout)
            config_files.append(str(temp_config_path))

    return config_files


@cli.command()
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
def evaluate(config: str, launcher: str, **kwargs) -> None:
    """
    Evaluate a model.

    Args:
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU evaluation ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    if launcher == "accelerate":
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.evaluate"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    elif launcher == "torchrun":
        base_cmd = ["torchrun", "-m", "axolotl.cli.evaluate"]
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
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for multi-GPU inference",
)
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def inference(config: str, launcher: str, gradio: bool, **kwargs) -> None:
    """
    Run inference with a trained model.

    Args:
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for multi-GPU inference ("accelerate", "torchrun", or "python").
        gradio: Whether to use Gradio browser interface or command line for inference.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    if launcher == "accelerate":
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.inference"]
        if config:
            base_cmd.append(config)
        if gradio:
            base_cmd.append("--gradio")
        cmd = build_command(base_cmd, kwargs)
        subprocess.run(cmd, check=True)  # nosec B603
    elif launcher == "torchrun":
        base_cmd = ["torchrun", "-m", "axolotl.cli.inference"]
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
    "--launcher",
    type=click.Choice(["accelerate", "torchrun", "python"]),
    default="accelerate",
    help="Launcher to use for weight merging",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def merge_sharded_fsdp_weights(config: str, launcher: str, **kwargs) -> None:
    """
    Merge sharded FSDP model weights.

    Args:
        config: Path to `axolotl` config YAML file.
        launcher: Launcher to use for weight merging ("accelerate", "torchrun", or "python").
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    if launcher == "accelerate":
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
    elif launcher == "torchrun":
        base_cmd = ["torchrun", "-m", "axolotl.cli.merge_sharded_fsdp_weights"]
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
def merge_lora(config: str, **kwargs) -> None:
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
def delinearize_llama4(model: str, output: str) -> None:
    from axolotl.cli.delinearize_llama4 import do_cli as do_delinearize_llama4

    do_delinearize_llama4(model, output)


cli.add_command(lm_eval)


def main():
    cli()


if __name__ == "__main__":
    main()
