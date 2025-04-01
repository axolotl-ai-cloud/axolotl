"""Click CLI definitions for various axolotl commands."""

# pylint: disable=redefined-outer-name

import logging
import os
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
    TrainerCliArgs,
    VllmServeCliArgs,
)
from axolotl.cli.sweeps import generate_sweep_configs
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
    filter_none_kwargs,
)
from axolotl.cli.vllm_serve import do_vllm_serve
from axolotl.integrations.lm_eval.cli import lm_eval
from axolotl.utils import set_pytorch_cuda_alloc_conf
from axolotl.utils.schemas.config import AxolotlInputConfig


@click.group()
@click.version_option(version=axolotl.__version__, prog_name="axolotl")
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""


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
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for multi-GPU training",
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
    accelerate: bool,
    cloud: Optional[str] = None,
    sweep: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Train or fine-tune a model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        cloud: Path to a cloud accelerator configuration file
        sweep: Path to YAML config for sweeping hyperparameters.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
    # Enable expandable segments for cuda allocation to improve VRAM usage
    set_pytorch_cuda_alloc_conf()

    if "use_ray" in kwargs and kwargs["use_ray"]:
        accelerate = False
    if sweep:
        # load the sweep configuration yaml file
        with open(sweep, "r", encoding="utf-8") as fin:
            sweep_config: dict[str, list] = yaml.safe_load(fin)
        with open(config, "r", encoding="utf-8") as fin:
            base_config: dict[str, list] = yaml.safe_load(fin)

        # generate all possible configurations
        permutations = generate_sweep_configs(base_config, sweep_config)

        def iter_configs():
            for perm in permutations:
                # open temp directory for temporary configurations
                with tempfile.TemporaryDirectory() as temp_dir:
                    with open(
                        Path(temp_dir) / "config.yaml", "w", encoding="utf-8"
                    ) as fout:
                        yaml.dump(perm, fout)
                    yield str(Path(temp_dir) / "config.yaml")

    else:

        def iter_configs():
            yield config

    for cfg_file in iter_configs():
        # handle errors from subprocess so we can continue rest of sweeps
        try:
            if accelerate:
                if cloud:
                    from axolotl.cli.cloud import do_cli_train

                    cwd = os.getcwd()
                    do_cli_train(
                        cloud_config=cloud,
                        config=config,
                        accelerate=True,
                        cwd=cwd,
                        **kwargs,
                    )
                else:
                    accelerate_args = []
                    if "main_process_port" in kwargs:
                        main_process_port = kwargs.pop("main_process_port", None)
                        accelerate_args.append("--main_process_port")
                        accelerate_args.append(str(main_process_port))
                    if "num_processes" in kwargs:
                        num_processes = kwargs.pop("num_processes", None)
                        accelerate_args.append("--num_processes")
                        accelerate_args.append(str(num_processes))

                    base_cmd = ["accelerate", "launch"]
                    base_cmd.extend(accelerate_args)
                    base_cmd.extend(["-m", "axolotl.cli.train"])
                    if cfg_file:
                        base_cmd.append(cfg_file)
                    cmd = build_command(base_cmd, kwargs)
                    subprocess.run(cmd, check=True)  # nosec B603
            else:
                if cloud:
                    from axolotl.cli.cloud import do_cli_train

                    do_cli_train(
                        cloud_config=cloud, config=config, accelerate=False, **kwargs
                    )
                else:
                    from axolotl.cli.train import do_cli

                    do_cli(config=cfg_file, **kwargs)
        except subprocess.CalledProcessError as exc:
            logging.error(f"Failed to train/fine-tune config '{cfg_file}': {exc}")
            if not sweep:
                raise exc


@cli.command()
@click.argument("config", type=click.Path(exists=True, path_type=str))
@click.option(
    "--accelerate/--no-accelerate",
    default=True,
    help="Use accelerate launch for multi-GPU training",
)
@add_options_from_dataclass(EvaluateCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def evaluate(config: str, accelerate: bool, **kwargs) -> None:
    """
    Evaluate a model.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
@filter_none_kwargs
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
    default=True,
    help="Use accelerate launch for weight merging",
)
@add_options_from_dataclass(TrainerCliArgs)
@add_options_from_config(AxolotlInputConfig)
@filter_none_kwargs
def merge_sharded_fsdp_weights(config: str, accelerate: bool, **kwargs) -> None:
    """
    Merge sharded FSDP model weights.

    Args:
        config: Path to `axolotl` config YAML file.
        accelerate: Whether to use `accelerate` launcher.
        kwargs: Additional keyword arguments which correspond to CLI args or `axolotl`
            config options.
    """
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
    do_vllm_serve(config, cli_args)


cli.add_command(lm_eval)


def main():
    cli()


if __name__ == "__main__":
    load_dotenv()
    main()
