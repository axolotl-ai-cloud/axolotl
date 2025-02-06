"""Click CLI definitions for various axolotl commands."""
# pylint: disable=redefined-outer-name

import logging
import random
import subprocess  # nosec B404
import tempfile
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Optional

import click
import yaml
from dotenv import load_dotenv

import axolotl
from axolotl.cli.args import EvaluateCliArgs, PreprocessCliArgs, TrainerCliArgs
from axolotl.cli.utils import (
    add_options_from_config,
    add_options_from_dataclass,
    build_command,
    fetch_from_github,
    filter_none_kwargs,
)
from axolotl.integrations.lm_eval.cli import lm_eval
from axolotl.utils import set_pytorch_cuda_alloc_conf
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig


def generate_sweep_configs(base_config, sweeps_config):
    """
    Recursively generates all possible configurations by applying sweeps to the base config.

    Args:
        base_config (dict): The original configuration dictionary
        sweeps_config (dict): Dictionary where keys are parameters and values are either:
            - lists of values to sweep independently
            - or for paired values, a list of dicts under the '_' key

    Returns:
        list: List of all possible configuration dictionaries

    Example:
        sweeps_config = {
            'learning_rate': [0.1, 0.01],
            '_': [
                {'load_in_8bit': True, 'adapter': 'lora'},
                {'load_in_4bit': True, 'adapter': 'qlora'}
            ]
        }
    """
    # Separate paired values from regular sweeps
    paired_values = sweeps_config.get("_", [])
    regular_sweeps = {k: v for k, v in sweeps_config.items() if k != "_"}

    # Process regular sweeps
    param_names = list(regular_sweeps.keys())
    param_values = list(regular_sweeps.values())

    # Generate combinations for regular sweeps
    regular_combinations = list(product(*param_values)) if param_values else [()]

    # Combine regular sweeps with paired values
    all_combinations = []
    for reg_combo in regular_combinations:
        if paired_values:
            for paired_set in paired_values:
                new_config = {}
                # new_config = deepcopy(base_config)
                # Combine regular parameters with paired parameters
                full_combo = {**dict(zip(param_names, reg_combo)), **paired_set}
                for param_name, param_value in full_combo.items():
                    new_config[param_name] = param_value
                print(new_config)
                all_combinations.append(new_config)
        else:
            # If no paired values, just use regular combinations
            # new_config = deepcopy(base_config)
            new_config = {}
            for param_name, param_value in zip(param_names, reg_combo):
                new_config[param_name] = param_value
            print(new_config)
            all_combinations.append(new_config)

    # randomize the order of trials
    random.seed(42)
    random.shuffle(all_combinations)

    # Generate a new config for each combination
    result_configs = []
    for combination in all_combinations:
        new_config = deepcopy(base_config)
        for param_name, param_value in combination.items():
            new_config[param_name] = param_value
        result_configs.append(new_config)

    return result_configs


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
    from axolotl.cli.cloud import do_cli_train

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
                    do_cli_train(
                        cloud_config=cloud, config=config, accelerate=True, **kwargs
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


cli.add_command(lm_eval)


def main():
    cli()


if __name__ == "__main__":
    load_dotenv()
    main()
