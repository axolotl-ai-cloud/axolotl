"""
CLI definition for various axolotl commands
"""
# pylint: disable=redefined-outer-name
import dataclasses
import hashlib
import json
import os
import subprocess  # nosec B404
from pathlib import Path
from types import NoneType
from typing import Any, Dict, List, Optional, Type, Union, get_args, get_origin

import click
import requests
from pydantic import BaseModel

from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs
from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig


def build_command(base_cmd: List[str], options: Dict[str, Any]) -> List[str]:
    """Build command list from base command and options."""
    cmd = base_cmd.copy()

    for key, value in options.items():
        if value is None:
            continue

        key = key.replace("_", "-")

        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        else:
            cmd.extend([f"--{key}", str(value)])

    return cmd


def fetch_from_github(dir_prefix: str, dest_dir: Optional[str] = None) -> None:
    """
    Sync files from a specific directory in the GitHub repository.
    Only downloads files that don't exist locally or have changed.

    Args:
        dir_prefix: Directory prefix to filter files (e.g., 'examples/', 'deepspeed_configs/')
        dest_dir: Local destination directory
    """
    api_url = "https://api.github.com/repos/axolotl-ai-cloud/axolotl/git/trees/main?recursive=1"
    raw_base_url = "https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main"

    # Get repository tree with timeout
    response = requests.get(api_url, timeout=30)
    response.raise_for_status()
    tree = json.loads(response.text)

    # Filter for files and get their SHA
    files = {
        item["path"]: item["sha"]
        for item in tree["tree"]
        if item["type"] == "blob" and item["path"].startswith(dir_prefix)
    }

    if not files:
        raise click.ClickException(f"No files found in {dir_prefix}")

    # Default destination directory is the last part of dir_prefix
    default_dest = Path(dir_prefix.rstrip("/"))
    dest_path = Path(dest_dir) if dest_dir else default_dest

    # Keep track of processed files for summary
    files_processed: Dict[str, List[str]] = {"new": [], "updated": [], "unchanged": []}

    for file_path, remote_sha in files.items():
        # Create full URLs and paths
        raw_url = f"{raw_base_url}/{file_path}"
        dest_file = dest_path / file_path.split(dir_prefix)[-1]

        # Check if file exists and needs updating
        if dest_file.exists():
            # Git blob SHA is calculated with a header
            with open(dest_file, "rb") as file:
                content = file.read()

                # Calculate git blob SHA
                blob = b"blob " + str(len(content)).encode() + b"\0" + content
                local_sha = hashlib.sha1(blob, usedforsecurity=False).hexdigest()

            if local_sha == remote_sha:
                print(f"Skipping {file_path} (unchanged)")
                files_processed["unchanged"].append(file_path)
                continue

            print(f"Updating {file_path}")
            files_processed["updated"].append(file_path)
        else:
            print(f"Downloading {file_path}")
            files_processed["new"].append(file_path)

        # Create directories if needed
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Download and save file
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()

        with open(dest_file, "wb") as file:
            file.write(response.content)

    # Print summary
    print("\nSync Summary:")
    print(f"New files: {len(files_processed['new'])}")
    print(f"Updated files: {len(files_processed['updated'])}")
    print(f"Unchanged files: {len(files_processed['unchanged'])}")


@click.group()
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""


def get_click_type(python_type: Type) -> Any:
    """Convert Python/Pydantic types to Click types."""
    # Handle Union/Optional types
    if get_origin(python_type) is Union:
        types = get_args(python_type)
        # If one of the types is None, it's Optional
        types = tuple(t for t in types if not isinstance(t, NoneType))
        if len(types) == 1:
            return get_click_type(types[0])

    # Map Python types to Click types
    type_map = {
        str: str,
        int: int,
        float: float,
        bool: bool,
    }
    return type_map.get(python_type, str)


def add_options_from_dataclass(config_class: Type[Any]):
    """Create Click options from the fields of a dataclass."""

    def decorator(function):
        for field in reversed(dataclasses.fields(config_class)):
            option_name = f"--{field.name.replace('_', '-')}"

            if field.type == bool:
                function = click.option(
                    option_name,
                    is_flag=True,
                    default=None,
                    help=field.metadata.get("description"),
                )(function)
            else:
                function = click.option(
                    option_name,
                    type=field.type,
                    default=None,
                    help=field.metadata.get("description"),
                )(function)
        return function

    return decorator


def add_options_from_config(config_class: Type[BaseModel]):
    """Create Click options from the fields of a Pydantic model."""

    def decorator(function):
        for name, field in reversed(config_class.model_fields.items()):
            option_name = f"--{name.replace('_', '-')}"

            if field.annotation == bool:
                function = click.option(
                    option_name, is_flag=True, default=None, help=field.description
                )(function)
            else:
                function = click.option(
                    option_name, default=None, help=field.description
                )(function)
        return function

    return decorator


@cli.command()
@click.argument("config", type=str)
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Allow GPU usage during preprocessing",
)
@add_options_from_dataclass(PreprocessCliArgs)
def preprocess(config: str, use_gpu: bool, **kwargs):
    """Preprocess datasets before training."""
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from axolotl.cli.preprocess import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=str)
@click.option(
    "--accelerate",
    is_flag=True,
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
    "--accelerate",
    is_flag=True,
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
    "--accelerate",
    is_flag=True,
    default=True,
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
    "--accelerate",
    is_flag=True,
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
