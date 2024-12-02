"""
CLI definition for various axolotl commands
"""
# pylint: disable=redefined-outer-name
import hashlib
import json
import os
import subprocess  # nosec B404
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import requests

from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs


def create_args_from_dataclass(cls, **kwargs):
    """Create args from dataclass, using its default values if not in kwargs."""
    field_defaults = {field.name: field.default for field in fields(cls)}

    # Update defaults with provided kwargs
    field_defaults.update(kwargs)
    return cls(**field_defaults)


def create_trainer_args(**kwargs) -> TrainerCliArgs:
    """Create TrainerCliArgs from cli options"""
    return create_args_from_dataclass(TrainerCliArgs, **kwargs)


def create_preprocess_args(**kwargs) -> PreprocessCliArgs:
    """Create PreprocessCliArgs from cli options"""
    return create_args_from_dataclass(PreprocessCliArgs, **kwargs)


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


def common_options(function):
    """Common options shared between commands"""
    function = click.option("--debug", is_flag=True, help="Enable debug mode")(function)
    function = click.option(
        "--debug-text-only", is_flag=True, help="Debug text processing only"
    )(function)
    function = click.option(
        "--debug-num-examples",
        type=int,
        default=0,
        help="Number of examples to use in debug mode",
    )(function)
    function = click.option("--prompter", type=str, help="Prompter to use")(function)

    return function


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


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option("--download", is_flag=True, default=True, help="Download datasets")
@click.option(
    "--use-gpu",
    is_flag=True,
    default=False,
    help="Allow GPU usage during preprocessing",
)
def preprocess(config: str, use_gpu: bool, **kwargs):
    """Preprocess datasets before training."""
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from axolotl.cli.preprocess import do_cli

    do_cli(config=config, **kwargs)


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option(
    "--accelerate",
    is_flag=True,
    default=False,
    help="Use accelerate launch for multi-GPU training",
)
def train(config: str, accelerate: bool, **kwargs):
    """Train or fine-tune a model."""
    cli_args = create_trainer_args(**kwargs)

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.train"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, vars(cli_args))
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.train import do_cli

        do_cli(config=config, **cli_args.__dict__)


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option(
    "--accelerate",
    is_flag=True,
    default=False,
    help="Use accelerate launch for multi-GPU inference",
)
@click.option("--lora-model-dir", help="Directory containing LoRA model")
@click.option("--base-model", help="Path to base model for non-LoRA models")
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@click.option("--load-in-8bit", is_flag=True, help="Load model in 8-bit mode")
def inference(config: str, accelerate: bool, **kwargs):
    """Run inference with a trained model."""
    cli_args = create_trainer_args(inference=True, **kwargs)

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.inference"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, vars(cli_args))
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.inference import do_cli

        do_cli(config=config, **cli_args.__dict__)


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option(
    "--accelerate",
    is_flag=True,
    default=False,
    help="Use accelerate launch for multi-GPU operations",
)
@click.option("--model-dir", help="Directory containing model weights to shard")
@click.option("--save-dir", help="Directory to save sharded weights")
def shard(config: str, accelerate: bool, **kwargs):
    """Shard model weights."""
    cli_args = create_trainer_args(shard=True, **kwargs)

    if accelerate:
        base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.shard"]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, vars(cli_args))
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.shard import do_cli

        do_cli(config=config, **cli_args.__dict__)


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option(
    "--accelerate",
    is_flag=True,
    default=False,
    help="Use accelerate launch for weight merging",
)
@click.option("--model-dir", help="Directory containing sharded weights")
@click.option("--save-path", help="Path to save merged weights")
def merge_sharded_fsdp_weights(config: str, accelerate: bool, **kwargs):
    """Merge sharded FSDP model weights."""
    cli_args = create_trainer_args(**kwargs)

    if accelerate:
        base_cmd = [
            "accelerate",
            "launch",
            "-m",
            "axolotl.cli.merge_sharded_fsdp_weights",
        ]
        if config:
            base_cmd.append(config)
        cmd = build_command(base_cmd, vars(cli_args))
        subprocess.run(cmd, check=True)  # nosec B603
    else:
        from axolotl.cli.merge_sharded_fsdp_weights import do_cli

        do_cli(config=config, **cli_args.__dict__)


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
