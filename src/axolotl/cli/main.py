"""
CLI definition for various axolotl commands
"""
import os
import subprocess  # nosec B404
import sys
from typing import Any, Dict, List

import click

from axolotl.common.cli import PreprocessCliArgs, TrainerCliArgs


def create_trainer_args(**kwargs) -> TrainerCliArgs:
    """Create TrainerCliArgs from click options"""
    return TrainerCliArgs(
        debug=kwargs.get("debug", False),
        debug_text_only=kwargs.get("debug_text_only", False),
        debug_num_examples=kwargs.get("debug_num_examples", 0),
        inference=kwargs.get("inference", False),
        merge_lora=kwargs.get("merge_lora", False),
        prompter=kwargs.get("prompter"),
        shard=kwargs.get("shard", False),
    )


def create_preprocess_args(**kwargs) -> PreprocessCliArgs:
    """Create PreprocessCliArgs from click options"""
    return PreprocessCliArgs(
        debug=kwargs.get("debug", False),
        debug_text_only=kwargs.get("debug_text_only", False),
        debug_num_examples=kwargs.get("debug_num_examples", 1),
        prompter=kwargs.get("prompter"),
        download=kwargs.get("download", True),
    )


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


@click.group()
def cli():
    """Axolotl CLI - Train and fine-tune large language models"""


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option("--download", is_flag=True, default=True, help="Download datasets")
def preprocess(config: str, **kwargs):
    """Preprocess datasets before training."""
    cli_args = create_preprocess_args(**kwargs)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""

    base_cmd = [sys.executable, "-m", "axolotl.cli.preprocess", config]
    cmd = build_command(base_cmd, vars(cli_args))

    subprocess.run(cmd, env=env, check=True)  # nosec B603


@cli.command()
@click.argument("config", type=str)
@common_options
def train(config: str, **kwargs):
    """Train or fine-tune a model."""
    cli_args = create_trainer_args(**kwargs)

    base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.train", config]
    cmd = build_command(base_cmd, vars(cli_args))

    subprocess.run(cmd, check=True)  # nosec B603


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option("--lora-model-dir", help="Directory containing LoRA model")
@click.option("--base-model", help="Path to base model for non-LoRA models")
@click.option("--gradio", is_flag=True, help="Launch Gradio interface")
@click.option("--load-in-8bit", is_flag=True, help="Load model in 8-bit mode")
def inference(config: str, **kwargs):
    """Run inference with a trained model."""
    cli_args = create_trainer_args(inference=True, **kwargs)

    base_cmd = ["accelerate", "launch", "-m", "axolotl.cli.inference", config]
    cmd = build_command(base_cmd, vars(cli_args))

    subprocess.run(cmd, check=True)  # nosec B603


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option("--model-dir", help="Directory containing model weights to shard")
@click.option("--save-dir", help="Directory to save sharded weights")
def shard(config: str, **kwargs):
    """Shard model weights."""
    cli_args = create_trainer_args(shard=True, **kwargs)

    base_cmd = [sys.executable, "-m", "axolotl.cli.shard", config]
    cmd = build_command(base_cmd, vars(cli_args))

    subprocess.run(cmd, check=True)  # nosec B603


@cli.command()
@click.argument("config", type=str)
@common_options
@click.option("--model-dir", help="Directory containing sharded weights")
@click.option("--save-path", help="Path to save merged weights")
def merge_sharded_fsdp_weights(config: str, **kwargs):
    """Merge sharded FSDP model weights."""
    cli_args = create_trainer_args(**kwargs)

    base_cmd = [sys.executable, "-m", "axolotl.cli.merge_sharded_fsdp_weights", config]
    cmd = build_command(base_cmd, vars(cli_args))

    subprocess.run(cmd, check=True)  # nosec B603


def main():
    cli()


if __name__ == "__main__":
    main()
