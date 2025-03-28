"""Utility methods for axolotl CLI."""

import concurrent.futures
import dataclasses
import hashlib
import json
import logging
from functools import wraps
from pathlib import Path
from types import NoneType
from typing import Any, Callable, Type, Union, get_args, get_origin

import click
import requests
from pydantic import BaseModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.models import load_model, load_processor, load_tokenizer

configure_logging()
LOG = logging.getLogger(__name__)


def strip_optional_type(field_type: type | str | None):
    """
    Extracts the non-`None` type from an `Optional` / `Union` type.

    Args:
        field_type: Type of field for Axolotl CLI command.

    Returns:
        If the input type is `Union[T, None]` or `Optional[T]`, returns `T`. Otherwise
            returns the input type unchanged.
    """
    if get_origin(field_type) is Union and type(None) in get_args(field_type):
        field_type = next(
            t for t in get_args(field_type) if not isinstance(t, NoneType)
        )

    return field_type


def filter_none_kwargs(func: Callable) -> Callable:
    """
    Wraps function to remove `None`-valued `kwargs`.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        """Filters out `None`-valued `kwargs`."""
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}

        return func(*args, **filtered_kwargs)

    return wrapper


def add_options_from_dataclass(config_class: Type[Any]) -> Callable:
    """
    Create Click options from the fields of a dataclass.

    Args:
        config_class: Dataclass with fields to parse from the CLI.

    Returns:
        Function decorator for Axolotl CLI command.
    """

    def decorator(function: Callable) -> Callable:
        # Process dataclass fields in reverse order for correct option ordering
        for field in reversed(dataclasses.fields(config_class)):
            field_type = strip_optional_type(field.type)

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


def add_options_from_config(config_class: Type[BaseModel]) -> Callable:
    """
    Create Click options from the fields of a Pydantic model.

    Args:
        config_class: PyDantic model with fields to parse from the CLI

    Returns:
        Function decorator for Axolotl CLI command.
    """

    def decorator(function: Callable) -> Callable:
        # Process model fields in reverse order for correct option ordering
        for name, field in reversed(config_class.model_fields.items()):
            field_type = strip_optional_type(field.annotation)

            if field_type == bool:
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


def build_command(base_cmd: list[str], options: dict[str, Any]) -> list[str]:
    """
    Build command list from base command and options.

    Args:
        base_cmd: Command without options.
        options: Options to parse and append to base command.

    Returns:
        List of strings giving shell command.
    """
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


def download_file(
    file_info: tuple, raw_base_url: str, dest_path: Path, dir_prefix: str
) -> tuple[str, str]:
    """
    Download a single file and return its processing status.

    Args:
        file_info: Tuple of (file_path, remote_sha).
        raw_base_url: Base URL for raw GitHub content.
        dest_path: Local destination directory.
        dir_prefix: Directory prefix to filter files.

    Returns:
        Tuple of (file_path, status) where status is 'new', 'updated', or 'unchanged'.
    """
    file_path, remote_sha = file_info
    raw_url = f"{raw_base_url}/{file_path}"
    dest_file = dest_path / file_path.split(dir_prefix)[-1]

    # Check if file exists and needs updating
    if dest_file.exists():
        with open(dest_file, "rb") as file:
            content = file.read()
            # Calculate git blob SHA
            blob = b"blob " + str(len(content)).encode() + b"\0" + content
            local_sha = hashlib.sha1(blob, usedforsecurity=False).hexdigest()

        if local_sha == remote_sha:
            print(f"Skipping {file_path} (unchanged)")
            return file_path, "unchanged"

        print(f"Updating {file_path}")
        status = "new"
    else:
        print(f"Downloading {file_path}")
        status = "new"

    # Create directories if needed
    dest_file.parent.mkdir(parents=True, exist_ok=True)

    # Download and save file
    try:
        response = requests.get(raw_url, timeout=30)
        response.raise_for_status()

        with open(dest_file, "wb") as file:
            file.write(response.content)

        return file_path, status
    except (requests.RequestException, IOError) as request_error:
        print(f"Error downloading {file_path}: {str(request_error)}")
        return file_path, "error"


def fetch_from_github(
    dir_prefix: str, dest_dir: str | None = None, max_workers: int = 5
) -> None:
    """
    Sync files from a specific directory in the GitHub repository.
    Only downloads files that don't exist locally or have changed.

    Args:
        dir_prefix: Directory prefix to filter files (e.g., 'examples/',
            'deepspeed_configs/').
        dest_dir: Local destination directory.
        max_workers: Maximum number of concurrent downloads.
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
    files_processed: dict[str, list[str]] = {
        "new": [],
        "updated": [],
        "unchanged": [],
        "error": [],
    }

    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(
                download_file,
                (file_path, remote_sha),
                raw_base_url,
                dest_path,
                dir_prefix,
            ): file_path
            for file_path, remote_sha in files.items()
        }

        # Process completed tasks as they finish
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_path, status = future.result()
                files_processed[status].append(file_path)
            except (requests.RequestException, IOError) as request_error:
                print(f"Error processing {file_path}: {str(request_error)}")
                files_processed["error"].append(file_path)

    # Log summary
    LOG.info("\nSync Summary:")
    LOG.info(f"New files: {len(files_processed['new'])}")
    LOG.info(f"Updated files: {len(files_processed['updated'])}")
    LOG.info(f"Unchanged files: {len(files_processed['unchanged'])}")
    if files_processed["error"]:
        LOG.info(f"Failed files: {len(files_processed['error'])}")


def load_model_and_tokenizer(
    *,
    cfg: DictDefault,
    inference: bool = False,
) -> tuple[
    PreTrainedModel,
    PreTrainedTokenizer | PreTrainedTokenizerFast | Any,
    ProcessorMixin | None,
]:
    """
    Helper function for loading a model, tokenizer, and processor specified in the given `axolotl`
    config.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        inference: Boolean denoting inference mode.

    Returns:
        Tuple of (PreTrainedModel, PreTrainedTokenizer, ProcessorMixin).
    """
    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    LOG.info("loading model...")
    model, _ = load_model(cfg, tokenizer, inference=inference)

    processor = None
    if cfg.is_multimodal:
        LOG.info("loading processor...")
        processor = load_processor(cfg, tokenizer)

    return model, tokenizer, processor
