"""Utilities for axolotl fetch CLI command."""

import concurrent.futures
import hashlib
import json
from pathlib import Path

import click
import requests

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _download_file(
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
        status = "updated"
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
                _download_file,
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
