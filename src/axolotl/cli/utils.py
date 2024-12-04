"""
utility methods for axoltl CLI
"""
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import requests


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
