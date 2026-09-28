"""Transfer closed training files without POSIX metadata operations on bucket mounts."""

import hashlib
import json
from pathlib import Path, PurePosixPath

MANIFEST = "nebius-checkpoint.json"


def copy_bytes(source, destination):
    """Stream bytes without explicit ftruncate, mmap or metadata preservation."""
    digest = hashlib.sha256()
    size = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return {"size": size, "sha256": digest.hexdigest()}


def publish(
    source, destination, manifest_name=MANIFEST, metadata=None, skip_checkpoints=False
):
    """Write a manifest only after all files in a closed snapshot were copied."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f"Missing local output directory: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / manifest_name
    if marker.exists():
        raise ValueError(f"Output snapshot already published: {destination}")
    files = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if skip_checkpoints and relative.parts[0].startswith("checkpoint-"):
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not supported in output snapshots")
        if path.is_file():
            if str(relative) == manifest_name:
                raise ValueError("Output contains a reserved manifest filename")
            files[relative.as_posix()] = copy_bytes(path, destination / relative)
    if not files:
        raise ValueError("Cannot publish an empty output snapshot")
    marker.write_text(
        json.dumps({"version": 1, "files": files, "training": metadata or {}}) + "\n",
        encoding="utf-8",
    )


def restore(source, destination):
    """Stage and verify a committed checkpoint before handing it to Trainer."""
    source, destination = Path(source), Path(destination)
    marker = source / MANIFEST
    if not marker.is_file():
        raise ValueError(
            "Checkpoint has no completion manifest; select a complete checkpoint from a successful save"
        )
    manifest = json.loads(marker.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("version") != 1
        or not isinstance(manifest.get("files"), dict)
        or not manifest["files"]
    ):
        raise ValueError("Invalid checkpoint manifest")
    if destination.exists():
        raise ValueError("Resume staging directory must not already exist")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or name == MANIFEST
            or not isinstance(expected, dict)
        ):
            raise ValueError("Invalid file path or entry in checkpoint manifest")
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe checkpoint file: {name}")
        actual = copy_bytes(path, destination / name)
        if actual != expected:
            raise ValueError(f"Checkpoint checksum or size mismatch: {name}")
    return destination
