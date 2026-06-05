"""Helpers for CUDA 13 uv images."""

from __future__ import annotations

from pathlib import Path


def cu13_library_path() -> str | None:
    """Return the nvidia.cu13 package lib path, when available."""
    try:
        import nvidia.cu13 as cu13
    except ImportError:
        return None

    package_paths = list(cu13.__path__)
    if not package_paths:
        return None
    return str(Path(package_paths[0]) / "lib")


def prepend_cu13_ld_library_path(ld_library_path: str | None) -> str:
    """
    Prepend the CUDA 13 library directory when nvidia.cu13 is installed.

    The path is intentionally kept small and idempotent:
    - no-op when the cu13 package is absent
    - keeps any existing LD_LIBRARY_PATH entries
    - avoids duplicating the package-derived cu13 lib path
    """
    cu13_lib = cu13_library_path()
    if cu13_lib is None:
        return ld_library_path or ""

    parts = [cu13_lib]
    for part in (ld_library_path or "").split(":"):
        if part and part != cu13_lib and part not in parts:
            parts.append(part)

    return ":".join(parts)
