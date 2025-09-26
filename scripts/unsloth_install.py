"""Emit the uv commands needed to install Unsloth without touching torch."""

from __future__ import annotations

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch via `pip install torch`") from exc

from packaging.version import Version as V

torch_version = V(torch.__version__.split("+")[0])

# Unsloth supports torch >= 2.6.0 via the 2025.9 builds.
MIN_TORCH = V("2.6.0")

if torch_version < MIN_TORCH:
    raise RuntimeError(
        f"Torch {torch.__version__} detected, but Unsloth requires >= {MIN_TORCH}."
    )

commands = (
    "uv pip install --system --no-deps unsloth-zoo==2025.9.12 && "
    'uv pip install --system --no-deps "unsloth[huggingface]==2025.9.9"'
)

print(commands)
