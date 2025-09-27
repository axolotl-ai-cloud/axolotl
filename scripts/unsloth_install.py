"""Emit the uv commands needed to install Unsloth without touching torch."""

from __future__ import annotations

import sys
from shlex import quote

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch via `pip install torch`") from exc

from packaging.version import Version as V

MIN_TORCH = V("2.6.0")

python_version = V(torch.__version__.split("+")[0])
if python_version < MIN_TORCH:
    raise RuntimeError(
        f"Torch {torch.__version__} detected, but Unsloth requires >= {MIN_TORCH}."
    )

python_path = quote(sys.executable)
commands = (
    f"uv pip install --python {python_path} --no-deps unsloth-zoo==2025.9.12 && "
    f'uv pip install --python {python_path} --no-deps "unsloth[huggingface]==2025.9.9"'
)

print(commands)
