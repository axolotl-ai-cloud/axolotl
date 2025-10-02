"""Emit the install commands for Unsloth without altering torch."""

from __future__ import annotations

import shutil
import sys
from shlex import quote

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch via `pip install torch`") from exc

from packaging.version import Version as V

MIN_TORCH = V("2.6.0")

if V(torch.__version__.split("+")[0]) < MIN_TORCH:
    raise RuntimeError(
        f"Torch {torch.__version__} detected, but Unsloth requires >= {MIN_TORCH}."
    )

USE_UV_FLAG = "--uv" in sys.argv[1:]
USE_PIP_FLAG = "--pip" in sys.argv[1:]

if USE_UV_FLAG and USE_PIP_FLAG:
    raise SystemExit("Specify only one of --uv or --pip")

if USE_PIP_FLAG:
    use_uv = False
elif USE_UV_FLAG:
    use_uv = True
else:
    use_uv = shutil.which("uv") is not None

python_exe = quote(sys.executable or shutil.which("python3") or "python")

if use_uv:
    installer = "uv pip install --system --no-deps"
else:
    installer = f"{python_exe} -m pip install --no-deps"

commands = [
    f"{installer} unsloth-zoo==2025.9.12",
    f'{installer} "unsloth[huggingface]==2025.9.9"',
]

print(" && ".join(commands))
