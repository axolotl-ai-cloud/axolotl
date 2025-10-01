"""Emit the install command for the Axolotl cut-cross-entropy fork."""

from __future__ import annotations

import shutil
import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover - defensive
    raise ImportError("Install torch via `pip install torch`") from exc

from packaging.version import Version as V

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

if V(torch.__version__) < V("2.4.0"):
    print("")
    sys.exit(0)

# No need to uninstall in CI runs; the environment is fresh. Just emit the install command.
installer = "uv pip install --system" if use_uv else "pip install"
command = (
    f"{installer} "
    '"cut-cross-entropy[transformers] '
    '@ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@147ea28"'
)

print(command)
