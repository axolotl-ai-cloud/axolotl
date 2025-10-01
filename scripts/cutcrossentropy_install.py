"""Script to output the correct installation command for cut-cross-entropy."""

import importlib.util
import sys
from shlex import quote

try:
    import torch
except ImportError as exc:
    raise ImportError("Install torch via `pip install torch`") from exc
from packaging.version import Version as V

v = V(torch.__version__)

# no cut-cross-entropy support for torch < 2.4.0
if v < V("2.4.0"):
    print("")
    sys.exit(0)

cce_spec = importlib.util.find_spec("cut_cross_entropy")

python_path = quote(sys.executable)

commands = []
if cce_spec and not importlib.util.find_spec("cut_cross_entropy.transformers"):
    commands.append(f"uv pip uninstall --python {python_path} cut-cross-entropy")

commands.append(
    f'uv pip install --python {python_path} "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@147ea28"'
)

print(" && ".join(commands))
