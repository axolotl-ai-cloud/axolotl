"""Script to output the correct installation command for cut-cross-entropy."""

import importlib.util
import sys

try:
    import torch
except ImportError as exc:
    raise ImportError("Install torch via `pip install torch`") from exc
from packaging.version import Version as V

USE_UV = "--uv" in sys.argv[1:]

v = V(torch.__version__)

# no cut-cross-entropy support for torch < 2.4.0
if v < V("2.4.0"):
    print("")
    sys.exit(0)

cce_spec = importlib.util.find_spec("cut_cross_entropy")

UNINSTALL_PREFIX = ""
if cce_spec:
    if not importlib.util.find_spec("cut_cross_entropy.transformers"):
        UNINSTALL_PREFIX = "pip uninstall -y cut-cross-entropy && "

UV_PREFIX = "uv " if USE_UV else ""

print(
    UNINSTALL_PREFIX
    + f'{UV_PREFIX}pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@f643b88"'
)
