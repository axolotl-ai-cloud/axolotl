"""Script to output the correct installation command for cut-cross-entropy."""

import importlib.util
import sys

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

uninstall_prefix = ""
if cce_spec:
    if not importlib.util.find_spec("cut_cross_entropy.transformers"):
        uninstall_prefix = "pip uninstall -y cut-cross-entropy && "

print(
    uninstall_prefix
    + 'pip install "cut-cross-entropy[transformers] @ git+https://github.com/apple/ml-cross-entropy.git@24fbe4b5dab9a6c250a014573613c1890190536c"'
)
