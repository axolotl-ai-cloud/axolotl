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

UNINSTALL_PREFIX = ""
if cce_spec:
    if not importlib.util.find_spec("cut_cross_entropy.transformers"):
        UNINSTALL_PREFIX = "pip uninstall -y cut-cross-entropy && "

print(
    UNINSTALL_PREFIX
    + 'pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@9c297c905f55b73594b5d650722d1e78183b77bd"'
)
