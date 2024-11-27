"""Script to output the correct installation command for cut-cross-entropy."""
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

print(
    'pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git@9c297c905f55b73594b5d650722d1e78183b77bd"'
)
