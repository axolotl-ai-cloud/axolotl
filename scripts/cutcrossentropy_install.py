"""Print the pip command to install Axolotl's cut_cross_entropy fork."""

from __future__ import annotations

import sys

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch via `pip install torch`") from exc

from packaging.version import Version as V

if V(torch.__version__.split("+")[0]) < V("2.6.0"):
    print("")
    sys.exit(0)

print(
    "python -m pip install "
    '"cut-cross-entropy[transformers] '
    '@ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@147ea28"'
)
