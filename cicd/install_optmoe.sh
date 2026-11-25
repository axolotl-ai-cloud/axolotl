#!/bin/bash
# The optmoe package and its compiled kernel extension (a prebuilt wheel for this Python, torch
# and CUDA version from the optmoe GitHub release).
set -euo pipefail

uv pip install --python "$(command -v python)" ".[optmoe]"
python -m optmoe.install_kernels
