#!/bin/bash
# Create a virtualenv for running Axolotl's GPU tests on Intel GPUs (XPU). Run from the repository root:
#   bash cicd/setup_xpu_env.sh [venv-dir]     (default: .venv-xpu; needs uv)
set -euo pipefail

VENV="${1:-.venv-xpu}"
PYTORCH_VERSION="${PYTORCH_VERSION:-2.14.0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

uv venv --python "$PYTHON_VERSION" "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install "torch==${PYTORCH_VERSION}" torchvision --index-url https://download.pytorch.org/whl/xpu

uv pip install packaging==26.0 setuptools==78.1.1

# pin the XPU torch build so the axolotl install cannot swap it for the CUDA wheel from PyPI;
# no build isolation, because isolated builds (deepspeed) cannot resolve the +xpu torch from PyPI
TORCH_PIN="torch==$(python -c 'import torch; print(torch.__version__)')"
uv pip install --no-build-isolation --overrides cicd/xpu-overrides.txt -e . "$TORCH_PIN" \
    "deepspeed>=0.18.6,<0.20.0" --group test blobfile tiktoken pytest-timeout

python - <<'XPU_ENV'
import importlib.metadata as md

import torch

assert torch.xpu.is_available(), "torch cannot see an Intel XPU"
print(f"torch {torch.__version__}: {torch.xpu.device_count()} x {torch.xpu.get_device_name(0)}", flush=True)
for dist in ("triton-xpu", "transformers", "peft", "accelerate", "trl", "deepspeed", "bitsandbytes", "kernels"):
    print(f"{dist}=={md.version(dist)}", flush=True)
XPU_ENV
