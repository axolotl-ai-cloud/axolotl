#!/bin/bash
# Single-GPU tests that cannot run on the L40S lanes (pytest -m gpu_blackwell, see tests/conftest.py).
set -euo pipefail

# The image installs axolotl's `blackwell` extra (AXOLOTL_EXTRAS) and cicd/deep_ep.py installs DeepEP
# before this runs; log what resolved. A missing one raises here instead of a skip-only run.
python - <<'BLACKWELL_DEPS'
import importlib.metadata as md

from packaging.requirements import Requirement

stack = [
    Requirement(r).name
    for r in md.requires("axolotl") or []
    if 'extra == "blackwell"' in r
]
for dist in [*stack, "nvidia-cutlass-dsl", "deep-ep", "triton", "torchao"]:
    print(f"{dist}=={md.version(dist)}", flush=True)
BLACKWELL_DEPS

python - <<'BLACKWELL_ENV'
import os

import torch

assert os.environ["PYTORCH_VERSION"] in torch.__version__, torch.__version__
major, minor = torch.cuda.get_device_capability()
assert major >= 10, f"gpu_blackwell tests need an SM100+ GPU, got sm{major}{minor}"
print(f"{torch.cuda.get_device_name()} sm{major}{minor}", flush=True)
BLACKWELL_ENV

bash ./cicd/prepare_hf_cache.sh

rm -f .coverage e2e-blackwell-coverage.xml blackwell-*-junit.xml
status=0
python cicd/feature_testmon.py blackwell -v --durations=10 --maxfail=10 \
  -m "gpu_blackwell and not slow" \
  --cov=axolotl --cov-append --cov-report=xml:e2e-blackwell-coverage.xml || status=$?

if [ -n "${CODECOV_TOKEN:-}" ] && [ -f e2e-blackwell-coverage.xml ]; then
  codecovcli upload-process -t "$CODECOV_TOKEN" -f e2e-blackwell-coverage.xml \
    -F "e2e,blackwell,pytorch-${PYTORCH_VERSION}" || true
fi

exit "$status"
