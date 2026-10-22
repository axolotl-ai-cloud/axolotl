#!/bin/bash
# Single-GPU tests that cannot run on the L40S lanes (pytest -m gpu_blackwell, see tests/conftest.py).
set -euo pipefail

python - <<'BLACKWELL_ENV'
import os

import torch

assert os.environ["PYTORCH_VERSION"] in torch.__version__, torch.__version__
major, minor = torch.cuda.get_device_capability()
assert major >= 10, f"gpu_blackwell tests need an SM100+ GPU, got sm{major}{minor}"
print(f"{torch.cuda.get_device_name()} sm{major}{minor}", flush=True)
BLACKWELL_ENV

bash ./cicd/prepare_hf_cache.sh

pytest -v --durations=10 --maxfail=10 -m "gpu_blackwell and not slow" \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/ \
  --junitxml=blackwell-junit.xml \
  --cov=axolotl --cov-report=xml:e2e-blackwell-coverage.xml

python - <<'BLACKWELL_JUNIT'
import xml.etree.ElementTree as ET

cases = list(ET.parse("blackwell-junit.xml").iter("testcase"))
ran = [case for case in cases if case.find("skipped") is None]
assert ran, f"all {len(cases)} gpu_blackwell tests skipped on this GPU"
print(f"gpu_blackwell: {len(ran)} ran, {len(cases) - len(ran)} skipped", flush=True)
BLACKWELL_JUNIT

if [ -n "${CODECOV_TOKEN:-}" ]; then
  codecovcli upload-process -t "$CODECOV_TOKEN" -f e2e-blackwell-coverage.xml \
    -F "e2e,blackwell,pytorch-${PYTORCH_VERSION}" || true
fi
