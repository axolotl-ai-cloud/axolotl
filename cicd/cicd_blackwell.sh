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

status=0
pytest -v --durations=10 --maxfail=10 -m "gpu_blackwell and not slow" \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/ \
  --junitxml=blackwell-junit.xml \
  --cov=axolotl --cov-report=xml:e2e-blackwell-coverage.xml || status=$?

# Some skips are expected (sm120-only tests on B200); a skip-only run means a missing dependency.
python - <<'BLACKWELL_JUNIT' || status=$((status ? status : 1))
import collections
import xml.etree.ElementTree as ET

cases = list(ET.parse("blackwell-junit.xml").iter("testcase"))
skips = collections.defaultdict(list)
for case in cases:
    skipped = case.find("skipped")
    if skipped is not None:
        reason = skipped.get("message") or (skipped.text or "").strip() or "<no reason>"
        skips[reason].append(f"{case.get('classname')}::{case.get('name')}")
n_skipped = sum(map(len, skips.values()))
print(f"gpu_blackwell: {len(cases) - n_skipped} ran, {n_skipped} skipped", flush=True)
for reason, names in sorted(skips.items(), key=lambda kv: -len(kv[1])):
    print(f"  skipped {len(names)}x: {reason}", flush=True)
    for name in names:
        print(f"    {name}", flush=True)
assert len(cases) > n_skipped, f"all {len(cases)} gpu_blackwell tests skipped on this GPU"
BLACKWELL_JUNIT

if [ -n "${CODECOV_TOKEN:-}" ]; then
  codecovcli upload-process -t "$CODECOV_TOKEN" -f e2e-blackwell-coverage.xml \
    -F "e2e,blackwell,pytorch-${PYTORCH_VERSION}" || true
fi

exit "$status"
