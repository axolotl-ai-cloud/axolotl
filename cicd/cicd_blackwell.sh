#!/bin/bash
# Single-GPU tests that cannot run on the L40S lanes (pytest -m gpu_blackwell, see tests/conftest.py).
set -euo pipefail

# Blackwell-only kernel stack, pinned to docker/Dockerfile-cloud-blackwell-uv; the shared CI
# image stays L40S-sized. torch is pinned so the resolver cannot swap the image's build.
# DeepEP comes from cicd/deep_ep.py's cached wheel, installed before this script runs.
torch_version=$(python -c "import torch; print(torch.__version__)")
uv pip install --python "$(command -v python)" \
  "torch==${torch_version}" \
  "flash-attn-4[cu13]==4.0.0b35" \
  "quack-kernels[cu13]==0.6.5"
# a missing package here would turn the lane into a skip-only run
python - <<'BLACKWELL_DEPS'
import importlib.metadata as md

missing = []
for dist in (
    "deep-ep",
    "flash-attn-4",
    "quack-kernels",
    "nvidia-cutlass-dsl",
    "triton",
    "torchao",
):
    try:
        print(f"{dist}=={md.version(dist)}", flush=True)
    except md.PackageNotFoundError:
        missing.append(dist)
assert not missing, f"Blackwell stack is missing {missing}"

import deep_ep  # noqa: F401  # the wheel links nvshmem and torch; import catches ABI drift
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
