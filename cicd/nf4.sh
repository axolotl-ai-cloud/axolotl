#!/bin/bash
set -euo pipefail

case "${NF4_TEST_STACK:-pinned}" in
  pinned|nightly) ;;
  *) echo "Unknown NF4_TEST_STACK: $NF4_TEST_STACK" >&2; exit 1 ;;
esac

python - <<'PY'
import importlib.metadata
import os

import torch

assert os.environ["PYTORCH_VERSION"] in torch.__version__, torch.__version__
assert torch.cuda.device_count() >= 3, "NF4 CUDA coverage requires at least three GPUs"
for rank in range(3):
    assert torch.cuda.get_device_properties(rank).major >= 8, "BF16 CUDA support required"
    assert torch.cuda.mem_get_info(rank)[0] >= 16 * 1024**3, "NF4 large-tensor tests need 16 GiB free"
for name in ("torch", "transformers", "accelerate", "torchao", "bitsandbytes", "peft"):
    print(f"{name}=={importlib.metadata.version(name)}", flush=True)
PY

# Unlike multigpu.sh this lane never fetches the HF cache: every selected test builds its
# own checkpoint. Offline mode keeps a stray Hub dependency a hard error, not a silent skip.
export HF_HUB_OFFLINE=1

# Each test owns its GPUs and starts its own distributed workers.
pytest -v --durations=10 -n0 -m slow -k cuda \
  --confcutdir=tests/monkeypatch tests/monkeypatch/test_nf4_loading.py \
  --junitxml=nf4-junit.xml --cov=axolotl --cov-report=xml:nf4-coverage.xml

python - <<'PY'
import xml.etree.ElementTree as ET

report = ET.parse("nf4-junit.xml")
assert list(report.iter("testcase")), "No NF4 CUDA tests ran"
skipped = [node.get("message", "") for node in report.iter("skipped")]
assert not skipped, (
    f"NF4 GPU CI unexpectedly skipped {len(skipped)} tests: {skipped}. "
    "If a test now needs a Hub asset, this lane must extract hf-cache.tar.zst the way "
    "cicd/multigpu.sh does and drop HF_HUB_OFFLINE."
)
PY

if [ -n "${CODECOV_TOKEN:-}" ]; then
  codecov upload-process -t "$CODECOV_TOKEN" -f nf4-coverage.xml \
    -F nf4,multigpu,docker-tests,pytorch-${PYTORCH_VERSION} || true
fi
