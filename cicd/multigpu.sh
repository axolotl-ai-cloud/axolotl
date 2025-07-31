#!/bin/bash
set -e

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
pytest -v --durations=10 -n2 \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/ \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/patched/ \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl

# Run solo tests with coverage append
pytest -v --durations=10 -n1 \
  /workspace/axolotl/tests/e2e/multigpu/solo/ \
  --cov=axolotl \
  --cov-append

pytest -v  --durations=10 -n1 /workspace/axolotl/tests/e2e/multigpu/patched/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:multigpu-coverage.xml

# Upload coverage to Codecov if CODECOV_TOKEN is available
if [ -n "$CODECOV_TOKEN" ]; then
  codecov upload-process -t "${CODECOV_TOKEN}" -f multigpu-coverage.xml -F multigpu,docker-tests,pytorch-${PYTORCH_VERSION} || true
fi
