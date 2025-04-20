#!/bin/bash
set -e

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
pytest -v -n2 \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/ \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/patched/ \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl \
  --cov-report=xml:multigpu-coverage.xml

pytest -v  --durations=10 -n1 /workspace/axolotl/tests/e2e/multigpu/solo/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:multigpu-coverage.xml

pytest -v  --durations=10 -n1 /workspace/axolotl/tests/e2e/multigpu/patched/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:multigpu-coverage.xml

# Upload coverage to Codecov
if [ -f multigpu-coverage.xml ]; then
  codecov -f multigpu-coverage.xml -F multigpu,docker-tests,pytorch-${PYTORCH_VERSION}
else
  echo "Coverage file not found. Coverage report may have failed."
fi
