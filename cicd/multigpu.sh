#!/bin/bash
set -e

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
pytest -v -n2 \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl \
  --cov-report=xml:multigpu-coverage.xml

# Upload coverage to Codecov
if [ -f multigpu-coverage.xml ]; then
  codecov -f multigpu-coverage.xml -F multigpu,docker-tests,pytorch-${PYTORCH_VERSION}
else
  echo "Coverage file not found. Coverage report may have failed."
fi
