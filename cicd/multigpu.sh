#!/bin/bash
set -e

# only run one test at a time so as not to OOM the GPU
pytest -v  --durations=10 -n2 /workspace/axolotl/tests/e2e/multigpu/ --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/
pytest -v  --durations=10 -n1 /workspace/axolotl/tests/e2e/multigpu/solo/
