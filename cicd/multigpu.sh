#!/bin/bash
set -e

# only run one test at a time so as not to OOM the GPU
pytest -v -n2 /workspace/axolotl/tests/e2e/multigpu/
