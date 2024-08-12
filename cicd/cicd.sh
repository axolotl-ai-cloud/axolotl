#!/bin/bash
set -e

pytest --ignore=tests/e2e/ /workspace/axolotl/tests/
pytest -n1 --dist loadfile -v /workspace/axolotl/tests/e2e/patched/
pytest --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ /workspace/axolotl/tests/e2e/
