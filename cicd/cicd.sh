#!/bin/bash
set -e

pytest --durations=10 -n8 --ignore=tests/e2e/ /workspace/axolotl/tests/
pytest --durations=10 -n1 --dist loadfile -v /workspace/axolotl/tests/e2e/patched/ /workspace/axolotl/tests/e2e/integrations/
pytest --durations=10 --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ --ignore=tests/e2e/integrations/ /workspace/axolotl/tests/e2e/
