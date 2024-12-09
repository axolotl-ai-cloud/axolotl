#!/bin/bash
set -e

pytest -v --durations=10 -n8 --ignore=tests/e2e/ --ignore=tests/patched/ /workspace/axolotl/tests/
# pytest -v --durations=10 -n8 --dist loadfile /workspace/axolotl/tests/patched/
pytest -v --durations=10 -n1 --dist loadfile /workspace/axolotl/tests/e2e/patched/
pytest -v --durations=10 -n1 --dist loadfile /workspace/axolotl/tests/e2e/integrations/
pytest -v --durations=10 --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ --ignore=tests/e2e/integrations/ /workspace/axolotl/tests/e2e/
