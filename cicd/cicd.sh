#!/bin/bash
set -e

pytest --ignore=tests/e2e/ --ignore=tests/integrations/e2e/ /workspace/axolotl/tests/
pytest -n1 --dist loadfile -v /workspace/axolotl/tests/e2e/patched/
pytest -n1 --dist loadfile -v /workspace/axolotl/tests/integrations/e2e/
pytest --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ /workspace/axolotl/tests/e2e/
