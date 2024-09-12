#!/bin/bash
set -e

pytest --ignore=tests/e2e/ /workspace/axolotl/tests/
pytest -n1 --dist loadfile -v /workspace/axolotl/tests/e2e/patched/ /workspace/axolotl/tests/e2e/integrations/
pytest --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ --ignore=tests/e2e/integrations/ /workspace/axolotl/tests/e2e/
