#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__"

pytest -v --durations=10 -n8 --ignore=tests/e2e/ --ignore=tests/patched/ /workspace/axolotl/tests/
# pytest -v --durations=10 -n8 --dist loadfile /workspace/axolotl/tests/patched/
pytest -v --durations=10 /workspace/axolotl/tests/e2e/patched/
pytest -v --durations=10 /workspace/axolotl/tests/e2e/integrations/
pytest -v --durations=10 --ignore=tests/e2e/patched/ --ignore=tests/e2e/multigpu/ --ignore=tests/e2e/integrations/ /workspace/axolotl/tests/e2e/

tests=$(pytest --collect-only -q tests/e2e/each)
for t in $tests; do
    pytest $t
done
