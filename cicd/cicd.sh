#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

set -o pipefail
for i in 1 2 3; do
  if curl --silent --show-error --fail -L \
    https://axolotl-ci.b-cdn.net/hf-cache.tar.zst \
    | tar -xpf - -C "${HF_HOME}/hub/" --use-compress-program unzstd --strip-components=1; then
    echo "HF cache extracted successfully"
    break
  fi
  echo "Attempt $i failed, cleaning up and retrying in 15s..."
  rm -rf "${HF_HOME}/hub/"*
  sleep 15
done
# hf download "NousResearch/Meta-Llama-3-8B"
# hf download "NousResearch/Meta-Llama-3-8B-Instruct"
# hf download "microsoft/Phi-4-reasoning"
# hf download "microsoft/Phi-3.5-mini-instruct"
# hf download "microsoft/Phi-3-medium-128k-instruct"

# Run unit tests with initial coverage report
pytest -v --durations=10 -n8 \
  --ignore=tests/e2e/ \
  --ignore=tests/patched/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/ \
  --cov=axolotl

# Run lora kernels tests with coverage append
pytest -v --durations=10 \
  /workspace/axolotl/tests/e2e/patched/lora_kernels \
  --cov=axolotl \
  --cov-append

# Run patched tests excluding lora kernels with coverage append
pytest --full-trace -vvv --durations=10 \
  --ignore=tests/e2e/patched/lora_kernels \
  /workspace/axolotl/tests/e2e/patched \
  --cov=axolotl \
  --cov-append

# Run solo tests with coverage append
pytest -v --durations=10 -n1 \
  /workspace/axolotl/tests/e2e/solo/ \
  --cov=axolotl \
  --cov-append

# Run integration tests with coverage append
pytest -v --durations=10 \
  /workspace/axolotl/tests/e2e/integrations/ \
  --cov=axolotl \
  --cov-append

pytest -v --durations=10 /workspace/axolotl/tests/cli \
  --cov=axolotl \
  --cov-append

# Run remaining e2e tests with coverage append and final report
pytest -v --durations=10 \
  --ignore=tests/e2e/solo/ \
  --ignore=tests/e2e/patched/ \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/e2e/integrations/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/e2e/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:e2e-coverage.xml
