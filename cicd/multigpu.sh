#!/bin/bash
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

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
pytest -v --durations=10 -n2 --maxfail=3 \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/ \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/patched/ \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl

# Run solo tests with coverage append
pytest -v --durations=10 -n1 \
  /workspace/axolotl/tests/e2e/multigpu/solo/ \
  --cov=axolotl \
  --cov-append

pytest -v  --durations=10 -n1 /workspace/axolotl/tests/e2e/multigpu/patched/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:multigpu-coverage.xml

# Upload coverage to Codecov if CODECOV_TOKEN is available
if [ -n "$CODECOV_TOKEN" ]; then
  codecov upload-process -t "${CODECOV_TOKEN}" -f multigpu-coverage.xml -F multigpu,docker-tests,pytorch-${PYTORCH_VERSION} || true
fi
