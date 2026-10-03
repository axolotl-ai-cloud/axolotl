#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

set -o pipefail
mkdir -p "${HF_HOME:?}/hub/"
for i in 1 2 3; do
  if curl --silent --show-error --fail -L \
    https://axolotl-ci.b-cdn.net/hf-cache.tar.zst \
    | tar -xpf - -C "${HF_HOME}/hub/" --use-compress-program unzstd --strip-components=1; then
    echo "HF cache extracted successfully"
    break
  fi
  if [ "$i" -eq 3 ]; then
    echo "HF cache download failed after 3 attempts" >&2
    exit 1
  fi
  echo "Attempt $i failed, cleaning up and retrying in 15s..."
  rm -rf "${HF_HOME:?}/hub/"*
  sleep 15
done

env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
pytest -v --durations=10 -n2 --maxfail=3 \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/ \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/patched/ \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl

pytest -v --durations=10 -n1 -m slow \
  /workspace/axolotl/tests/e2e/multigpu/test_ringmaster.py \
  -k test_axolotl_gdn_cp_parity \
  --cov=axolotl \
  --cov-append

# Run solo tests with coverage append
pytest -v --durations=10 -n1 \
  /workspace/axolotl/tests/e2e/multigpu/solo/ \
  --cov=axolotl \
  --cov-append

pytest -v --durations=10 -n1 \
  /workspace/axolotl/tests/integrations/test_expert_parallel_nvfp4_wrap.py \
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
