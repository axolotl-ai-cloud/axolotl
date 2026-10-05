#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

bash ./cicd/prepare_hf_cache.sh

env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

TESTMON_BASELINE_DIR="/workspace/data/testmon/${BASE_TAG:?}/single-gpu"
TESTMON_DIR="/workspace/axolotl/.testmon"
mkdir -p "$TESTMON_DIR"
if [ -d "$TESTMON_BASELINE_DIR" ]; then
  cp -a "$TESTMON_BASELINE_DIR"/. "$TESTMON_DIR"/
  echo "testmon: restored baseline from $TESTMON_BASELINE_DIR"
else
  echo "testmon: no baseline at $TESTMON_BASELINE_DIR, running everything"
fi

tm() {
  local name=$1; shift
  TESTMON_DATAFILE="$TESTMON_DIR/${name}.testmondata" pytest --testmon-forceselect --maxfail=10 "$@"
}

# testmon only traces the pytest process; these files run the code under test in child processes
SUBPROCESS_TESTS="/workspace/axolotl/tests/e2e/test_evaluate.py /workspace/axolotl/tests/e2e/test_preprocess.py /workspace/axolotl/tests/e2e/test_qwen.py /workspace/axolotl/tests/e2e/integrations/test_kd.py /workspace/axolotl/tests/e2e/test_quantization.py"

tm patched --full-trace -vvv --durations=10 /workspace/axolotl/tests/e2e/patched

tm solo -v --durations=10 -n1 \
  --ignore=tests/e2e/solo/test_reward_model_smollm2.py \
  /workspace/axolotl/tests/e2e/solo/

tm solo-reward-model -v --durations=10 -s \
  /workspace/axolotl/tests/e2e/solo/test_reward_model_smollm2.py

tm integrations -v --durations=10 \
  --ignore=tests/e2e/integrations/test_kd.py \
  /workspace/axolotl/tests/e2e/integrations/

tm e2e -v --durations=10 \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/e2e/solo/ \
  --ignore=tests/e2e/patched/ \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/e2e/integrations/ \
  --ignore=tests/cli \
  --ignore=tests/e2e/test_evaluate.py \
  --ignore=tests/e2e/test_preprocess.py \
  --ignore=tests/e2e/test_qwen.py \
  --ignore=tests/e2e/test_quantization.py \
  /workspace/axolotl/tests/e2e/

# shellcheck disable=SC2086
pytest -v --durations=10 --maxfail=10 $SUBPROCESS_TESTS

if [ "${GITHUB_REF:-}" = "refs/heads/main" ]; then
  mkdir -p "$TESTMON_BASELINE_DIR"
  cp -a "$TESTMON_DIR"/. "$TESTMON_BASELINE_DIR"/
  echo "testmon: saved baseline to $TESTMON_BASELINE_DIR"
fi
