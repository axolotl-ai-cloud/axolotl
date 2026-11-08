#!/bin/bash
set -e

export COVERAGE_CORE=sysmon

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

bash ./cicd/prepare_hf_cache.sh

env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=4, trust_remote_code=True)"

baseline_stack=pinned
if [ "${NIGHTLY_BUILD:-}" = "true" ]; then
  baseline_stack=nightly
fi
TESTMON_BASELINE_DIR="/workspace/data/testmon/${BASE_TAG:?}/${baseline_stack}/single-gpu"
TESTMON_DIR="/workspace/axolotl/.testmon"
mkdir -p "$TESTMON_DIR"
if [ -d "$TESTMON_BASELINE_DIR" ]; then
  cp -a "$TESTMON_BASELINE_DIR"/. "$TESTMON_DIR"/
  echo "testmon: restored baseline from $TESTMON_BASELINE_DIR"
else
  echo "testmon: no baseline at $TESTMON_BASELINE_DIR, running everything"
fi

# Every block runs even after a failure, so the baseline covers all of them
testmon_option=--testmon-forceselect
if [ "${TESTMON_FULL:-}" = "true" ] || [ "${TESTMON_SELECT:-true}" = "false" ]; then
  testmon_option=--testmon-noselect
fi
failed=0
tm() {
  local name=$1; shift
  TESTMON_DATAFILE="$TESTMON_DIR/${name}.testmondata" pytest "$testmon_option" --maxfail=10 "$@" || failed=1
}

# testmon only traces the pytest process; these files run the code under test in child processes
SUBPROCESS_TESTS="/workspace/axolotl/tests/e2e/test_evaluate.py /workspace/axolotl/tests/e2e/test_preprocess.py /workspace/axolotl/tests/e2e/test_qwen.py /workspace/axolotl/tests/e2e/integrations/test_kd.py /workspace/axolotl/tests/e2e/test_quantization.py"

tm gpu-unit -v --durations=10 -n8 -m "gpu and not slow and not nf4_distributed and not gpu_blackwell" \
  --ignore=tests/e2e/ \
  --ignore=tests/integrations/ \
  --ignore=tests/patched/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/

tm patched --full-trace -vvv --durations=10 /workspace/axolotl/tests/e2e/patched

tm solo -v --durations=10 -n1 \
  --ignore=tests/e2e/solo/test_reward_model_smollm2.py \
  /workspace/axolotl/tests/e2e/solo/

tm solo-reward-model -v --durations=10 -s \
  /workspace/axolotl/tests/e2e/solo/test_reward_model_smollm2.py

feature_full=${TESTMON_FULL:-false}
if [ "${GITHUB_REF:-}" = "refs/heads/main" ]; then
  feature_full=false
fi
export TESTMON_SELECT=${TESTMON_SELECT:-true}
TESTMON_FULL="$feature_full" python cicd/feature_testmon.py integrations -v --durations=10 --maxfail=10 || failed=1

tm gpu-integrations -v --durations=10 -n8 --dist loadfile -m "gpu and not gpu_blackwell" \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/integrations/kernels/ \
  /workspace/axolotl/tests/integrations/

tm e2e -v --durations=10 -m "not slow and not gpu_blackwell" \
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
pytest -v --durations=10 --maxfail=10 -m "not slow and not gpu_blackwell" $SUBPROCESS_TESTS || failed=1

# Saved after test failures too; testmon re-runs a failing test's file until it passes
if [ "${GITHUB_REF:-}" = "refs/heads/main" ]; then
  mkdir -p "$TESTMON_BASELINE_DIR"
  cp -a "$TESTMON_DIR"/. "$TESTMON_BASELINE_DIR"/
  echo "testmon: saved baseline to $TESTMON_BASELINE_DIR"
fi

exit $failed
