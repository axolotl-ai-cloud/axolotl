#!/bin/bash
# Run only the single-GPU e2e test files the selector picked (E2E_SELECTED_TESTS, comma-separated).
set -eo pipefail

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

: "${E2E_SELECTED_TESTS:?no selected tests}"
IFS=',' read -ra SELECTED <<< "$E2E_SELECTED_TESTS"

bash ./cicd/prepare_hf_cache.sh
env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

PATCHED=() SOLO=() REWARD=() INTEGRATIONS=() REST=()
for f in "${SELECTED[@]}"; do
  case "$f" in
    tests/e2e/multigpu/*|tests/e2e/kernels/*) ;;
    tests/e2e/patched/*) PATCHED+=("$f") ;;
    tests/e2e/solo/test_reward_model_smollm2.py) REWARD+=("$f") ;;
    tests/e2e/solo/*) SOLO+=("$f") ;;
    tests/e2e/integrations/*) INTEGRATIONS+=("$f") ;;
    tests/e2e/*) REST+=("$f") ;;
  esac
done

run() {
  local -n files=$1; shift
  [ ${#files[@]} -eq 0 ] && return 0
  echo "=== ${#files[@]} selected file(s): $*"
  pytest --durations=10 --maxfail=10 "$@" "${files[@]}"
}

# same process layout as cicd.sh, minus coverage
run PATCHED --full-trace -vvv
run SOLO -v -n1
run REWARD -v -s
run INTEGRATIONS -v
run REST -v
