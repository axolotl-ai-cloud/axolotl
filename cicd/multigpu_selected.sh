#!/bin/bash
# Run only the multi-GPU e2e test files the selector picked (E2E_SELECTED_TESTS, comma-separated).
set -eo pipefail

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

: "${E2E_SELECTED_TESTS:?no selected tests}"
IFS=',' read -ra SELECTED <<< "$E2E_SELECTED_TESTS"

bash ./cicd/prepare_hf_cache.sh
env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

SOLO=() PATCHED=() REST=()
for f in "${SELECTED[@]}"; do
  case "$f" in
    tests/e2e/multigpu/solo/*) SOLO+=("$f") ;;
    tests/e2e/multigpu/patched/*) PATCHED+=("$f") ;;
    tests/e2e/multigpu/*) REST+=("$f") ;;
  esac
done

run() {
  local -n files=$1; shift
  [ ${#files[@]} -eq 0 ] && return 0
  echo "=== ${#files[@]} selected file(s): $*"
  pytest -v --durations=10 --maxfail=10 "$@" "${files[@]}"
}

# same process layout as multigpu.sh, minus coverage; NVFP4 has its own SM100 job
run REST -n2
run SOLO -n1 -m "not nvfp4 and not slow"
run PATCHED -n1
