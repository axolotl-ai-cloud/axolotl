#!/bin/bash
set -euo pipefail

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, torch.__version__"

case "${MULTIGPU_TEST_SUITE}" in
  gpu_slow_kernels)
    required_gpus=1
    tests=(tests/e2e/kernels/ tests/utils/lora/test_merge_lora.py)
    selection=(--ignore=tests/e2e/kernels/lora_patching/)
    bash ./cicd/prepare_hf_cache.sh
    ;;
  gpu_slow_distributed)
    required_gpus=2
    tests=(tests/e2e/multigpu/test_ringmaster.py)
    selection=(-k "not test_axolotl_gdn_cp_parity and not test_fla_cp_four_rank_parity")
    ;;
  gpu_slow_four_rank)
    required_gpus=4
    tests=(tests/e2e/multigpu/test_ringmaster.py)
    selection=(-k test_fla_cp_four_rank_parity)
    ;;
  *) echo "Unknown slow GPU suite: $MULTIGPU_TEST_SUITE" >&2; exit 1 ;;
esac

python -c "import torch; assert torch.cuda.device_count() >= $required_gpus, 'Insufficient CUDA devices'"
export OMP_NUM_THREADS=1

pytest -v --durations=10 -n0 -m "gpu and slow" \
  "${selection[@]}" "${tests[@]}" \
  --cov=axolotl --cov-report="xml:${MULTIGPU_TEST_SUITE}-coverage.xml"

if [ -n "${CODECOV_TOKEN:-}" ]; then
  codecovcli upload-process -t "$CODECOV_TOKEN" \
    -f "${MULTIGPU_TEST_SUITE}-coverage.xml" \
    -F "gpu-slow,${MULTIGPU_TEST_SUITE},pytorch-${PYTORCH_VERSION}" || true
fi
