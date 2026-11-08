#!/bin/bash
set -euo pipefail

case "${MULTIGPU_TEST_SUITE:-multigpu}" in
  multigpu) bash cicd/multigpu.sh ;;
  nf4|nvfp4)
    python cicd/feature_command.py "$MULTIGPU_TEST_SUITE" bash "cicd/$MULTIGPU_TEST_SUITE.sh"
    ;;
  gpu_slow_distributed|gpu_slow_four_rank)
    python cicd/feature_command.py "$MULTIGPU_TEST_SUITE" env TESTMON_FULL=true bash cicd/gpu_slow.sh
    ;;
  gpu_slow_kernels) bash cicd/gpu_slow.sh ;;
  *) echo "Unknown multi-GPU suite: $MULTIGPU_TEST_SUITE" >&2; exit 1 ;;
esac
