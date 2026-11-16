#!/bin/bash
# Single-GPU tests on an Intel GPU (XPU): the paths of cicd/cicd.sh and cicd/cicd_cuda_kernels.sh, minus tests
# marked skip_xpu. Run from the repository root in an environment from cicd/setup_xpu_env.sh.
set -euo pipefail

# one card only (the first the runner exposes): with several visible, tests mix tensors across xpu:0 and xpu:N
ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK%%,*}"

python - <<'XPU_ENV'
import os

import torch

assert torch.xpu.is_available() and not torch.cuda.is_available(), "the XPU lane needs an Intel GPU and no CUDA"
expected = os.environ.get("PYTORCH_VERSION")
assert not expected or expected in torch.__version__, torch.__version__
assert torch.xpu.device_count() == 1, f"the XPU lane needs exactly one visible GPU, got {torch.xpu.device_count()}"
print(f"torch {torch.__version__}: {torch.xpu.get_device_name(0)} (ZE_AFFINITY_MASK={os.environ['ZE_AFFINITY_MASK']})", flush=True)
XPU_ENV

# a self-hosted runner keeps its own HF_HOME, so the CI cache archive is not fetched
python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

status=0
run() {
  local rc=0
  # thread method: a test stuck in the GPU driver never returns to Python to take the signal
  pytest -v --durations=10 --maxfail=10 --timeout=900 --timeout-method=thread "$@" || rc=$?
  # exit 5: every selected test in the block is marked skip_xpu
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 5 ]; then
    status=$rc
  fi
}

# no xdist workers: they stalled on a 4 x B60 host
run -m "gpu and not slow and not nf4_distributed and not gpu_blackwell and not skip_xpu" \
  --ignore=tests/e2e/ \
  --ignore=tests/integrations/ \
  --ignore=tests/patched/ \
  --ignore=tests/cli \
  tests/

run --full-trace -m "not slow and not skip_xpu" tests/e2e/patched

run -n1 -m "not slow and not skip_xpu" \
  --ignore=tests/e2e/solo/test_reward_model_smollm2.py \
  tests/e2e/solo/

run -s -m "not slow and not skip_xpu" tests/e2e/solo/test_reward_model_smollm2.py

run -m "not slow and not skip_xpu" tests/e2e/integrations/

run -m "gpu and not gpu_blackwell and not skip_xpu" \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/integrations/kernels/ \
  tests/integrations/

run -m "not slow and not gpu_blackwell and not skip_xpu" \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/e2e/solo/ \
  --ignore=tests/e2e/patched/ \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/e2e/integrations/ \
  --ignore=tests/cli \
  tests/e2e/

run -m "not slow and not skip_xpu" tests/e2e/kernels/lora_patching/

run -m "not slow and not gpu_blackwell and not skip_xpu" \
  --ignore=tests/e2e/kernels/lora_patching/ \
  tests/e2e/kernels/ \
  tests/integrations/kernels/ \
  tests/integrations/monkeypatch/test_tiled_mlp_moe.py

run -m "slow and not skip_xpu" tests/monkeypatch/test_fla_mamba.py

exit "$status"
