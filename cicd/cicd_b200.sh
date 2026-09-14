#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

# Certified launch environment for the B200 factored LoRA MLP kernel: cuBLAS
# reads this once at handle creation, and the launch-budget goldens were
# certified under it (workspace size influences cuBLAS algorithm selection).
export CUBLAS_WORKSPACE_CONFIG=":4096:2"

# No HF cache download: the B200 suite uses raw tensors and tiny in-memory
# models only.

pytest -v --durations=10 \
  -m b200 \
  /workspace/axolotl/tests/e2e/kernels/test_blackwell_lora_mlp.py \
  --cov=axolotl \
  --cov-report=xml:e2e-b200-coverage.xml

codecov upload-process -t "$CODECOV_TOKEN" -f e2e-b200-coverage.xml -F e2e,b200,pytorch-${PYTORCH_VERSION} || true
