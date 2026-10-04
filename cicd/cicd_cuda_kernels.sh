#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

bash ./cicd/prepare_hf_cache.sh

# Keep process-wide model patching isolated from kernel parity tests.
pytest -v --durations=10 --maxfail=10 \
  /workspace/axolotl/tests/e2e/kernels/lora_patching/ \
  --cov=axolotl

pytest -v --durations=10 --maxfail=10 \
  --ignore=tests/e2e/kernels/lora_patching/ \
  /workspace/axolotl/tests/e2e/kernels/ \
  /workspace/axolotl/tests/integrations/kernels/ \
  /workspace/axolotl/tests/integrations/monkeypatch/test_tiled_mlp_moe.py \
  /workspace/axolotl/tests/integrations/test_gemma4_moe.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_lora.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_lora_kernels.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_multi_lora.py \
  /workspace/axolotl/tests/integrations/test_sonicmoe_multi_lora.py \
  --cov=axolotl \
  --cov-append

pytest -v --durations=10 --maxfail=10 -m slow \
  /workspace/axolotl/tests/monkeypatch/test_fla_mamba.py \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:e2e-kernel-coverage.xml

codecov upload-process -t "$CODECOV_TOKEN" -f e2e-kernel-coverage.xml -F e2e,kernels,pytorch-${PYTORCH_VERSION} || true
