#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

set -o pipefail
for i in 1 2 3; do
  if curl --silent --show-error --fail -L \
    https://axolotl-ci.b-cdn.net/hf-cache.tar.zst \
    | tar -xpf - -C "${HF_HOME}/hub/" --use-compress-program unzstd --strip-components=1; then
    echo "HF cache extracted successfully"
    break
  fi
  echo "Attempt $i failed, cleaning up and retrying in 15s..."
  rm -rf "${HF_HOME}/hub/"*
  sleep 15
done

pytest -v --durations=10 \
  /workspace/axolotl/tests/integrations/kernels/ \
  /workspace/axolotl/tests/integrations/monkeypatch/test_tiled_mlp_moe.py \
  /workspace/axolotl/tests/integrations/test_gemma4_moe.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_lora.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_lora_kernels.py \
  /workspace/axolotl/tests/integrations/test_scattermoe_multi_lora.py \
  /workspace/axolotl/tests/integrations/test_sonicmoe_multi_lora.py \
  --cov=axolotl \
  --cov-report=xml:e2e-kernel-coverage.xml

codecov upload-process -t "$CODECOV_TOKEN" -f e2e-kernel-coverage.xml -F e2e,kernels,pytorch-${PYTORCH_VERSION} || true
