#!/bin/bash
set -e

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

bash ./cicd/prepare_hf_cache.sh
# hf download "NousResearch/Meta-Llama-3-8B"
# hf download "NousResearch/Meta-Llama-3-8B-Instruct"
# hf download "microsoft/Phi-4-reasoning"
# hf download "microsoft/Phi-3.5-mini-instruct"
# hf download "microsoft/Phi-3-medium-128k-instruct"

env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=3, trust_remote_code=True)"

# Run unit tests with initial coverage report
pytest -v --durations=10 -n8 --maxfail=10 -m "gpu and not slow and not nf4_distributed" \
  --ignore=tests/e2e/ \
  --ignore=tests/integrations/ \
  --ignore=tests/patched/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/ \
  --cov=axolotl

# Run patched training tests with coverage append
pytest --full-trace -vvv --durations=10 --maxfail=10 \
  /workspace/axolotl/tests/e2e/patched \
  --cov=axolotl \
  --cov-append

# Run solo tests with coverage append
# test_rm_lora is run in its own process below (it fails on py3.11 when sharing
# the solo process with other tests; isolating it avoids cross-test state).
pytest -v --durations=10 --maxfail=10 -n1 \
  --ignore=tests/e2e/solo/test_reward_model_smollm2.py \
  /workspace/axolotl/tests/e2e/solo/ \
  --cov=axolotl \
  --cov-append

# Run reward-model test isolated in its own process
pytest -v --durations=10 --maxfail=10 -s \
  /workspace/axolotl/tests/e2e/solo/test_reward_model_smollm2.py \
  --cov=axolotl \
  --cov-append

# Run E2E integration tests with coverage append
pytest -v --durations=10 --maxfail=10 \
  /workspace/axolotl/tests/e2e/integrations/ \
  --cov=axolotl \
  --cov-append

pytest -v --durations=10 -n8 --dist loadfile --maxfail=10 -m gpu \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/integrations/kernels/ \
  --ignore=tests/integrations/monkeypatch/test_tiled_mlp_moe.py \
  --ignore=tests/integrations/test_gemma4_moe.py \
  --ignore=tests/integrations/test_scattermoe_lora.py \
  --ignore=tests/integrations/test_scattermoe_lora_kernels.py \
  --ignore=tests/integrations/test_scattermoe_multi_lora.py \
  --ignore=tests/integrations/test_sonicmoe_multi_lora.py \
  /workspace/axolotl/tests/integrations/ \
  --cov=axolotl \
  --cov-append

# Run remaining e2e tests with coverage append and final report
pytest -v --durations=10 --maxfail=10 \
  --ignore=tests/e2e/kernels/ \
  --ignore=tests/e2e/solo/ \
  --ignore=tests/e2e/patched/ \
  --ignore=tests/e2e/multigpu/ \
  --ignore=tests/e2e/integrations/ \
  --ignore=tests/cli \
  /workspace/axolotl/tests/e2e/ \
  --cov=axolotl \
  --cov-append \
  --cov-report=xml:e2e-coverage.xml

codecov upload-process -t $CODECOV_TOKEN -f e2e-coverage.xml -F e2e,pytorch-${PYTORCH_VERSION} || true
