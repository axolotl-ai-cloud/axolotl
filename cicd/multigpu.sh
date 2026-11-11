#!/bin/bash
set -e

export COVERAGE_CORE=sysmon

python -c "import torch; assert '$PYTORCH_VERSION' in torch.__version__, f'Expected torch $PYTORCH_VERSION but got {torch.__version__}'"

set -o pipefail
rm -f .coverage multigpu-coverage.xml
bash ./cicd/prepare_hf_cache.sh

env -u CODECOV_TOKEN python -c "from kernels import get_kernel; get_kernel(\"kernels-community/flash-attn2\", version=4, trust_remote_code=True)"

python cicd/feature_command.py multigpu-quantization pytest -v --durations=10 -n1 --maxfail=10 -m gpu \
  /workspace/axolotl/tests/monkeypatch/test_fsdp2_bnb_checkpoint.py \
  /workspace/axolotl/tests/integrations/test_expert_parallel_bnb.py \
  --cov=axolotl

python cicd/feature_command.py multigpu-cce pytest -v --durations=10 -n2 --maxfail=10 \
  tests/e2e/multigpu/test_cce_lora_lm_head.py --cov=axolotl --cov-append

# Only run two tests at a time to avoid OOM on GPU (with coverage collection)
python cicd/feature_command.py multigpu-general pytest -v --durations=10 -n2 --maxfail=10 \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/test_cce_lora_lm_head.py \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/solo/ \
  --ignore=/workspace/axolotl/tests/e2e/multigpu/patched/ \
  /workspace/axolotl/tests/e2e/multigpu/ \
  --cov=axolotl \
  --cov-append

python cicd/feature_command.py multigpu-context-parallel pytest -v --durations=10 -n1 --maxfail=10 -m slow \
  /workspace/axolotl/tests/e2e/multigpu/test_ringmaster.py \
  -k test_axolotl_gdn_cp_parity \
  --cov=axolotl \
  --cov-append

# Run solo tests with coverage append; the NVFP4 suite has its own SM100 job
python cicd/feature_command.py multigpu-solo pytest -v --durations=10 -n1 --maxfail=10 -m "not nvfp4 and not slow" \
  '{suite_files}' \
  --cov=axolotl \
  --cov-append

python cicd/feature_command.py multigpu-solo-rl pytest -v --durations=10 -n1 --maxfail=10 -m "not nvfp4 and not slow" \
  '{suite_files}' \
  --cov=axolotl \
  --cov-append

python cicd/feature_command.py multigpu-wrap pytest -v --durations=10 -n1 --maxfail=10 -m gpu \
  /workspace/axolotl/tests/integrations/expert_parallel/test_nvfp4_wrap.py \
  --cov=axolotl \
  --cov-append

python cicd/feature_command.py multigpu-patched pytest -v  --durations=10 -n1 --maxfail=10 /workspace/axolotl/tests/e2e/multigpu/patched/ \
  --cov=axolotl \
  --cov-append

if [ -f .coverage ]; then
  coverage xml -o multigpu-coverage.xml
fi

# Upload coverage to Codecov if CODECOV_TOKEN is available
if [ -n "${CODECOV_TOKEN:-}" ] && [ -f multigpu-coverage.xml ]; then
  codecovcli upload-process -t "${CODECOV_TOKEN}" -f multigpu-coverage.xml -F multigpu,docker-tests,pytorch-${PYTORCH_VERSION} || true
fi
