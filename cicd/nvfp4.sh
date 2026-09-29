#!/bin/bash
set -euo pipefail

python - <<'NVFP4_ENV'
import importlib.metadata
import os

import torch

assert os.environ["PYTORCH_VERSION"] in torch.__version__, torch.__version__
assert torch.cuda.device_count() >= 2, "NVFP4 CUDA coverage requires two GPUs"
for rank in range(2):
    assert torch.cuda.get_device_properties(rank).major >= 10, (
        "dynamic NVFP4 coverage requires SM100+ GPUs"
    )
for name in ("torch", "transformers", "accelerate", "torchao", "deepspeed", "peft"):
    print(f"{name}=={importlib.metadata.version(name)}", flush=True)
NVFP4_ENV

pytest -v --durations=10 -n0 \
  tests/e2e/multigpu/solo/test_native_nvfp4_dynamic_ste.py \
  tests/e2e/multigpu/solo/test_sonicmoe_nvfp4_fsdp2.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_fsdp2_lora_parity.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_fsdp2_merge_aware_resume.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_fsdp2_recipe.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_tp_hf_load.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_tp_lora_parity.py \
  tests/e2e/multigpu/solo/test_torchao_lora_deepspeed.py \
  tests/e2e/multigpu/solo/test_torchao_lora_deepspeed_checkpoint.py \
  tests/e2e/multigpu/solo/test_torchao_lora_deepspeed_zero3_checkpoint.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_deepspeed_lora_merge_aware.py \
  tests/e2e/multigpu/solo/test_native_nvfp4_deepspeed_lora_merge_lifecycle.py \
  --junitxml=nvfp4-junit.xml --cov=axolotl --cov-report=xml:nvfp4-coverage.xml

python - <<'NVFP4_JUNIT'
import xml.etree.ElementTree as ET

report = ET.parse("nvfp4-junit.xml")
assert list(report.iter("testcase")), "No NVFP4 GPU tests ran"
skipped = [node.get("message", "") for node in report.iter("skipped")]
assert not skipped, f"NVFP4 GPU CI unexpectedly skipped {len(skipped)} tests: {skipped}"
NVFP4_JUNIT

if [ -n "${CODECOV_TOKEN:-}" ]; then
  codecov upload-process -t "$CODECOV_TOKEN" -f nvfp4-coverage.xml \
    -F nvfp4,multigpu,docker-tests,pytorch-${PYTORCH_VERSION} || true
fi
