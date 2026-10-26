# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Run ``nemotron_3_5_lightning.py`` for a few steps on Modal.

One container with two H200s: the vLLM server on the first, training on the second.
The Hugging Face cache (weights and hub kernels) lives on a Modal volume, so a rerun
skips the download::

    modal run examples/modal_nemotron_3_5_lightning.py --max-steps 3
"""

from __future__ import annotations

import os
import shlex
import subprocess  # nosec B404
import time
import urllib.request
from pathlib import Path

import modal

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
PACKAGE = Path(__file__).resolve().parents[1]
REMOTE_PACKAGE = "/root/tree-recurrence"
HF_CACHE = "/root/.cache/huggingface"
SERVER = "http://localhost:8000"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.30.0")
    .uv_pip_install(
        "trl==1.16.0",
        "peft",
        "kernels==0.17.2",
        "math-verify[antlr4_13_2]",
        "datasets",
        "accelerate",
    )
    # vLLM caps transformers below the release the hooks are tested against
    .uv_pip_install("transformers==5.20.1", extra_options="--no-deps")
    .env({"TRL_EXPERIMENTAL_SILENCE": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(
        PACKAGE,
        REMOTE_PACKAGE,
        copy=True,
        ignore=["**/__pycache__", "**/*.egg-info", "**/.pytest_cache", ".venv"],
    )
    .run_commands(f"python -m pip install --no-deps {REMOTE_PACKAGE}")
)
hf_cache = modal.Volume.from_name("tree-recurrence-hf-cache", create_if_missing=True)
app = modal.App("tree-recurrence-nemotron", image=image)


def _wait_for(server: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vLLM exited with {server.returncode}")
        try:
            with urllib.request.urlopen(f"{SERVER}/health", timeout=5):  # nosec B310
                return
        except OSError:
            time.sleep(10)
    raise TimeoutError(f"vLLM did not come up in {timeout_s:.0f} s")


@app.function(gpu="H200:2", volumes={HF_CACHE: hf_cache}, timeout=2 * 60 * 60)
def train(recipe_args: str) -> None:
    server = subprocess.Popen(  # nosec B603 B607
        [
            "vllm",
            "serve",
            MODEL,
            "--max-model-len",
            "4096",
            "--mamba-ssm-cache-dtype",
            "float32",
            "--logprobs-mode",
            "processed_logprobs",
            "--weight-transfer-config",
            '{"backend":"nccl"}',
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_SERVER_DEV_MODE": "1"},
    )
    try:
        _wait_for(server, timeout_s=45 * 60)
        hf_cache.commit()
        subprocess.run(  # nosec B603 B607
            [
                "python",
                f"{REMOTE_PACKAGE}/examples/nemotron_3_5_lightning.py",
                *shlex.split(recipe_args),
            ],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
            cwd=f"{REMOTE_PACKAGE}/examples",
            check=True,
        )
    finally:
        server.terminate()
        server.wait(timeout=60)
        hf_cache.commit()


@app.local_entrypoint()
def main(
    max_steps: int = 3,
    recipe_args: str = "--prompts-per-step 2 --num-generations 4 "
    "--max-completion-length 256 --token-budget 8192",
) -> None:
    train.remote(f"--max-steps {max_steps} --output-dir /tmp/run {recipe_args}")
