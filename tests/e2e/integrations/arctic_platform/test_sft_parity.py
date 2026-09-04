# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Quality test: native Axolotl SFT vs Arctic Platform SFT loss parity.

Both paths run the same short SFT job (same seed / LR / batches) under DeepSpeed
with an **identical config** — the only way the two execution paths produce
identical numerics. Each side runs in its own correctly-launched process and
writes per-step losses to JSON; this test compares them:

* **native** — axolotl baseline started via the ``deepspeed`` launcher (like
  ``torchrun``), so DeepSpeed gets a real distributed rendezvous. Uses
  ``cfg.deepspeed`` = the shared DeepSpeed config.
* **arctic** — ``ArcticSFTPlugin`` as a CPU-only client; the AP server owns the
  GPU and launches its own DeepSpeed workers with the same config.

Shared config: bf16 mixed precision, ZeRO-0, DeepSpeed's default **FusedAdam**,
identical gradient clipping / lr / betas / eps. See ``sft_parity_runner``.

NOTE: FlashAttention-3 is not installed in this environment; the runner uses
``flash_attention_2`` on both sides (parity holds for any attention as long as
it is identical on both). Switch once FA3 is installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ARCTIC_PLATFORM_INSTALLED = importlib.util.find_spec("arctic_platform") is not None

RUNNER = str(Path(__file__).parent / "sft_parity_runner.py")

MAX_STEPS = 3
LEARNING_RATE = 1e-3
# Identical DeepSpeed engine + config on both sides → match to fp noise. The
# only residual is the HF-Trainer vs AP call ordering (fp32 reductions).
RTOL = 1e-3
ATOL = 1e-3
MIN_REMOTE_MOVE = 1e-1  # |L[i] - L[0]| must exceed this for some i > 0
SUBPROCESS_TIMEOUT = 1800.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _read_losses(path: str) -> list[float]:
    with open(path, encoding="utf-8") as fin:
        return [float(x) for x in json.load(fin)["losses"]]


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    not ARCTIC_PLATFORM_INSTALLED, reason="arctic_platform package not installed"
)
class TestArcticSFTLossParity:
    """Native Axolotl SFT and AP/SFT should track the same loss for a few steps."""

    def test_native_vs_arctic_sft_losses(self, temp_dir):
        # Multi-GPU DataParallel breaks the native path; pin to one device.
        if torch.cuda.device_count() != 1:
            pytest.skip(
                "Set CUDA_VISIBLE_DEVICES to a single GPU "
                f"(saw {torch.cuda.device_count()} devices)"
            )

        native_dir = f"{temp_dir}/native"
        arctic_dir = f"{temp_dir}/arctic"
        native_losses_json = f"{temp_dir}/native_losses.json"
        arctic_losses_json = f"{temp_dir}/arctic_losses.json"
        master_port = _free_port()
        server_port = _free_port()

        # --- native: axolotl baseline via the deepspeed launcher -----------
        native_cmd = [
            "deepspeed",
            "--no_ssh",
            "--node_rank=0",
            "--num_gpus=1",
            f"--master_port={master_port}",
            "--no_local_rank",
            RUNNER,
            "--mode",
            "native",
            "--output-dir",
            native_dir,
            "--losses-out",
            native_losses_json,
        ]
        subprocess.run(
            native_cmd,
            check=True,
            timeout=SUBPROCESS_TIMEOUT,
            cwd=str(Path(RUNNER).parent),
        )
        native_losses = _read_losses(native_losses_json)
        assert len(native_losses) == MAX_STEPS, (
            f"native expected {MAX_STEPS} step losses, got {native_losses}"
        )

        # --- arctic: CPU-only client; server owns the GPU ------------------
        arctic_env = dict(os.environ)
        arctic_env["CUDA_VISIBLE_DEVICES"] = ""  # client stays off the GPU
        arctic_cmd = [
            sys.executable,
            RUNNER,
            "--mode",
            "arctic",
            "--output-dir",
            arctic_dir,
            "--losses-out",
            arctic_losses_json,
            "--port",
            str(server_port),
        ]
        subprocess.run(
            arctic_cmd,
            check=True,
            timeout=SUBPROCESS_TIMEOUT,
            env=arctic_env,
        )
        arctic_losses = _read_losses(arctic_losses_json)
        assert len(arctic_losses) == MAX_STEPS, (
            f"arctic expected {MAX_STEPS} step losses, got {arctic_losses}"
        )

        # --- compare -------------------------------------------------------
        assert native_losses == pytest.approx(arctic_losses, rel=RTOL, abs=ATOL), (
            f"per-step loss mismatch under lr={LEARNING_RATE}: "
            f"native={native_losses} arctic={arctic_losses}"
        )

        # Large LR must actually move the remote weights (guards a no-op server).
        remote_move = max(abs(x - arctic_losses[0]) for x in arctic_losses[1:])
        assert remote_move > MIN_REMOTE_MOVE, (
            f"arctic loss did not move under lr={LEARNING_RATE}: {arctic_losses}"
        )
