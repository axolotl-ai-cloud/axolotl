# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Arctic Platform SFT integration for Axolotl.

Routes axolotl's preprocessed SFT batches to a remote Arctic Platform
training server (``ArcticSFTClient``) instead of running forward/backward
locally. The axolotl process itself needs no GPUs — all GPU work happens on
the server (HTTP now, Ray optional). Model weights, the optimizer, and
gradient updates all live on the server.

Usage::

    plugins:
      - axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin

    arctic_sft:
      host: localhost
      port: 8765
      training_gpus: 2
      launch_local_server: true
      server_cuda_visible_devices: "0,1"

    learning_rate: 1e-5   # top-level, forwarded to the server optimizer
"""

from .args import ArcticSFTArgs, ArcticSFTConfig
from .deps import require_arctic_platform, require_arctic_sft_client
from .plugin import ArcticSFTPlugin

__all__ = [
    "ArcticSFTArgs",
    "ArcticSFTConfig",
    "ArcticSFTPlugin",
    "require_arctic_platform",
    "require_arctic_sft_client",
]
