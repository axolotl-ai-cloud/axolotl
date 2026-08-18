# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI

"""Arctic Platform SFT integration for Axolotl.

Routes axolotl's preprocessed SFT batches to a remote Arctic Platform
training server (``ArcticSFTClient``) instead of running forward/backward
locally. The axolotl process itself needs no GPUs — all GPU work happens on
the server (HTTP now, Ray optional). Model weights, the optimizer, and
gradient updates all live on the server.

On-prem (``backend: onprem``) — local server, ``http`` or ``ray``. No
``host`` / ``port``::

    plugins:
      - axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin

    arctic_sft:
      backend: onprem
      comm_protocol: http
      training_gpus: 2
      launch_local_server: true
      server_cuda_visible_devices: "0,1"
      checkpoint_path: ./arctic_sft_ckpt

Remote (``backend: remote``) — ``host`` / ``port``, ``http`` or ``cortex``.
Not accepted until ``TODO(arctic-sft-backends)``; do not copy-paste yet::

    plugins:
      - axolotl.integrations.arctic_platform.sft.ArcticSFTPlugin

    arctic_sft:
      backend: remote
      comm_protocol: http
      host: dss-gpu-host.example.com
      port: 8765
      training_gpus: 2
      checkpoint_path: ./arctic_sft_ckpt

Other knobs (loss, logits tiling, sampling, timeouts):

- Plugin usage: ``axolotl/integrations/arctic_platform/README.md``
- Full ``arctic_sft:`` fields: ``args.py`` (``ArcticSFTConfig``)
- Worked YAML: ``examples/arctic_sft.yaml``
- Server client: arctic-platform ``docs/sft.md``
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
