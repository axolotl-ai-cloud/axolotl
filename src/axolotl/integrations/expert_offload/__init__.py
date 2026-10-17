# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-granularity CPU offload integration for axolotl.

Streams the frozen 4-bit experts of a MoE model from pinned CPU RAM to the GPU one block at a time,
lowering peak VRAM so a MoE whose experts exceed the card can QLoRA-train on a single small GPU.
Attention, router, norms and the trainable LoRA adapters stay GPU-resident.
"""

from .args import ExpertOffloadArgs
from .plugin import ExpertOffloadPlugin

__all__ = [
    "ExpertOffloadArgs",
    "ExpertOffloadPlugin",
]
