# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-offload plugin for axolotl: per-replica CPU offload of frozen 4-bit MoE experts."""

from __future__ import annotations

from axolotl.integrations.base import BasePlugin


class ExpertOffloadPlugin(BasePlugin):
    """Stream frozen 4-bit MoE experts from pinned CPU RAM one block at a time.

    Moves only the experts (the bulk of a MoE's parameters), unlike whole-layer
    ``layer_offloading``, so per-step PCIe traffic stays small. See the integration README.
    """

    def get_input_args(self):
        return "axolotl.integrations.expert_offload.ExpertOffloadArgs"

    def post_model_load(self, cfg, model):
        """Install the offload after the model is built, quantized, PEFT-wrapped and on the GPU."""
        if not getattr(cfg, "expert_offload", False):
            return
        from .offload import install_expert_offload

        install_expert_offload(
            model, pin=getattr(cfg, "expert_offload_pin_memory", True)
        )
