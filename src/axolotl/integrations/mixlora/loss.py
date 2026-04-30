# Copyright 2025 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Auxiliary loss collection for MixLoRA.

Collects the router load-balance losses accumulated during forward passes
and produces a single scaled loss to add to the primary training objective.
"""

import torch

from axolotl.integrations.mixlora.model import MixLoraFFN
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def collect_mixlora_aux_loss(
    model: torch.nn.Module,
    router_aux_loss_coef: float = 0.01,
) -> torch.Tensor:
    """Collect and reset auxiliary load-balance losses from all MixLoRA FFN blocks.

    This function should be called once per training step, after the forward pass.
    It walks all MixLoraFFN modules, collects their accumulated auxiliary losses,
    computes the mean, scales by the coefficient, and resets the accumulators.

    Note: Under gradient accumulation, only the last micro-batch's auxiliary loss
    is captured since ``MixLoraFFN.forward()`` overwrites (not accumulates)
    ``_aux_loss`` on each call.  This is intentional — the auxiliary loss is added
    to the per-micro-batch loss inside ``MixLoraTrainer.compute_loss``, so each
    micro-batch contributes its own scaled auxiliary term to the gradient.

    Args:
        model: The model (may be wrapped in PeftModel, DataParallel, etc.).
        router_aux_loss_coef: Coefficient for the auxiliary loss.

    Returns:
        Scalar tensor: the weighted auxiliary loss to add to the primary loss.
        Returns 0.0 if no MixLoRA blocks are found or no losses were accumulated.
    """
    aux_losses = []

    for module in model.modules():
        if isinstance(module, MixLoraFFN):
            if module.aux_loss is not None:
                aux_losses.append(module.aux_loss)
            module.reset_aux_loss()

    if not aux_losses:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    total_aux_loss = torch.stack(aux_losses).mean()
    return router_aux_loss_coef * total_aux_loss
