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
MixLoRA trainer
"""

from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.integrations.mixlora.loss import collect_mixlora_aux_loss


class MixLoraTrainer(AxolotlTrainer):
    """
    Custom trainer subclass for MixLoRA to add auxiliary load-balance loss
    """

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

        if return_outputs:
            loss, outputs = result
            loss = self._add_mixlora_aux_loss(loss, model)
            return loss, outputs

        result = self._add_mixlora_aux_loss(result, model)
        return result

    def _add_mixlora_aux_loss(self, loss, model):
        """Add MixLoRA router auxiliary load-balance loss if applicable."""
        coef = getattr(self.axolotl_cfg, "mixlora_router_aux_loss_coef", None)
        router_aux_loss_coef = coef if coef is not None else 0.01
        
        aux_loss = collect_mixlora_aux_loss(
            model, router_aux_loss_coef=router_aux_loss_coef
        )
        
        loss = loss + aux_loss.to(loss.device)
        
        self.store_metrics(
            {"mixlora_aux_loss": aux_loss.item()}, train_eval="train"
        )
        return loss
