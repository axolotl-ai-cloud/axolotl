# Copyright 2024 Axolotl AI. All rights reserved.
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
KD trainer
"""

import torch.nn as nn
from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer

from .kernels.liger import LigerFusedLinearKLTopKLogprobLoss


def _resolve_lm_head(model: nn.Module) -> nn.Module:
    base = model
    if hasattr(base, "get_base_model"):
        base = base.get_base_model()
    if hasattr(base, "language_model") and hasattr(base.language_model, "lm_head"):
        return base.language_model.lm_head
    if hasattr(base, "lm_head"):
        return base.lm_head
    raise AttributeError(f"could not find lm_head on {type(model).__name__}")


class AxolotlKDTrainer(AxolotlTrainer):
    """
    Custom trainer subclass for Knowledge Distillation (KD)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True

        self._kd_loss_fn = LigerFusedLinearKLTopKLogprobLoss(
            self.args.kd_ce_alpha,  # hard label loss
            self.args.kd_alpha,  # kd loss
            self.args.kd_temperature,
            self.args.kd_beta or 0.0,
            compute_ce_loss=bool(self.args.kd_ce_alpha),
            normalize_topk=self.args.kd_normalize_topk,
        )

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        columns_to_add = []
        if self._signature_columns:
            if "target_logprobs" not in self._signature_columns:
                columns_to_add.append("target_logprobs")
            if "target_token_ids" not in self._signature_columns:
                columns_to_add.append("target_token_ids")
            if "target_mask" not in self._signature_columns:
                columns_to_add.append("target_mask")
            if columns_to_add:
                self._signature_columns += columns_to_add

    @override
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        inputs = dict(inputs)

        required_keys = ("labels", "target_token_ids", "target_logprobs", "target_mask")
        missing = [k for k in required_keys if k not in inputs]
        if missing:
            raise KeyError(f"KD batch missing required keys: {missing}")

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        labels = inputs.pop("labels")
        target_token_ids = inputs.pop("target_token_ids")
        target_logprobs = inputs.pop("target_logprobs")
        target_mask = inputs.pop("target_mask")

        # num_items_in_batch is a loss kwarg, not a forward kwarg.
        inputs.pop("num_items_in_batch", None)

        inputs["output_hidden_states"] = True
        inputs["return_dict"] = True
        inputs["logits_to_keep"] = 1
        outputs = model(**inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError(
                f"{type(model).__name__}.forward did not return hidden_states"
            )
        hidden_states = hidden_states[-1]

        lm_head = _resolve_lm_head(model)
        hidden_states = hidden_states.to(lm_head.weight.dtype)

        loss = self._kd_loss_fn(
            lm_head.weight,
            hidden_states,
            target_token_ids,
            target_logprobs,
            target_mask,
            true_labels=labels,
        )

        if num_items_in_batch is not None and num_items_in_batch > 0:
            loss = loss / num_items_in_batch

        return (loss, outputs) if return_outputs else loss
