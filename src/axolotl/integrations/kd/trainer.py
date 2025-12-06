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

from typing_extensions import override

from axolotl.core.trainers.base import AxolotlTrainer

from .kernels.liger import LigerFusedLinearKLTopKLogprobLoss


class AxolotlKDTrainer(AxolotlTrainer):
    """
    Custom trainer subclass for Knowledge Distillation (KD)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True

        loss_fn = LigerFusedLinearKLTopKLogprobLoss(
            self.args.kd_ce_alpha,  # hard label loss
            self.args.kd_alpha,  # kd loss
            self.args.kd_temperature,
            self.args.kd_beta or 0.0,
            compute_ce_loss=bool(self.args.kd_ce_alpha),
            normalize_topk=self.args.kd_normalize_topk,
        )
        target = self.model

        # Unwrap PEFT wrapper
        if hasattr(target, "get_base_model"):
            target = target.get_base_model()

        # Set on the actual model instance
        target._loss_function = loss_fn

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
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (
            self.args.sample_packing
            and hasattr(inputs, "attention_mask")
            and hasattr(inputs, "position_ids")
        ):
            del inputs["attention_mask"]

        if num_items_in_batch is None and "labels" in inputs:
            num_items_in_batch = (inputs["labels"] != -100).sum().item()

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        outputs = model(**inputs)

        if isinstance(outputs, dict):
            loss = outputs["loss"]
        elif isinstance(outputs, tuple):
            loss = outputs[0]
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else outputs

        return (loss, outputs) if return_outputs else loss
