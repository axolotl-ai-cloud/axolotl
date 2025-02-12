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

from axolotl.core.trainers.base import AxolotlTrainer

from .topk_logprob.forward_kl import loss as topk_kd_loss
from .topk_logprob.forward_kl import topk_kd_loss_with_zscore


class AxolotlKDTrainer(AxolotlTrainer):
    """
    Custom trainer subclass for Knowledge Distillation (KD)
    """

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

        target_logprobs = inputs.pop("target_logprobs")
        target_token_ids = inputs.pop("target_token_ids")
        target_mask = inputs.pop("target_mask")

        seq_len = target_token_ids.shape[1]

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)

        # FIXME: account for tokenizer.padding_side
        student_logits = outputs["logits"][:, : seq_len - 1, :].contiguous()

        shift_logits = student_logits.contiguous()
        target_logprobs_for_loss = target_logprobs[..., 1:, :].contiguous()
        target_token_ids_for_loss = target_token_ids[..., 1:, :].contiguous()
        target_mask_for_loss = target_mask[..., 1:, :].contiguous()

        if self.args.kd_zscore_base_temp:
            loss_kd = topk_kd_loss_with_zscore(
                shift_logits,
                target_token_ids_for_loss,
                target_logprobs_for_loss,
                target_mask_for_loss,
                kd_temperature=self.args.kd_temperature,
                zscore_base_temp=self.args.kd_zscore_base_temp,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            loss_kd = topk_kd_loss(
                shift_logits,
                target_token_ids_for_loss,
                target_logprobs_for_loss,
                target_mask_for_loss,
                num_items_in_batch=num_items_in_batch,
                kd_temperature=self.args.kd_temperature,
                top_k_before_softmax=1 if self.args.kd_top_k_before_softmax else 0,
            )

        if self.args.kd_ce_alpha > 0:
            kd_alpha = self.args.kd_alpha
            loss = self.args.kd_ce_alpha * outputs["loss"] + kd_alpha * loss_kd
        else:
            loss = loss_kd
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[  # pylint: disable=attribute-defined-outside-init
                self.args.past_index
            ]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
