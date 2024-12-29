"""
KD trainer
"""

import torch

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.kd.topk_logprob.forward_kl import loss as topk_kd_loss


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

    # def compute_loss_w_triton(
    #     self, model, inputs, return_outputs=False, num_items_in_batch=None
    # ):
    #     target_logprobs = inputs.pop("target_logprobs")
    #     target_token_ids = inputs.pop("target_token_ids")
    #     target_mask = inputs.pop("target_mask")
    #
    #     if self.model_accepts_loss_kwargs:
    #         loss_kwargs = {}
    #         if num_items_in_batch is not None:
    #             loss_kwargs["num_items_in_batch"] = num_items_in_batch
    #         inputs = {**inputs, **loss_kwargs}
    #     outputs = model(**inputs)
    #
    #     student_logits = outputs["logits"]
    #     # Slice or gather student logits to match teacher seq len
    #     # e.g.:
    #     teacher_seq_len = target_token_ids.shape[1]
    #     student_logits_for_kd = student_logits[
    #         :, :teacher_seq_len, :
    #     ]  # [B, seq_len, vocab_size]
    #
    #     # GATHER top-K from student
    #     student_logits_topk = torch.gather(
    #         student_logits_for_kd,
    #         dim=-1,
    #         index=target_token_ids,  # same shape [B, seq_len, K]
    #     )
    #
    #     # Now call the Triton-based KD loss
    #     kd_sum = kd_loss_triton(
    #         student_logits_topk,
    #         target_logprobs,  # teacher logprobs [B, seq_len, K]
    #         target_mask,  # mask [B, seq_len, K]
    #     )
    #
    #     # Normalize however you want
    #     if num_items_in_batch is not None:
    #         loss_kd = kd_sum / num_items_in_batch
    #     else:
    #         # or do e.g. average over valid tokens
    #         # quick example:
    #         total_valid = target_mask.sum()
    #         loss_kd = kd_sum / (total_valid + 1e-8)
    #
    #     # optionally combine with CE loss
    #     if self.args.kd_ce_alpha > 0:
    #         kd_alpha = self.args.kd_alpha
    #         loss = self.args.kd_ce_alpha * outputs["loss"] + kd_alpha * loss_kd
    #     else:
    #         loss = loss_kd
    #
    #     return (loss, outputs) if return_outputs else loss

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
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
        student_logits = outputs["logits"][:, :seq_len, :].contiguous()

        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_target_logprobs = target_logprobs[..., 1:, :].contiguous()
        shift_target_token_ids = target_token_ids[..., 1:, :].contiguous()
        shift_target_mask = target_mask[..., 1:, :].contiguous()

        loss_kd = topk_kd_loss(
            shift_logits,
            shift_target_token_ids,
            shift_target_logprobs,
            shift_target_mask,
            num_items_in_batch=num_items_in_batch,
            kd_temperature=self.args.kd_temperature,
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

        torch.cuda.empty_cache()

        return (loss, outputs) if return_outputs else loss
