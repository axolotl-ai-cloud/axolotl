"""
KD trainer
"""

from typing import Optional

import torch

from axolotl.core.trainers.base import AxolotlTrainer


def kd_loss_function(
    student_logits,
    target_token_ids,
    target_logprobs,
    target_mask,
    num_items_in_batch: Optional[int] = None,
):
    # teacher_mask: [B, teacher_seq_len, K], where 1 indicates a valid token and 0 indicates padding

    # Determine the teacher sequence length
    teacher_seq_len = target_token_ids.shape[1]

    # Slice student logits to match the teacher-provided sequence length
    student_logits_for_kd = student_logits[
        :, :teacher_seq_len, :
    ]  # [B, teacher_seq_len, vocab_size]

    # Gather student logits for teacher's top-K tokens
    student_logits_topk = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )  # [B, teacher_seq_len, K]

    # Convert student top-k logits to logprobs
    student_logprobs_topk = student_logits_topk - torch.logsumexp(
        student_logits_topk, dim=-1, keepdim=True
    )  # [B, seq_len, K]

    # Convert teacher_mask to boolean for indexing
    valid_mask = target_mask.bool()

    # Prune tensors to only keep valid tokens
    # This will result in 1D arrays of only valid positions
    student_logprobs_topk = student_logprobs_topk[valid_mask]  # [N_valid_tokens]
    target_logprobs = target_logprobs[valid_mask]  # [N_valid_tokens]

    # Since teacher_logprobs are already normalized, just exponentiate to get probabilities
    teacher_probs = target_logprobs.exp()

    # Compute forward KL:
    # KL = sum p^T_k (log p^T_k - log p^S_k), summed over all valid tokens.
    kd_loss_per_token = teacher_probs * (target_logprobs - student_logprobs_topk)
    kd_loss = kd_loss_per_token.sum()

    # Normalize by number of items or mean over valid tokens
    if num_items_in_batch is not None:
        # If you know how many items should be considered in the batch
        kd_loss = kd_loss / num_items_in_batch
    else:
        # Otherwise, just average over all valid tokens
        kd_loss = kd_loss / kd_loss_per_token.size(0)

    return kd_loss


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
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        target_logprobs = inputs.pop("target_logprobs")
        target_token_ids = inputs.pop("target_token_ids")
        target_mask = inputs.pop("target_mask")

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)

        student_logits = outputs["logits"]
        loss_kd = kd_loss_function(
            student_logits,
            target_token_ids,
            target_logprobs,
            target_mask,
            num_items_in_batch=num_items_in_batch,
        )

        if self.args.kd_ce_alpha > 0:
            loss = self.args.kd_ce_alpha * outputs["loss"] + loss_kd
        else:
            loss = loss_kd
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[  # pylint: disable=attribute-defined-outside-init
                self.args.past_index
            ]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss_kd *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss
