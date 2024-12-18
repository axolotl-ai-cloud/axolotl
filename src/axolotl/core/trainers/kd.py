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
    num_items_in_batch: Optional[int] = None,
    **kwargs,  # pylint: disable=unused-argument
):
    # student_logits: [B, seq_len, vocab_size] from the student's forward pass
    # target_token_ids: [B, teacher_seq_len, K] top-K token IDs from teacher
    # target_logprobs: [B, teacher_seq_len, K] teacher logprobs for these top-K tokens

    teacher_seq_len = target_token_ids.shape[1]

    # Slice the student logits to match the teacher-provided seq length
    student_logits_for_kd = student_logits[
        :, -teacher_seq_len:, :
    ]  # Now [B, teacher_seq_len, vocab_size]

    # Gather student logits for teacher's top-K tokens
    student_logits_topk = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )  # [B, teacher_seq_len, K]

    # Convert student top-K logits to logprobs
    student_logprobs_topk = student_logits_topk - torch.logsumexp(
        student_logits_topk, dim=-1, keepdim=True
    )

    # teacher_probs are simply exp of teacher_logprobs (already scaled)
    teacher_probs = target_logprobs.exp()

    # Compute forward KL
    # L_kl = sum_k p^T_k (log p^T_k - log p^S_k)
    kd_loss_per_position = (
        teacher_probs * (target_logprobs - student_logprobs_topk)
    ).sum(
        dim=-1
    )  # [B, teacher_seq_len]

    # gradient accumulation fixes
    if num_items_in_batch:
        kd_loss = kd_loss_per_position.sum() / num_items_in_batch  # Scalar
    else:
        kd_loss = kd_loss_per_position.mean()  # Scalar

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
            num_items_in_batch=num_items_in_batch,
        )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[  # pylint: disable=attribute-defined-outside-init
                self.args.past_index
            ]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss_kd *= self.accelerator.num_processes

        return (loss_kd, outputs) if return_outputs else loss_kd
