"""
loss for top_k KL divergence
"""
from typing import Optional

import torch


def loss(
    student_logits,
    target_token_ids,
    target_logprobs,
    target_mask,
    num_items_in_batch: Optional[int] = None,
    kd_temperature: float = 1.0,
):
    # teacher_mask: [B, teacher_seq_len, K], where 1 indicates a valid token and 0 indicates padding

    # Determine the teacher sequence length
    # _, teacher_seq_len, top_k = target_token_ids.shape
    teacher_seq_len = target_token_ids.shape[1]

    # Slice student logits to match the teacher-provided sequence length
    student_logits_for_kd = student_logits[
        :, :teacher_seq_len, :
    ]  # [B, teacher_seq_len, vocab_size]

    # Gather student logits for teacher's top-K tokens
    #  shape -> [B, teacher_seq_len, K]
    student_logits_topk = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )

    # Apply KD temperature to student’s logits:
    #  z_s(T) = z_s / T
    if kd_temperature != 1.0:
        student_logits_topk = student_logits_topk / kd_temperature

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

    # 9) Multiply by T^2 (classical KD scaling)
    if kd_temperature != 1.0:
        kd_loss = kd_loss * (kd_temperature**2)

    # Normalize by number of items or mean over valid tokens
    if num_items_in_batch is not None:
        # If you know how many items should be considered in the batch
        kd_loss = kd_loss / num_items_in_batch
    else:
        # Otherwise, just average over all valid tokens
        kd_loss = kd_loss / kd_loss_per_token.size(0)

    return kd_loss
