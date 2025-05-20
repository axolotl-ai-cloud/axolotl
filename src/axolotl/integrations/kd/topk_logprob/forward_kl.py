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
loss for top_k KL divergence
"""
import torch
from torch import nn


def zscore_standardize(
    logits: torch.Tensor,
    mask: torch.Tensor = None,
    base_temperature: float = 1.0,
    eps: float = 1e-9,
):
    """
    Z-score standardize along the last dimension of `logits`.
    i.e., for each [B, seq_len] row, across K entries:
        z = (logits - mean) / std,
    then scale by 1 / base_temperature if desired.

    mask can be broadcastable or None. If None, we standardize all elements.
    """
    if mask is None:
        # shape: [B, seq_len, K]
        # Mean and std over dim=-1
        mean = logits.mean(dim=-1, keepdim=True)
        var = logits.var(dim=-1, unbiased=False, keepdim=True)
    else:
        # If you have to exclude some tokens, multiply by mask, etc.
        float_mask = mask.to(logits.dtype)
        count = float_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (logits * float_mask).sum(dim=-1, keepdim=True) / count
        var = (float_mask * (logits - mean) ** 2).sum(dim=-1, keepdim=True) / count

    std = torch.sqrt(var.clamp_min(eps))
    z = (logits - mean) / std

    # Scale by 1 / base_temperature
    z = z / base_temperature
    return z


@torch.jit.script
def loss(
    student_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_mask: torch.Tensor,
    num_items_in_batch: int = -1,  # Use -1 to indicate "None"
    kd_temperature: float = 1.0,
    top_k_before_softmax: int = 0,
) -> torch.Tensor:
    """
    A KD loss function that is TorchScript-friendly.

    Arguments:
        student_logits (torch.Tensor): The logits of the student model.
            Shape: [B, student_seq_len, vocab_size]
        target_token_ids (torch.Tensor): The top-k teacher/target token IDs
            Shape: [B, teacher_seq_len, top_k]
        target_logprobs (torch.Tensor): The top-k teacher/target logprobs, these should already be re-normalized.
            Shape: [B, teacher_seq_len, top_k]
        target_mask (torch.Tensor): The mask for valid tokens.
            Shape: [B, teacher_seq_len, top_k]
        num_items_in_batch (int, optional): The number of items in the batch.
        kd_temperature (float, optional): The temperature for KD.
            Default: 1.0
        top_k_before_softmax (int, optional): Flag of whether to apply softmax before gathering student top-k logits
            Default: 0
    """

    target_logprobs = target_logprobs.float()

    # Determine the teacher sequence length
    # target_token_ids shape: [B, teacher_seq_len, K]
    # student_logits shape:   [B, student_seq_len, vocab_size]
    teacher_seq_len = target_token_ids.shape[1]

    if top_k_before_softmax:
        # Slice student logits to match teacher-provided sequence length
        student_logits_for_kd = student_logits[
            :, :teacher_seq_len, :
        ]  # [B, teacher_seq_len, vocab_size]

        # Gather student logits for teacher's top-K tokens
        student_logits_topk = torch.gather(
            student_logits_for_kd, dim=-1, index=target_token_ids
        )  # [B, teacher_seq_len, K]

        student_logits_topk = student_logits_topk.float()

        # Apply KD temperature to student’s logits
        if kd_temperature != 1.0:
            student_logits_topk = student_logits_topk / kd_temperature

        # Convert student top-k logits to logprobs
        student_logprobs_topk = student_logits_topk - torch.logsumexp(
            student_logits_topk, dim=-1, keepdim=True
        )  # [B, teacher_seq_len, K]
    else:
        # Slice student logits to match teacher-provided sequence length
        student_logits_for_kd = (
            student_logits[:, :teacher_seq_len, :] / kd_temperature
        )  # [B, teacher_seq_len, vocab_size]

        # keep in full precision for numerical stability of loss
        student_logits_for_kd = student_logits_for_kd.float()

        # Gather student logits for teacher's top-K tokens
        student_logits_topk = torch.gather(
            student_logits_for_kd, dim=-1, index=target_token_ids
        )  # [B, teacher_seq_len, K]

        # Compute logsumexp across full vocabulary
        student_lse = torch.logsumexp(student_logits_for_kd, dim=-1, keepdim=True)

        #  Convert just the top-k logits to logprobs
        student_logprobs_topk = student_logits_topk - student_lse

    # Convert teacher_mask to boolean for indexing
    # In TorchScript, .bool() is sometimes unsupported, so we do:
    valid_mask = target_mask.to(torch.bool)

    # Prune tensors to only keep valid tokens
    student_logprobs_topk = student_logprobs_topk[valid_mask]
    target_logprobs = target_logprobs[valid_mask]

    # Convert teacher logprobs to probabilities
    teacher_probs = target_logprobs.exp()

    # Compute forward KL
    kd_loss_per_token = teacher_probs * (target_logprobs - student_logprobs_topk)
    kd_loss = kd_loss_per_token.sum()

    # Multiply by T^2 (classical KD scaling)
    if kd_temperature != 1.0:
        kd_loss = kd_loss * (kd_temperature**2)

    # Normalize by number of items (if provided) or by valid tokens
    if num_items_in_batch > 0:
        kd_loss = kd_loss / float(num_items_in_batch)
    else:
        # Fall back to average over valid tokens
        kd_loss = kd_loss / float(kd_loss_per_token.size(0))

    return kd_loss


def topk_kd_loss_with_zscore(
    student_logits: torch.Tensor,  # [B, seq_len, vocab_size]
    target_token_ids: torch.Tensor,  # [B, seq_len, K]
    target_logprobs: torch.Tensor,  # [B, seq_len, K], sums to 1.0 in prob space
    target_mask: torch.Tensor,  # [B, seq_len, K] or [B, seq_len]
    kd_temperature: float = 1.0,  # classic KD temperature
    zscore_base_temp: float = 1.0,  # from the paper
    num_items_in_batch: int = -1,
):
    """
    A variant of top_k KL divergence with Z-score scaling
    from "Logit Standardization in Knowledge Distillation".
    """

    target_logprobs = target_logprobs.float()

    B, teacher_seq_len, K = target_logprobs.shape  # pylint: disable=invalid-name
    # 1) Gather the student's top-k logits to match teacher
    student_logits_for_kd = student_logits[
        :, :teacher_seq_len, :
    ]  # [B, seq_len, vocab]
    student_topk_logits = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )  # [B, seq_len, K]

    student_topk_logits = student_topk_logits.float()

    # 2) If you want to keep the "classical" T scaling, apply it first
    if kd_temperature != 1.0:
        student_topk_logits = student_topk_logits / kd_temperature

    # 3) Convert teacher logprobs -> treat them as “logits” for z-score
    #    (They differ by +some_constant from real logits, but in z-score
    #     that constant is subtracted out anyway.)
    teacher_logits_for_zscore = target_logprobs  # rename variable for clarity

    # 4) Z-score teacher and student
    #    If target_mask is 2D, expand to 3D for the K dimension
    if target_mask.dim() == 2 and target_mask.shape[:2] == (B, teacher_seq_len):
        target_mask = target_mask.unsqueeze(-1).expand(-1, -1, K)

    teacher_z = zscore_standardize(
        teacher_logits_for_zscore, mask=target_mask, base_temperature=zscore_base_temp
    )
    student_z = zscore_standardize(
        student_topk_logits, mask=target_mask, base_temperature=zscore_base_temp
    )

    # 5) Convert to log-probs for KL
    teacher_logprobs_z = teacher_z - torch.logsumexp(teacher_z, dim=-1, keepdim=True)
    student_logprobs_z = student_z - torch.logsumexp(student_z, dim=-1, keepdim=True)

    # 6) Restrict to valid tokens if needed
    valid_mask = target_mask.bool()  # shape [B, seq_len, K]
    teacher_probs_z = teacher_logprobs_z.exp()
    teacher_probs_z = teacher_probs_z[valid_mask]
    teacher_logprobs_z = teacher_logprobs_z[valid_mask]
    student_logprobs_z = student_logprobs_z[valid_mask]

    # 7) forward KL:  sum( p_teacher * [log(p_teacher) - log(p_student)] )
    kd_loss_per_token = teacher_probs_z * (teacher_logprobs_z - student_logprobs_z)
    kd_loss = kd_loss_per_token.sum()

    # 8) If using classical KD scaling by T^2
    if kd_temperature != 1.0:
        kd_loss = kd_loss * (kd_temperature**2)

    # Optionally scale by zscore_base_temp**2 if you want (paper might differ).
    # kd_loss = kd_loss * (zscore_base_temp**2)

    # 9) Normalize
    if num_items_in_batch is not None and num_items_in_batch > 0:
        kd_loss = kd_loss / float(num_items_in_batch)
    else:
        kd_loss = kd_loss / float(kd_loss_per_token.size(0))

    return kd_loss


class ChunkedTopKKDLoss(nn.Module):
    """
    A wrapper that chunks (splits) the student and teacher outputs along the time dimension
    to reduce peak memory usage when upcasting from bf16 to fp32, especially for large vocabularies.

    Usage is analogous to ForwardKLWithChunkedOutputLoss but adapted to your top-K teacher logprobs.
    """

    def __init__(self, num_output_chunks: int = 8, kd_temperature: float = 1.0):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.kd_temperature = kd_temperature

    def forward(
        self,
        student_logits: torch.Tensor,  # [B, seq_len, vocab_size]
        target_token_ids: torch.Tensor,  # [B, seq_len, K]
        target_logprobs: torch.Tensor,  # [B, seq_len, K]
        target_mask: torch.Tensor,  # [B, seq_len, K]
        num_items_in_batch: int = -1,  # optional batch size for normalization
        top_k_before_softmax: int = 0,  # optional top-k before softmax for teacher logits
    ) -> torch.Tensor:

        # 1. Split along the "token" dimension (dim=1).
        student_logits_chunks = student_logits.chunk(self.num_output_chunks, dim=1)
        token_ids_chunks = target_token_ids.chunk(self.num_output_chunks, dim=1)
        logprobs_chunks = target_logprobs.chunk(self.num_output_chunks, dim=1)
        mask_chunks = target_mask.chunk(self.num_output_chunks, dim=1)

        # We'll accumulate a global "sum of losses" and "sum of valid tokens"
        # so that our final average is consistent with the entire sequence/batch.
        total_loss = 0.0
        total_valid_tokens = 0

        # 2. Loop over each chunk and compute a chunk-specific loss.
        for st_chunk, tid_chunk, lp_chunk, msk_chunk in zip(
            student_logits_chunks, token_ids_chunks, logprobs_chunks, mask_chunks
        ):
            # We pass num_items_in_batch=-1 so that the kd_loss
            # will average over *this chunk's* valid tokens only.
            chunk_loss = loss(
                student_logits=st_chunk,
                target_token_ids=tid_chunk,
                target_logprobs=lp_chunk,
                target_mask=msk_chunk,
                num_items_in_batch=-1,  # ensure per-chunk averaging by valid tokens
                kd_temperature=self.kd_temperature,
                top_k_before_softmax=top_k_before_softmax,
            )

            # kd_loss returns an average over the chunk's valid tokens.
            # We want a global average in the end, so we need to re‐weight
            # by the number of valid tokens in this chunk and keep track of the total.
            chunk_valid_mask = msk_chunk.to(torch.bool)
            chunk_valid_count = chunk_valid_mask.sum()  # scalar tensor

            # Re-scale "chunk average" back to "chunk sum"
            chunk_loss_sum = chunk_loss * chunk_valid_count

            total_loss += chunk_loss_sum
            total_valid_tokens += chunk_valid_count

        # 3. Normalize *once* at the end.
        if num_items_in_batch > 0:
            # If the user gave us a manual denominator (e.g. total items in batch),
            # we divide by it. Typically used if each item is of different length.
            final_loss = total_loss / float(num_items_in_batch)
        else:
            # Otherwise, divide by total valid tokens across all chunks.
            # to get the same result as a non-chunked approach.
            final_loss = total_loss / float(total_valid_tokens)

        return final_loss
