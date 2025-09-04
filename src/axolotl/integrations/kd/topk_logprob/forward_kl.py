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


@torch.jit.script
def loss(
    student_logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    target_logprobs: torch.Tensor,
    target_mask: torch.Tensor,
    num_items_in_batch: int = -1,  # Use -1 to indicate "None"
    kd_temperature: float = 1.0,
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
    """

    target_logprobs = target_logprobs.float()

    # Determine the teacher sequence length
    # target_token_ids shape: [B, teacher_seq_len, K]
    # student_logits shape:   [B, student_seq_len, vocab_size]
    teacher_seq_len = target_token_ids.shape[1]

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

    # Normalize by number of items (if provided) or by valid tokens
    if num_items_in_batch > 0:
        kd_loss = kd_loss / float(num_items_in_batch)
    else:
        # Fall back to average over valid tokens
        kd_loss = kd_loss / float(kd_loss_per_token.size(0))

    return kd_loss


class ChunkedTopKKDLoss(nn.Module):
    """
    A wrapper that chunks (splits) the student and teacher outputs along the time dimension
    to reduce peak memory usage when upcasting from bf16 to fp32, especially for large vocabularies.

    Usage is analogous to ForwardKLWithChunkedOutputLoss but adapted to top-K teacher logprobs.
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
            student_logits_chunks,
            token_ids_chunks,
            logprobs_chunks,
            mask_chunks,
            strict=False,
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
