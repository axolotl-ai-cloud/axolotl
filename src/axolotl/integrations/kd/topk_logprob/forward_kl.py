# Copyright 2024 Axolotl AI. All rights reserved.
#
# This software may be used and distributed according to
# the terms of the Axolotl Community License Agreement (the "License");
# you may not use this file except in compliance with the License.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""
loss for top_k KL divergence
"""
import torch


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
    """

    # Determine the teacher sequence length
    # target_token_ids shape: [B, teacher_seq_len, K]
    # student_logits shape:   [B, student_seq_len, vocab_size]
    teacher_seq_len = target_token_ids.shape[1]

    # Slice student logits to match teacher-provided sequence length
    student_logits_for_kd = student_logits[
        :, :teacher_seq_len, :
    ]  # [B, teacher_seq_len, vocab_size]

    # Gather student logits for teacher's top-K tokens
    student_logits_topk = torch.gather(
        student_logits_for_kd, dim=-1, index=target_token_ids
    )  # [B, teacher_seq_len, K]

    # Apply KD temperature to student’s logits
    if kd_temperature != 1.0:
        student_logits_topk = student_logits_topk / kd_temperature

    # Convert student top-k logits to logprobs
    student_logprobs_topk = student_logits_topk - torch.logsumexp(
        student_logits_topk, dim=-1, keepdim=True
    )  # [B, teacher_seq_len, K]

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
