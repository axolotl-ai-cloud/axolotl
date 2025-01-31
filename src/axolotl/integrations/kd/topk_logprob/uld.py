# Copyright 2025 Axolotl AI. All rights reserved.
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
loss for Universal Logit Distillation
"""
import torch
import torch.nn.functional as F


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
    A Universal Logit Distillation (ULD) loss function that is TorchScript-friendly.
    Computes Wasserstein distance between sorted probability distributions.

    Args:
        student_logits: Student model logits [B, seq_len, vocab_size]
        target_token_ids: Teacher's token ids [B, seq_len, K]
        target_logprobs: Teacher's log probabilities [B, seq_len, K]
        target_mask: Mask indicating valid tokens [B, seq_len]
        num_items_in_batch: Number of items to normalize by (-1 for token count)
        kd_temperature: Temperature scaling factor
    """
    # Determine the teacher sequence length
    teacher_seq_len = target_token_ids.shape[1]

    # Slice student logits to match teacher-provided sequence length
    student_logits = student_logits[:, :teacher_seq_len, :]

    # Apply temperature scaling to student logits
    if kd_temperature != 1.0:
        student_logits = student_logits / kd_temperature

    # Convert student logits to probabilities
    student_probs = F.softmax(student_logits, dim=-1)

    # Convert teacher logprobs to probabilities
    teacher_probs = target_logprobs.exp()

    # Convert mask to boolean
    valid_mask = target_mask.to(torch.bool)

    # Get masked student probabilities
    student_probs_masked = student_probs[valid_mask]

    # Get masked teacher probabilities
    teacher_probs_masked = teacher_probs[valid_mask]

    # Sort student probabilities in descending order
    student_probs_sorted, _ = torch.sort(student_probs_masked, dim=-1, descending=True)

    # For teacher probs, we already have top-K, so just ensure they're sorted
    teacher_probs_sorted, _ = torch.sort(teacher_probs_masked, dim=-1, descending=True)

    # Pad the smaller distribution to match the larger one
    max_vocab_size = max(student_probs_sorted.size(1), teacher_probs_sorted.size(1))
    if student_probs_sorted.size(1) < max_vocab_size:
        student_probs_sorted = F.pad(
            student_probs_sorted, (0, max_vocab_size - student_probs_sorted.size(1))
        )
    if teacher_probs_sorted.size(1) < max_vocab_size:
        teacher_probs_sorted = F.pad(
            teacher_probs_sorted, (0, max_vocab_size - teacher_probs_sorted.size(1))
        )

    # Compute Wasserstein distance
    wasserstein_distance = torch.abs(student_probs_sorted - teacher_probs_sorted).sum(
        dim=-1
    )
    uld_loss = wasserstein_distance.sum()

    # Apply temperature scaling factor
    if kd_temperature != 1.0:
        uld_loss = uld_loss * (kd_temperature**2)

    # Normalize by batch size or token count
    if num_items_in_batch > 0:
        uld_loss = uld_loss / float(num_items_in_batch)
    else:
        uld_loss = uld_loss / float(wasserstein_distance.size(0))

    return uld_loss
