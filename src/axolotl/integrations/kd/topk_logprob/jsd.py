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
loss for Jensen-Shannon divergence
"""
import torch


@torch.jit.script
def loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target_mask: torch.Tensor,
    num_items_in_batch: int = -1,  # Use -1 to indicate "None"
    kd_temperature: float = 1.0,
    beta: float = 0.5,
) -> torch.Tensor:
    """
    A JSD loss function that is TorchScript-friendly.
    Computes generalized Jensen-Shannon Divergence between student and teacher distributions.

    Args:
        student_logits: Student model logits [B, seq_len, vocab_size]
        teacher_logits: Teacher model logits [B, seq_len, vocab_size]
        target_mask: Mask indicating valid tokens [B, seq_len]
        num_items_in_batch: Number of items to normalize by (-1 for token count)
        kd_temperature: Temperature for softmax
        beta: Interpolation coefficient between student and teacher (default: 0.5)
    """
    # Apply temperature scaling
    if kd_temperature != 1.0:
        student_logits = student_logits / kd_temperature
        teacher_logits = teacher_logits / kd_temperature

    # Convert to log probabilities
    student_log_probs = student_logits - torch.logsumexp(
        student_logits, dim=-1, keepdim=True
    )
    teacher_log_probs = teacher_logits - torch.logsumexp(
        teacher_logits, dim=-1, keepdim=True
    )

    # Compute log of mixture distribution
    # log(β*p + (1-β)*q) = logsumexp(log(β) + log(p), log(1-β) + log(q))
    mixture_log_probs = torch.logsumexp(
        torch.stack(
            [
                student_log_probs + torch.log(beta),
                teacher_log_probs + torch.log(1 - beta),
            ]
        ),
        dim=0,
    )

    # Convert mask to boolean for indexing
    valid_mask = target_mask.to(torch.bool)

    # Compute KL divergences
    kl_teacher = torch.sum(
        teacher_log_probs.exp() * (teacher_log_probs - mixture_log_probs), dim=-1
    )[valid_mask]

    kl_student = torch.sum(
        student_log_probs.exp() * (student_log_probs - mixture_log_probs), dim=-1
    )[valid_mask]

    # Compute final JSD loss
    jsd_loss = (beta * kl_teacher + (1 - beta) * kl_student).sum()

    # Apply temperature scaling factor
    if kd_temperature != 1.0:
        jsd_loss = jsd_loss * (kd_temperature**2)

    # Normalize by batch size or token count
    if num_items_in_batch > 0:
        jsd_loss = jsd_loss / float(num_items_in_batch)
    else:
        jsd_loss = jsd_loss / float(kl_teacher.size(0))

    return jsd_loss
