"""
chunked ce loss
"""

from typing import List, Optional

import torch
import torch.nn.functional as F


# copied and modified from torchtune.modules.loss.CEWithChunkedOutputLoss
class CEWithChunkedOutputLoss(torch.nn.Module):
    """
    Cross-entropy with chunked outputs that saves memory by only upcasting one chunk at a time.

    For more details, please refer to: https://github.com/pytorch/torchtune/pull/1390
    """

    def __init__(
        self,
        num_output_chunks: int = 8,
        ignore_index: int = -100,
        use_dft: bool = False,
    ):
        super().__init__()
        self.num_output_chunks = num_output_chunks
        self.ignore_index = ignore_index
        self.use_dft = use_dft

    def compute_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        normalize: bool = True,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """
        Upcast logits to fp32 and compute cross entropy loss.
        """
        ce_loss = F.cross_entropy(
            logits.float(), labels, ignore_index=self.ignore_index, reduction="none"
        )

        if self.use_dft:
            # Compute probabilities and gather the ones corresponding to labels
            with torch.no_grad():  # Stop gradient
                probs = torch.softmax(logits.float(), dim=-1)
                # Create mask for valid tokens (not ignore_index)
                valid_mask = labels != self.ignore_index
                # Gather probabilities for the correct tokens
                label_probs = probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                # Apply mask to only scale valid tokens
                label_probs = label_probs * valid_mask
                # Avoid multiplication by 0 for ignored tokens
                label_probs = torch.where(
                    valid_mask, label_probs, torch.ones_like(label_probs)
                )

            # Scale the loss by the probability (DFT)
            ce_loss = ce_loss * label_probs

        return ce_loss.sum()

    def forward(
        self, logits: List[torch.Tensor], labels: torch.Tensor, reduction="sum"
    ) -> torch.Tensor:
        """
        Args:
            logits (List[torch.Tensor]): List of chunked logits of length
                ``self.num_output_chunks``, where each chunk has shape
                ``(batch_size, num_tokens / num_output_chunks, vocab_size)``.
            labels (torch.Tensor): Ground truth labels of shape ``(batch_size, num_tokens)``.
            reduction (str): The reduction to apply to the output.

        Returns:
            torch.Tensor: Cross entropy loss of shape (1,).
        """

        total_elements = (labels != self.ignore_index).sum()

        # chunk and reshape labels (bsz, num_tokens, vocab) -> [(bsz*num_tokens/num_chunks, vocab)]
        labels = [
            target_chunk.reshape(-1)
            for target_chunk in labels.chunk(self.num_output_chunks, dim=1)
        ]
        # reshape logits [(bsz, num_tokens/num_chunks, vocab)] -> [(bsz*num_tokens/num_chunks, vocab)]
        logits = [
            logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits
        ]

        # compute one chunk at a time
        total_loss = 0.0
        for logits_chunk, labels_chunk in zip(logits, labels):
            total_loss += self.compute_cross_entropy(logits_chunk, labels_chunk)

        if reduction == "sum":
            return total_loss
        return total_loss / total_elements


def _build_chunked_ce_loss_fn(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    loss_fn_ce = CEWithChunkedOutputLoss(num_output_chunks, ignore_index, use_dft)
    loss_fn_ce.compute_cross_entropy = torch.compile(
        loss_fn_ce.compute_cross_entropy, backend="inductor"
    )
    return loss_fn_ce


def get_causal_lm_loss(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    loss_fn_ce = _build_chunked_ce_loss_fn(num_output_chunks, ignore_index, use_dft)

    def chunked_fix_cross_entropy(
        source,
        target,
        num_items_in_batch: int = None,
        ignore_index: int = -100,
        **kwargs,
    ):  # pylint: disable=unused-argument
        reduction = "sum" if num_items_in_batch is not None else "mean"
        logit_chunks = [  # pylint: disable=unnecessary-comprehension
            chunk for chunk in source.chunk(loss_fn_ce.num_output_chunks, dim=1)
        ]
        loss = loss_fn_ce(logit_chunks, target, reduction=reduction)
        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss

    def for_causal_lm_chunked_loss(
        logits,
        labels,
        vocab_size: int = None,  # pylint: disable=unused-argument
        num_items_in_batch: Optional[int] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # skip the upcast to float since we handle that in the chunking loss
        if shift_labels is None:
            # Shift so that tokens < n predict n
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        # Skip Flattening the tokens
        # Enable model parallelism
        shift_labels = shift_labels.to(logits.device)
        loss = chunked_fix_cross_entropy(
            logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
        )
        return loss

    return for_causal_lm_chunked_loss


def patch_chunked_ce_loss_fn(
    num_output_chunks: int = 8, ignore_index: int = -100, use_dft: bool = False
):
    import transformers.loss.loss_utils

    for_causal_lm_chunked_loss = get_causal_lm_loss(
        num_output_chunks, ignore_index, use_dft
    )
    transformers.loss.loss_utils.ForCausalLMLoss = for_causal_lm_chunked_loss
    transformers.loss.loss_utils.LOSS_MAPPING["ForCausalLM"] = (
        for_causal_lm_chunked_loss
    )
