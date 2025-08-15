"""Custom trainer for diffusion LM training."""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class DiffusionTrainer(AxolotlTrainer):  # pylint: disable=too-many-ancestors
    """Custom trainer for diffusion LM training that overrides loss computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._config = None
        self._special_token_ids = None

    def set_config(self, config: DictDefault):
        """Set config for diffusion training."""
        self._config = config
        self._cache_special_token_ids()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Override compute_loss to use diffusion loss."""
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if input_ids is None:
            raise ValueError("input_ids is required for diffusion training")

        loss, outputs = self._compute_diffusion_loss(model, input_ids, attention_mask)

        if return_outputs:
            return loss, outputs

        return loss

    def _cache_special_token_ids(self):
        """Cache special token IDs to avoid repeated tokenizer access."""
        if self.processing_class is None:
            self._special_token_ids = set()
            return

        tokenizer = self.processing_class
        special_tokens = set()

        if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
            special_tokens.add(tokenizer.bos_token_id)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            special_tokens.add(tokenizer.eos_token_id)
        if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
            special_tokens.add(tokenizer.pad_token_id)

        self._special_token_ids = special_tokens

    def _forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        eps: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward noising process. A timestep is sampled along the process, and tokens are
        masked with probability determined by the configured noise schedule.

        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].
            eps: Small epsilon value for minimum masking probability.

        Returns:
            noisy_batch: Input with some tokens masked.
            masked_indices: Boolean mask indicating which tokens were masked.
            p_mask: Masking probabilities for each token [batch_size, seq_len].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Sample random timesteps for each sample in batch
        t = torch.rand(batch_size, device=device)

        # Calculate masking probability with epsilon
        p_mask = (1 - eps) * t + eps  # [batch_size]
        p_mask = p_mask[:, None].repeat(1, seq_len)  # [batch_size, seq_len]

        # Don't mask padding tokens if attention_mask is provided
        if attention_mask is not None:
            valid_mask = attention_mask.bool()
            p_mask = p_mask * valid_mask.float()

        # Create mask to exclude special tokens (BOS, EOS, PAD) using cached IDs
        special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self._special_token_ids:
            for token_id in self._special_token_ids:
                special_token_mask |= input_ids == token_id

        # Create random mask based on probability, excluding special tokens
        masked_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        masked_indices = masked_indices & ~special_token_mask
        if attention_mask is not None:
            masked_indices = masked_indices & attention_mask.bool()

        # Get mask token ID from config
        mask_token_id = self._config.mask_token_id

        # Create masked input using configured mask token
        noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

        return noisy_batch, masked_indices, p_mask

    def _create_bidirectional_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Create bidirectional attention mask to override default causal masking.
        Handles sample-packed sequences where different samples are identified
        by different attention mask values.

        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            bidirectional_mask: 4D attention mask [batch_size, 1, seq_len, seq_len].
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None or not self._config.sample_packing:
            # Simple case: no attention mask, allow all-to-all attention
            return torch.ones(
                batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device
            )

        # Create attention mask by comparing sample IDs element-wise
        mask_i = attention_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
        mask_j = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]

        # Tokens can attend to each other if they have the same non-zero sample ID
        bidirectional_mask = (mask_i == mask_j) & (mask_i > 0)

        # Add head dimension: [batch_size, 1, seq_len, seq_len]
        bidirectional_mask = bidirectional_mask.unsqueeze(1)

        return bidirectional_mask

    def _compute_diffusion_loss(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | Any]:
        """
        Compute diffusion loss.

        Args:
            model: The model to compute loss for.
            input_ids: Ground truth token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            loss: Cross-entropy loss.
            metrics: Dictionary of metrics.
        """
        # Apply forward process
        noisy_batch, masked_indices, p_mask = self._forward_process(
            input_ids, attention_mask, self._config.eps
        )

        # Create bidirectional attention mask
        bidirectional_mask = self._create_bidirectional_attention_mask(
            input_ids, attention_mask
        )

        # Forward pass
        outputs = model(
            input_ids=noisy_batch,
            attention_mask=bidirectional_mask,
        )
        logits = outputs.logits

        if masked_indices.sum() > 0:
            valid_indices = torch.where(masked_indices)
            batch_indices, seq_indices = valid_indices

            # Extract the relevant data
            masked_logits = logits[
                batch_indices, seq_indices
            ]  # [num_masked_tokens, vocab_size]
            masked_targets = input_ids[
                batch_indices, seq_indices
            ]  # [num_masked_tokens]
            masked_p_mask = p_mask[batch_indices, seq_indices]  # [num_masked_tokens]

            # Compute cross-entropy loss without reduction (cast to fp32 for stability)
            token_loss = F.cross_entropy(
                masked_logits.float(), masked_targets, reduction="none"
            )

            # Apply importance weighting if enabled
            if self._config.importance_weighting:
                masked_p_mask = masked_p_mask.float()
                weighted_loss = token_loss / masked_p_mask
            else:
                weighted_loss = token_loss

            # Final loss: sum weighted losses, normalize by total tokens
            loss = weighted_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            ce_loss = token_loss.mean()

            # Compute accuracy on masked tokens
            with torch.no_grad():
                pred_tokens = masked_logits.argmax(dim=-1)
                accuracy = (pred_tokens == masked_targets).float().mean()
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            accuracy = torch.tensor(0.0, device=input_ids.device)
            ce_loss = torch.tensor(0.0, device=input_ids.device)
            masked_p_mask = torch.tensor(1.0, device=input_ids.device)

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "mask_ratio": masked_indices.float().mean().item(),
            "num_masked_tokens": (masked_indices.sum().item(), "sum"),
            "avg_p_mask": p_mask[masked_indices].mean().item(),
            "ce_loss": ce_loss.item(),
        }

        if self._config.importance_weighting:
            metrics["importance_weight_avg"] = (1.0 / masked_p_mask).mean().item()

        train_eval: Literal["train", "eval"] = "train" if model.training else "eval"
        self.store_metrics(metrics, train_eval=train_eval)

        return loss, outputs
