"""Custom trainer for diffusion LM training."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class DiffusionTrainer(AxolotlTrainer):  # pylint: disable=too-many-ancestors
    """Custom trainer for diffusion LM training that overrides loss computation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = None

    def set_config(self, config: DictDefault):
        """Set config for diffusion training."""
        self.config = config

    def forward_process(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        eps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Create random mask based on probability
        masked_indices = torch.rand((batch_size, seq_len), device=device) < p_mask
        if attention_mask is not None:
            masked_indices = masked_indices & attention_mask.bool()

        # Get tokenizer
        tokenizer = self.processing_class
        assert tokenizer is not None, "Tokenizer not available on Trainer object."

        # Get mask token ID
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        if mask_token_id is None:
            mask_token_id = getattr(tokenizer, "unk_token_id", None)

        # Create masked input using configured mask token
        noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)

        return noisy_batch, masked_indices, p_mask

    def create_bidirectional_attention_mask(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Create bidirectional attention mask to override default causal masking.

        Args:
            input_ids: Input token ids [batch_size, seq_len].
            attention_mask: Attention mask [batch_size, seq_len].

        Returns:
            bidirectional_mask: 4D attention mask [batch_size, 1, seq_len, seq_len].
        """
        batch_size, seq_len = input_ids.shape

        # Create bidirectional attention mask to override default causal masking
        # Shape: [batch_size, 1, seq_len, seq_len]
        bidirectional_mask = torch.ones(
            seq_len, seq_len, dtype=torch.bool, device=input_ids.device
        )
        bidirectional_mask = (
            bidirectional_mask.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 1, seq_len, seq_len)
        )

        # Apply padding mask if provided
        if attention_mask is not None:
            # Convert attention_mask to 4D and apply
            expanded_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
            expanded_mask = expanded_mask.expand(batch_size, 1, seq_len, seq_len)

            bidirectional_mask = (
                bidirectional_mask & expanded_mask & expanded_mask.transpose(-1, -2)
            )

        return bidirectional_mask

    def compute_diffusion_loss(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
        noisy_batch, masked_indices, p_mask = self.forward_process(
            input_ids, attention_mask, self.config.eps
        )

        # Create bidirectional attention mask (always required for diffusion training)
        bidirectional_mask = self.create_bidirectional_attention_mask(
            input_ids, attention_mask
        )

        # Forward pass
        outputs = model(
            input_ids=noisy_batch,
            attention_mask=bidirectional_mask,
        )
        logits = outputs.logits

        # Apply attention mask to masked_indices if provided
        if attention_mask is not None:
            loss_mask = masked_indices & attention_mask.bool()
        else:
            loss_mask = masked_indices

        if loss_mask.sum() > 0:
            valid_indices = torch.where(loss_mask)
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
            if self.config.importance_weighting:
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
            "num_masked_tokens": loss_mask.sum().item(),
            "avg_p_mask": (
                p_mask[masked_indices].mean().item()
                if masked_indices.sum() > 0
                else 0.0
            ),
            "ce_loss": ce_loss.item() if loss_mask.sum() > 0 else 0.0,
        }

        if self.config.importance_weighting:
            metrics["importance_weight_avg"] = (
                (1.0 / masked_p_mask).mean().item() if loss_mask.sum() > 0 else 0.0
            )

        return loss, metrics

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Override compute_loss to use diffusion loss."""
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        if input_ids is None:
            raise ValueError("input_ids is required for diffusion training")

        loss, metrics = self.compute_diffusion_loss(model, input_ids, attention_mask)

        # Log metrics
        if self.state.is_local_process_zero:
            for key, value in metrics.items():
                self.log({f"train/diffusion_{key}": value})

        if return_outputs:
            # TODO: compute outputs (?)
            outputs = [loss]
            return (loss, outputs)

        return loss
