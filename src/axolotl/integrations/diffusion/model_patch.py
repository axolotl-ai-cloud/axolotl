"""Model patches for diffusion training."""

import torch


def patch_model_for_bidirectional_attention(model):
    """
    Patch model to handle diffusion training with forward process and bidirectional
    attention.

    This monkey-patches the model's forward method to:
    - Apply forward diffusion process (masking) during training
    - Use bidirectional attention masks
    - Store info for loss computation
    """
    original_forward = model.forward

    def diffusion_forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ):
        # Check if this is diffusion training
        if (
            hasattr(self.config, "loss_type")
            and self.config.loss_type == "ForDiffusionLM"
            and self.training
        ):

            # Store original input_ids for loss computation
            original_input_ids = input_ids.clone()

            # Apply forward diffusion process (masking)
            diffusion_config = getattr(self.config, "diffusion_config", {})
            noisy_input_ids, masked_indices, p_mask = _forward_process(
                input_ids, attention_mask, labels, diffusion_config
            )

            # Use noisy input for model forward
            input_ids = noisy_input_ids

            # Convert attention mask to bidirectional
            if attention_mask is not None:
                attention_mask = _create_bidirectional_attention_mask(
                    input_ids, attention_mask
                )

            # Store diffusion info in the model for loss computation
            self._diffusion_info = {
                "original_input_ids": original_input_ids,
                "masked_indices": masked_indices,
                "p_mask": p_mask,
            }

        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs
        )

    # Replace the forward method
    model.forward = diffusion_forward.__get__(model, model.__class__)


def _create_bidirectional_attention_mask(
    input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """
    Create bidirectional attention mask from 2D attention mask.

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: 2D attention mask [batch_size, seq_len]

    Returns:
        bidirectional_mask: 4D attention mask [batch_size, 1, seq_len, seq_len]
    """
    batch_size, seq_len = input_ids.shape

    # Simple bidirectional mask - all tokens can attend to all valid tokens
    # Expand 2D mask to 4D: [batch_size, seq_len] -> [batch_size, 1, seq_len, seq_len]
    bidirectional_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
    bidirectional_mask = bidirectional_mask.expand(batch_size, 1, seq_len, seq_len)

    # Apply row-wise masking (padded tokens can't attend to anything)
    row_mask = attention_mask.unsqueeze(1).unsqueeze(3)  # [B, 1, S, 1]
    bidirectional_mask = bidirectional_mask & row_mask

    return bidirectional_mask


def _forward_process(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    diffusion_config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply forward diffusion process (random masking).

    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        labels: Labels for SFT training [batch_size, seq_len]
        diffusion_config: Diffusion configuration dict

    Returns:
        noisy_input_ids: Input with masked tokens
        masked_indices: Boolean mask of which tokens were masked
        p_mask: Masking probabilities used
    """
    if diffusion_config is None:
        diffusion_config = {}

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    eps = diffusion_config.get("eps", 1e-3)
    mask_token_id = diffusion_config.get("mask_token_id", 128002)

    # Sample random timesteps for each sample
    t = torch.rand(batch_size, device=device)

    # Calculate masking probability with epsilon
    p_mask = (1 - eps) * t + eps  # [batch_size]
    p_mask = p_mask.unsqueeze(1).expand(-1, seq_len)  # [batch_size, seq_len]

    # Don't mask padding tokens
    if attention_mask is not None:
        p_mask = p_mask * attention_mask.float()

    # Create random mask based on p_mask
    random_values = torch.rand_like(p_mask)
    masked_indices = random_values < p_mask

    # Apply attention mask constraints
    if attention_mask is not None:
        masked_indices = masked_indices & attention_mask.bool()

    # For SFT data, only mask answer tokens (where labels != -100)
    if labels is not None:
        answer_mask = labels != -100
        masked_indices = masked_indices & answer_mask

    # Create noisy input by replacing masked tokens
    noisy_input_ids = input_ids.clone()
    noisy_input_ids[masked_indices] = mask_token_id

    return noisy_input_ids, masked_indices, p_mask
