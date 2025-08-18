"""Sample generation utilities for diffusion training."""

import logging
from typing import Any, List, Optional

import torch

logger = logging.getLogger(__name__)


def generate_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    val_dataloader: Optional[Any] = None,
    num_generation_samples: int = 3,
    max_length: int = 100,
    num_diffusion_steps: int = 128,
    temperature: float = 0.0,
    mask_token_id: int = 32000,
) -> List[dict]:
    """
    Generate text samples using the diffusion model by randomly masking sequences
    from the validation dataset and running the reverse diffusion process.

    Args:
        model: The wrapped or unwrapped model
        tokenizer: Tokenizer for encoding/decoding
        val_dataloader: Validation dataloader (for sampling sequences)
        num_generation_samples: Number of samples to generate
        max_length: Maximum length of sequences to use
        num_diffusion_steps: Number of diffusion steps for generation
        temperature: Temperature for sampling (0.0 = deterministic)
        mask_token_id: Token ID used for masking

    Returns:
        List of dictionaries with original text, masked text, and generated text
    """
    if val_dataloader is None:
        logger.warning("No validation dataloader provided, cannot generate samples")
        return []

    # Get the actual model (unwrap if needed)
    unwrapped_model = model.module if hasattr(model, "module") else model
    unwrapped_model.eval()
    generations = []

    # Sample sequences from validation dataset
    sampled_sequences = _sample_sequences_from_dataloader(
        val_dataloader, num_generation_samples, max_length, unwrapped_model.device
    )
    logger.info(f"Sampled {len(sampled_sequences)} sequences from validation dataset")

    # Generate samples using reverse diffusion process
    with torch.no_grad():
        for original_sequence in sampled_sequences:
            generation_result = _generate(
                unwrapped_model,
                tokenizer,
                original_sequence,
                num_diffusion_steps,
                temperature,
                mask_token_id,
            )
            generations.append(generation_result)

    unwrapped_model.train()
    return generations


def _sample_sequences_from_dataloader(
    val_dataloader: Any, num_samples: int, max_length: int, device: torch.device
) -> List[torch.Tensor]:
    """Sample sequences from validation dataloader."""
    sampled_sequences = []
    sample_count = 0

    # Add randomness by skipping a random number of batches
    skip_batches = torch.randint(0, 6, (1,)).item()
    batch_count = 0

    for batch in val_dataloader:
        # Skip some batches for variety
        if batch_count < skip_batches:
            batch_count += 1
            continue

        if sample_count >= num_samples:
            break

        batch_count += 1
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        # Randomly sample from sequences in this batch
        batch_indices = torch.randperm(input_ids.size(0)).tolist()

        for i in batch_indices:
            if sample_count >= num_samples:
                break

            # Get actual sequence length (non-padded)
            if attention_mask is not None:
                seq_len = attention_mask[i].sum().item()
            else:
                seq_len = input_ids.size(1)

            # Limit sequence length to max_length
            actual_length = min(seq_len, max_length)
            if actual_length < 10:  # Skip very short sequences
                continue

            # Extract the sequence
            sequence = input_ids[i][:actual_length].unsqueeze(0).to(device)
            sampled_sequences.append(sequence)
            sample_count += 1

    return sampled_sequences


def _generate(
    model: torch.nn.Module,
    tokenizer: Any,
    original_sequence: torch.Tensor,
    num_diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
) -> dict:
    """Generate a single sample using reverse diffusion."""
    # Get original text for comparison
    original_text = tokenizer.decode(
        original_sequence[0].cpu(), skip_special_tokens=True
    )

    # Apply custom masking with random ratio (10% to 70%)
    total_tokens = original_sequence.size(1)
    min_ratio, max_ratio = 0.1, 0.7
    target_mask_ratio = torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
    target_masked_tokens = int(total_tokens * target_mask_ratio)

    # Create random mask indices
    mask_positions = torch.randperm(total_tokens)[:target_masked_tokens]
    masked_indices = torch.zeros(
        1, total_tokens, dtype=torch.bool, device=original_sequence.device
    )
    masked_indices[0, mask_positions] = True

    # Create masked sequence
    masked_sequence = original_sequence.clone()
    masked_sequence[masked_indices] = mask_token_id

    # Calculate actual mask ratio
    masked_tokens = masked_indices.sum().item()
    mask_ratio = masked_tokens / total_tokens

    # Get masked text for comparison
    masked_text = tokenizer.decode(masked_sequence[0].cpu(), skip_special_tokens=False)
    # Clean up mask token representation
    masked_text = _clean_masked_text(masked_text, tokenizer, mask_token_id)

    # Run reverse diffusion process
    sequence = masked_sequence.clone()
    for step in range(num_diffusion_steps):
        sequence = _diffusion_step(
            model, sequence, step, num_diffusion_steps, temperature, mask_token_id
        )

    # Get final generated text
    generated_text = tokenizer.decode(sequence[0].cpu(), skip_special_tokens=True)

    return {
        "original": original_text,
        "masked": masked_text,
        "generated": generated_text,
        "mask_ratio": mask_ratio,
        "masked_tokens": masked_tokens,
        "total_tokens": total_tokens,
        "formatted": (
            f"Original: '{original_text}' → Masked: '{masked_text}' "
            f"({mask_ratio:.1%}) → Generated: '{generated_text}'"
        ),
    }


def _clean_masked_text(masked_text: str, tokenizer: Any, mask_token_id: int) -> str:
    """Clean up masked text for display."""
    # Get the mask token representation from the tokenizer
    mask_token_repr = tokenizer.decode([mask_token_id], skip_special_tokens=False)
    cleaned = masked_text.replace(mask_token_repr, "[MASK]")

    # Clean up special tokens and whitespace
    cleaned = cleaned.replace("<s>", "").replace("</s>", "").strip()
    cleaned = " ".join(cleaned.split())

    return cleaned


def _diffusion_step(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    step: int,
    num_diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
) -> torch.Tensor:
    """Perform a single diffusion step with remasking."""
    # Only process if there are masked tokens remaining
    current_mask = sequence == mask_token_id
    if not current_mask.any():
        return sequence

    # Create bidirectional attention mask for diffusion
    batch_size, seq_len = sequence.shape
    attention_mask = torch.ones(
        batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=sequence.device
    )

    # Forward pass
    outputs = model(input_ids=sequence, attention_mask=attention_mask)
    logits = outputs.logits

    # Only sample at currently masked positions
    if current_mask.any():
        masked_logits = logits[current_mask]

        # Apply temperature scaling
        if temperature > 0:
            scaled_logits = masked_logits / temperature
        else:
            scaled_logits = masked_logits

        # Suppress mask token in outputs
        scaled_logits[:, mask_token_id] = -float("inf")

        # Sample predictions
        if temperature > 0:
            # Add Gumbel noise for sampling
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scaled_logits, dtype=torch.float32))
            )
            gumbel_logits = scaled_logits + gumbel_noise
            predicted_tokens = torch.argmax(gumbel_logits, dim=-1)
        else:
            # Deterministic sampling when temperature is 0
            predicted_tokens = torch.argmax(scaled_logits, dim=-1)

        # Calculate probabilities for confidence scoring
        probs = torch.softmax(scaled_logits, dim=-1)
        predicted_token_probs = probs[range(len(predicted_tokens)), predicted_tokens]

        # Determine how many tokens to unmask this step
        remaining_masked = current_mask.sum().item()
        if step == num_diffusion_steps - 1:
            num_to_unmask = remaining_masked
        else:
            unmask_ratio = 1.0 / (num_diffusion_steps - step)
            num_to_unmask = max(1, int(remaining_masked * unmask_ratio))

        # Select highest confidence predictions to unmask
        if num_to_unmask >= remaining_masked:
            sequence[current_mask] = predicted_tokens
        else:
            _, top_indices = predicted_token_probs.topk(num_to_unmask)
            mask_positions = torch.where(current_mask)[1]
            positions_to_unmask = mask_positions[top_indices]
            sequence[0, positions_to_unmask] = predicted_tokens[top_indices]

    return sequence
