"""Sample generation utilities for diffusion training."""

import re
from typing import Any, List, Literal, Optional

import torch

from axolotl.utils.logging import get_logger

from .utils import create_bidirectional_attention_mask, shift_logits_to_input_positions

LOG = get_logger(__name__)


def generate_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: Optional[Any] = None,
    num_generation_samples: int = 3,
    max_length: int = 100,
    num_diffusion_steps: int = 128,
    temperature: float = 0.0,
    mask_token_id: int = 32000,
    mode: Literal["random", "completion"] = "random",
    completion_tokens: int = 0,
    target_mask_ratio: Optional[float] = None,
) -> List[dict]:
    """
    Generate text samples using the diffusion model by randomly masking sequences from
    the given dataset and running the reverse diffusion process.

    Args:
        model: The wrapped or unwrapped model
        tokenizer: Tokenizer for encoding/decoding
        dataloader: Validation dataloader (for sampling sequences)
        num_generation_samples: Number of samples to generate
        max_length: Maximum length of sequences to use
        num_diffusion_steps: Number of diffusion steps for generation
        temperature: Temperature for sampling (0.0 = deterministic)
        mask_token_id: Token ID used for masking

    Returns:
        List of dictionaries with original text, masked text, and generated text
    """
    if dataloader is None:
        LOG.warning("No validation dataloader provided, cannot generate samples")
        return []

    unwrapped_model = model.module if hasattr(model, "module") else model
    training = unwrapped_model.training
    unwrapped_model.eval()

    # Resolve device robustly (some modules don't expose `.device`)
    device = getattr(unwrapped_model, "device", None)
    if device is None:
        try:
            device = next(unwrapped_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    generations = []

    # Sample sequences from validation dataset
    sampled_sequences = _sample_sequences_from_dataloader(
        dataloader, num_generation_samples, max_length, device
    )
    LOG.info(f"Sampled {len(sampled_sequences)} sequences from validation dataset")

    # Generate samples using reverse diffusion process
    with torch.no_grad():
        for sample in sampled_sequences:
            if isinstance(sample, dict):
                original_sequence = sample.get("input_ids")
                labels_seq = sample.get("labels")
                attn_seq = sample.get("attention_mask")
            else:
                original_sequence = sample
                labels_seq = None
                attn_seq = None
            generation_result = generate(
                unwrapped_model,
                tokenizer,
                original_sequence,
                num_diffusion_steps,
                temperature,
                mask_token_id,
                mode=mode,
                completion_tokens=completion_tokens,
                target_mask_ratio=target_mask_ratio,
                labels=labels_seq,
                attention_mask=attn_seq,
            )
            generations.append(generation_result)

    # Restore prior training state
    if training:
        unwrapped_model.train()
    else:
        unwrapped_model.eval()

    return generations


def _sample_sequences_from_dataloader(
    dataloader: Any, num_samples: int, max_length: int, device: torch.device
) -> List[Any]:
    """Sample sequences from validation dataloader."""
    sampled_sequences: list[dict[str, torch.Tensor] | torch.Tensor] = []
    sample_count = 0

    # Skip a random number of batches (we could be more clever about this)
    skip_batches = torch.randint(0, 10, (1,)).item()
    batch_count = 0

    for batch in dataloader:
        # Skip some batches for variety
        if batch_count < skip_batches:
            batch_count += 1
            continue

        if sample_count >= num_samples:
            break

        batch_count += 1
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")

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

            if seq_len < 10:
                continue

            # Determine truncation length
            max_total = min(seq_len, max_length)
            if labels is not None:
                labels_i = labels[i][:seq_len]
                answer_mask = labels_i != -100
                if not answer_mask.any():
                    # No answer tokens; skip for SFT masking
                    continue
                first_ans_idx = int(
                    torch.nonzero(answer_mask, as_tuple=False)[0].item()
                )
                prompt_len = first_ans_idx
                if prompt_len >= max_total:
                    # Prompt alone reaches cap; cannot include any answer
                    continue
                remaining_answer = int(answer_mask[prompt_len:].sum().item())
                allowed_answer = max_total - prompt_len
                take_answer = min(remaining_answer, allowed_answer)
                if take_answer <= 0:
                    continue
                actual_length = prompt_len + take_answer
            else:
                actual_length = max_total

            # Extract the (possibly truncated) sequence
            sequence = input_ids[i][:actual_length].unsqueeze(0).to(device)
            attn_seq = (
                attention_mask[i][:actual_length].unsqueeze(0).to(device)
                if attention_mask is not None
                else None
            )
            if labels is not None:
                labels_seq = labels[i][:actual_length].unsqueeze(0).to(device)
                sampled_sequences.append(
                    {
                        "input_ids": sequence,
                        "labels": labels_seq,
                        "attention_mask": attn_seq,
                    }
                )
            else:
                if attn_seq is not None:
                    sampled_sequences.append(
                        {"input_ids": sequence, "attention_mask": attn_seq}
                    )
                else:
                    sampled_sequences.append(sequence)
            sample_count += 1

    return sampled_sequences


def generate(
    model: torch.nn.Module,
    tokenizer: Any,
    original_sequence: torch.Tensor,
    num_diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
    *,
    mode: Literal["random", "completion"] = "random",
    completion_tokens: int = 0,
    target_mask_ratio: Optional[float] = None,
    labels: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> dict:
    """Generate a single sample using reverse diffusion."""
    # Get original text for comparison
    original_text = tokenizer.decode(
        original_sequence[0].cpu(), skip_special_tokens=True
    )

    # Build masked sequence
    if (
        labels is not None
        and labels.numel() > 0
        and (labels == -100).any()
        and (labels != -100).any()
    ):
        # SFT case: completely mask all answer tokens (labels != -100)
        total_tokens = original_sequence.size(1)
        masked_indices = (labels != -100).to(dtype=torch.bool)
        masked_sequence = original_sequence.clone()
        masked_sequence[masked_indices] = mask_token_id
        masked_tokens = int(masked_indices.sum().item())
        mask_ratio = masked_tokens / max(int(total_tokens), 1)
    elif mode == "completion" and completion_tokens > 0:
        # Append mask tokens to the right for completion
        total_tokens = original_sequence.size(1) + int(completion_tokens)
        masked_indices = torch.zeros(
            1, total_tokens, dtype=torch.bool, device=original_sequence.device
        )
        masked_indices[0, -int(completion_tokens) :] = True

        append = torch.full(
            (1, int(completion_tokens)), mask_token_id, device=original_sequence.device
        )
        masked_sequence = torch.cat([original_sequence, append], dim=1)
        masked_tokens = int(completion_tokens)
        mask_ratio = masked_tokens / total_tokens
    else:
        # Apply random masking with optional fixed ratio
        total_tokens = original_sequence.size(1)
        if target_mask_ratio is None:
            min_ratio, max_ratio = 0.1, 0.7
            target_mask_ratio = (
                torch.rand(1).item() * (max_ratio - min_ratio) + min_ratio
            )
        target_masked_tokens = max(1, int(total_tokens * float(target_mask_ratio)))

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
    masked_text = _clean_masked_text(masked_text, tokenizer, mask_token_id)

    # Run reverse diffusion process
    sequence = masked_sequence.clone()
    attention_mask = create_bidirectional_attention_mask(
        sequence, attention_mask, sample_packing=attention_mask is not None
    )
    for step in range(num_diffusion_steps):
        sequence = _diffusion_step(
            model,
            sequence,
            step,
            num_diffusion_steps,
            temperature,
            mask_token_id,
            attention_mask,
        )
    generated_text = tokenizer.decode(sequence[0].cpu(), skip_special_tokens=True)

    # Collect diagnostic info
    final_ids = sequence[0].detach().cpu().tolist()
    orig_ids_for_render = original_sequence[0].detach().cpu().tolist()
    if masked_indices is not None:
        masked_positions = (
            torch.where(masked_indices[0])[0].detach().cpu().tolist()
            if masked_indices.ndim == 2
            else []
        )
    else:
        masked_positions = []

    result = {
        "original": original_text,
        "masked": masked_text,
        "generated": generated_text,
        "mask_ratio": mask_ratio,
        "masked_tokens": masked_tokens,
        "total_tokens": total_tokens,
        "generated_ids": final_ids,
        "masked_positions": masked_positions,
        "orig_ids": orig_ids_for_render,
        "formatted": (
            f"Original: '{original_text}' → Masked: '{masked_text}' "
            f"({mask_ratio:.1%}) → Generated: '{generated_text}'"
        ),
    }

    return result


def _clean_masked_text(masked_text: str, tokenizer: Any, mask_token_id: int) -> str:
    """Clean up masked text for display."""
    mask_token_repr = tokenizer.decode([mask_token_id], skip_special_tokens=False)
    cleaned = masked_text.replace(mask_token_repr, "[MASK]")

    # Remove literal special token strings
    if hasattr(tokenizer, "special_tokens_map"):
        for token_value in tokenizer.special_tokens_map.values():
            if token_value and isinstance(token_value, str):
                cleaned = cleaned.replace(token_value, "")

    # Normalize whitespace but preserve newlines
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.split("\n")).strip()
    return cleaned


def _diffusion_step(
    model: torch.nn.Module,
    sequence: torch.Tensor,
    step: int,
    num_diffusion_steps: int,
    temperature: float,
    mask_token_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Perform a single diffusion step with remasking."""
    # Only process if there are masked tokens remaining
    current_mask = sequence == mask_token_id
    if not current_mask.any():
        return sequence

    # Create or use provided attention mask
    if attention_mask is None:
        batch_size, seq_len = sequence.shape
        attention_mask = torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=sequence.device
        )

    # Forward pass
    outputs = model(input_ids=sequence, attention_mask=attention_mask)
    logits = shift_logits_to_input_positions(outputs.logits)

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

        if temperature > 0:
            # Add Gumbel noise for sampling
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(scaled_logits, dtype=torch.float32))
            )
            gumbel_logits = scaled_logits + gumbel_noise
            predicted_tokens = torch.argmax(gumbel_logits, dim=-1)
        else:
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
