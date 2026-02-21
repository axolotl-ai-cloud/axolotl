"""Sample generation utilities for SFT/Pretrain training."""

from typing import Any, List, Optional

import torch
from accelerate.utils import extract_model_from_parallel
from colorama import Fore, Style

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def generate_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    dataloader: Any,
    num_generation_samples: int = 3,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    do_sample: bool = True,
    prompt_ratio: float = 0.5,
) -> List[dict]:
    """
    Generate samples from the model during training for monitoring.

    Args:
        model: The model to generate from
        tokenizer: The tokenizer to use for encoding/decoding
        dataloader: Dataloader to sample prompts from
        num_generation_samples: Number of samples to generate
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (0.0 = greedy)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling vs greedy decoding
        prompt_ratio: Ratio of sequence to use as prompt (0.0-1.0)

    Returns:
        List of dicts with 'prompt', 'generated', and 'full_text' keys
    """
    unwrapped_model = extract_model_from_parallel(model)

    training = unwrapped_model.training
    unwrapped_model.eval()

    device = next(unwrapped_model.parameters()).device

    generations = []

    try:
        with torch.no_grad():
            samples_collected = 0

            for batch in dataloader:
                if samples_collected >= num_generation_samples:
                    break

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                batch_size = input_ids.shape[0]

                indices = torch.randperm(batch_size)[
                    : num_generation_samples - samples_collected
                ]

                for idx in indices:
                    if samples_collected >= num_generation_samples:
                        break

                    sequence = input_ids[idx]

                    if attention_mask is not None:
                        seq_len = attention_mask[idx].sum().item()
                    else:
                        seq_len = sequence.shape[0]

                    if seq_len < 5:
                        continue

                    prompt_len = max(1, int(seq_len * prompt_ratio))
                    prompt_ids = sequence[:prompt_len].unsqueeze(0)

                    try:
                        generation_config = {
                            "max_new_tokens": max_new_tokens,
                            "do_sample": do_sample,
                            "pad_token_id": tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id,
                        }

                        if do_sample:
                            generation_config["temperature"] = temperature
                            if top_p is not None:
                                generation_config["top_p"] = top_p
                            if top_k is not None:
                                generation_config["top_k"] = top_k

                        generated_ids = unwrapped_model.generate(
                            prompt_ids, **generation_config
                        )

                        prompt_text = tokenizer.decode(
                            prompt_ids[0], skip_special_tokens=True
                        )
                        generated_text = tokenizer.decode(
                            generated_ids[0][prompt_len:], skip_special_tokens=True
                        )
                        full_text = tokenizer.decode(
                            generated_ids[0], skip_special_tokens=True
                        )

                        generations.append(
                            {
                                "prompt": prompt_text,
                                "generated": generated_text,
                                "full_text": full_text,
                            }
                        )

                        samples_collected += 1

                    except Exception as e:
                        LOG.warning(f"Failed to generate sample: {e}", exc_info=True)
                        continue

    except Exception as e:
        LOG.warning(f"Error during sample generation: {e}", exc_info=True)

    if training:
        unwrapped_model.train()
    else:
        unwrapped_model.eval()

    return generations


def format_generation_for_logging(
    sample: dict, sample_idx: int, step: int
) -> tuple[str, str]:
    """
    Format a generation sample for pretty logging.

    Args:
        sample: Dict with 'prompt', 'generated', and 'full_text' keys
        sample_idx: Index of the sample
        step: Current training step

    Returns:
        Tuple of (console_text, wandb_text)
    """
    console_text = (
        f"\n{Style.BRIGHT}{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n"
        f"{Style.BRIGHT}{Fore.GREEN}Sample {sample_idx + 1} (Step {step}){Style.RESET_ALL}\n"
        f"{Style.BRIGHT}{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n"
        f"{Style.BRIGHT}{Fore.YELLOW}[PROMPT]{Style.RESET_ALL}\n{sample['prompt']}\n\n"
        f"{Style.BRIGHT}{Fore.MAGENTA}[GENERATED]{Style.RESET_ALL}\n{sample['generated']}\n"
        f"{Style.BRIGHT}{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n"
    )
    wandb_text = (
        f"\n{'=' * 80}\n"
        f"Sample {sample_idx + 1} (Step {step})\n"
        f"{'=' * 80}\n"
        f"[PROMPT]\n{sample['prompt']}\n\n"
        f"[GENERATED]\n{sample['generated']}\n"
        f"{'=' * 80}\n"
    )

    return console_text, wandb_text
