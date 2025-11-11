"""Shared utilities for diffusion integration."""

from __future__ import annotations

from typing import Any, Optional

import torch

from axolotl.utils.dict import DictDefault


def resolve_mask_token_id(
    tokenizer: Any,
    cfg: DictDefault,
    *,
    allow_add: bool,
    model: Any | None = None,
    default_token: str = "<|diffusion_mask|>",
) -> int:
    """Resolve mask token id. Training may add a new special token; inference won't."""
    # Determine vocab size if available
    vocab_size = None
    if tokenizer is not None:
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
            try:
                vocab_size = int(tokenizer.vocab_size)  # type: ignore[arg-type]
            except Exception:
                vocab_size = None
        elif hasattr(tokenizer, "__len__"):
            try:
                vocab_size = int(len(tokenizer))
            except Exception:
                vocab_size = None

    # Use explicit id from config if provided
    diffusion_cfg = getattr(cfg, "diffusion", None)
    # Fallback to top-level attr names only if nested missing (shouldn't happen)
    cfg_id = (
        getattr(diffusion_cfg, "mask_token_id", None)
        if diffusion_cfg is not None
        else getattr(cfg, "diffusion_mask_token_id", None)
    )
    if isinstance(cfg_id, int) and cfg_id >= 0:
        if vocab_size is None or cfg_id < vocab_size:
            return int(cfg_id)

    def _existing_special_token_id(token_str: str | None) -> int | None:
        """Attempt to resolve an existing special token string to a real ID."""
        if not token_str or not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
        except Exception:
            return None

        if not isinstance(token_id, int) or token_id < 0:
            return None

        # Ensure it's registered as special and not UNK, and within vocab
        unk_id = getattr(tokenizer, "unk_token_id", None)
        specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
        addl = set(getattr(tokenizer, "additional_special_tokens", []) or [])
        is_special = token_str in specials or token_str in addl
        in_vocab = vocab_size is None or token_id < vocab_size
        if (
            (unk_id is not None and token_id == unk_id)
            or not is_special
            or not in_vocab
        ):
            return None
        return token_id

    # Try mask token string if provided
    token_str = (
        getattr(diffusion_cfg, "mask_token_str", None)
        if diffusion_cfg is not None
        else getattr(cfg, "diffusion_mask_token_str", None)
    )
    for candidate in (token_str, default_token):
        token_id = _existing_special_token_id(candidate)
        if isinstance(token_id, int):
            try:
                if diffusion_cfg is None:
                    cfg.diffusion_mask_token_id = int(token_id)  # legacy fallback
                else:
                    diffusion_cfg.mask_token_id = int(token_id)
            except Exception:
                pass
            return int(token_id)

    # Optionally add and return a dedicated special token during training
    if allow_add and hasattr(tokenizer, "add_special_tokens"):
        token_to_add = token_str or default_token
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": [token_to_add]})

            # Resize embeddings if possible
            if (
                model is not None
                and hasattr(tokenizer, "__len__")
                and hasattr(model, "resize_token_embeddings")
            ):
                try:
                    model.resize_token_embeddings(len(tokenizer))
                except Exception:
                    pass
            new_id = tokenizer.convert_tokens_to_ids(token_to_add)
            if isinstance(new_id, int) and new_id >= 0:
                try:
                    if diffusion_cfg is None:
                        cfg.diffusion_mask_token_id = int(new_id)  # legacy fallback
                    else:
                        diffusion_cfg.mask_token_id = int(new_id)
                except Exception:
                    pass
                return int(new_id)
        except Exception:
            pass

    # Fallback to unk or 0 (do not update cfg)
    fallback = getattr(tokenizer, "unk_token_id", 0) or 0
    return int(fallback)


def create_bidirectional_attention_mask(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    sample_packing: bool = False,
) -> torch.Tensor:
    """
    Create bidirectional attention mask to override default causal masking.
    Handles sample-packed sequences where different samples are identified
    by different attention mask values.

    Args:
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Attention mask [batch_size, seq_len]
        sample_packing: Whether sample packing is enabled

    Returns:
        bidirectional_mask: 4D attention mask [batch_size, 1, seq_len, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    if attention_mask is None or not sample_packing:
        return torch.ones(
            batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=device
        )

    # Handle sample packing: tokens can only attend within their sample
    mask_i = attention_mask.unsqueeze(2)  # [batch_size, seq_len, 1]
    mask_j = attention_mask.unsqueeze(1)  # [batch_size, 1, seq_len]

    # Tokens can attend to each other if they have the same non-zero sample ID
    bidirectional_mask = (mask_i == mask_j) & (mask_i > 0)

    # Add head dimension: [batch_size, 1, seq_len, seq_len]
    return bidirectional_mask.unsqueeze(1)


def shift_logits_to_input_positions(logits: torch.Tensor) -> torch.Tensor:
    """Align next-token logits with their input token positions for diffusion."""
    if logits.size(1) <= 1:
        return logits
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
