"""Shared utilities for diffusion integration."""

from __future__ import annotations

from typing import Any

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
            except Exception:  # pragma: no cover
                vocab_size = None
        elif hasattr(tokenizer, "__len__"):
            try:
                vocab_size = int(len(tokenizer))
            except Exception:  # pragma: no cover
                vocab_size = None

    # Use explicit id from config if valid
    cfg_id = (
        getattr(cfg, "mask_token_id", None)
        if hasattr(cfg, "mask_token_id")
        else cfg.get("mask_token_id")
    )
    if isinstance(cfg_id, int) and cfg_id >= 0:
        if vocab_size is None or cfg_id < vocab_size:
            return int(cfg_id)

    def _existing_special_token_id(token_str: str | None) -> int | None:
        """Attempt to resolve an existing special token string to a real ID."""
        if not token_str or not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        try:
            tid = tokenizer.convert_tokens_to_ids(token_str)
        except Exception:  # pragma: no cover
            return None

        if not isinstance(tid, int) or tid < 0:
            return None

        # Ensure it's registered as special and not UNK, and within vocab
        unk_id = getattr(tokenizer, "unk_token_id", None)
        specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
        addl = set(getattr(tokenizer, "additional_special_tokens", []) or [])
        is_special = token_str in specials or token_str in addl
        in_vocab = vocab_size is None or tid < vocab_size
        if (unk_id is not None and tid == unk_id) or not is_special or not in_vocab:
            return None
        return tid

    # Try mask_token_str from config, else the default training token
    token_str = (
        getattr(cfg, "mask_token_str", None)
        if hasattr(cfg, "mask_token_str")
        else cfg.get("mask_token_str")
    )
    for candidate in (token_str, default_token):
        tid = _existing_special_token_id(candidate)
        if isinstance(tid, int):
            cfg.mask_token_id = int(tid)
            return int(tid)

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
                except Exception:  # pragma: no cover
                    pass
            new_id = tokenizer.convert_tokens_to_ids(token_to_add)
            if isinstance(new_id, int) and new_id >= 0:
                cfg.mask_token_id = int(new_id)
                return int(new_id)
        except Exception:  # pragma: no cover
            pass

    # Fallback to unk or 0 (do not update cfg)
    fallback = getattr(tokenizer, "unk_token_id", 0) or 0
    return int(fallback)
