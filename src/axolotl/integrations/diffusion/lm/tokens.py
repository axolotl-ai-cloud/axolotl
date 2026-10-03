"""Mask-token resolution for diffusion training."""

from __future__ import annotations

from typing import Any

from axolotl.utils.dict import DictDefault

from .config import get_diffusion_config


def _set_mask_token_id(
    cfg: DictDefault, diffusion_cfg: Any | None, token_id: int
) -> None:
    try:
        if diffusion_cfg is None:
            cfg.diffusion_mask_token_id = token_id
            return
        diffusion_cfg.mask_token_id = token_id
    except Exception:
        pass


def resolve_mask_token_id(
    tokenizer: Any,
    cfg: DictDefault,
    *,
    allow_add: bool,
    model: Any | None = None,
    default_token: str = "<|diffusion_mask|>",
) -> int:
    vocab_size = None
    if tokenizer is not None:
        if hasattr(tokenizer, "vocab_size") and tokenizer.vocab_size is not None:
            try:
                vocab_size = int(tokenizer.vocab_size)
            except Exception:
                vocab_size = None
        elif hasattr(tokenizer, "__len__"):
            try:
                vocab_size = int(len(tokenizer))
            except Exception:
                vocab_size = None

    diffusion_cfg = get_diffusion_config(cfg)
    cfg_id = (
        getattr(diffusion_cfg, "mask_token_id", None)
        if diffusion_cfg is not None
        else getattr(cfg, "diffusion_mask_token_id", None)
    )
    if isinstance(cfg_id, int) and cfg_id >= 0:
        if vocab_size is None or cfg_id < vocab_size:
            return int(cfg_id)

    def existing_special_token_id(token_str: str | None) -> int | None:
        if not token_str or not hasattr(tokenizer, "convert_tokens_to_ids"):
            return None
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
        except Exception:
            return None
        if not isinstance(token_id, int) or token_id < 0:
            return None
        unk_id = getattr(tokenizer, "unk_token_id", None)
        specials = set(getattr(tokenizer, "all_special_tokens", []) or [])
        additional = set(getattr(tokenizer, "additional_special_tokens", []) or [])
        if (
            (unk_id is not None and token_id == unk_id)
            or token_str not in specials | additional
            or (vocab_size is not None and token_id >= vocab_size)
        ):
            return None
        return token_id

    token_str = (
        getattr(diffusion_cfg, "mask_token_str", None)
        if diffusion_cfg is not None
        else getattr(cfg, "diffusion_mask_token_str", None)
    )
    for candidate in (token_str, default_token):
        token_id = existing_special_token_id(candidate)
        if token_id is not None:
            _set_mask_token_id(cfg, diffusion_cfg, int(token_id))
            return int(token_id)

    if allow_add and hasattr(tokenizer, "add_special_tokens"):
        token_to_add = token_str or default_token
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": [token_to_add]})
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
                _set_mask_token_id(cfg, diffusion_cfg, int(new_id))
                return int(new_id)
        except Exception:
            pass
    return int(getattr(tokenizer, "unk_token_id", 0) or 0)
