"""Dense reference masks for packed encoder/canvas diffusion."""

from __future__ import annotations

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    create_block_mask = None


def encoder_causal_mask(
    document_ids: torch.Tensor,
    validity: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None = None,
) -> torch.Tensor:
    """Permit causal attention only within each logical encoder document."""
    length = document_ids.shape[1]
    same_document = document_ids[:, :, None] == document_ids[:, None, :]
    causal = (
        torch.arange(length, device=document_ids.device)[:, None]
        >= torch.arange(length, device=document_ids.device)[None, :]
    )
    mask = same_document & causal[None] & validity[:, :, None] & validity[:, None, :]
    if sliding_window is not None:
        distance = position_ids[:, :, None] - position_ids[:, None, :]
        mask &= (distance >= 0) & (distance < sliding_window)
    return mask.unsqueeze(1)


def decoder_prefix_canvas_mask(
    encoder_document_ids: torch.Tensor,
    canvas_document_ids: torch.Tensor,
    encoder_validity: torch.Tensor,
    canvas_validity: torch.Tensor,
    decoder_prefix_lengths: torch.Tensor,
    encoder_position_ids: torch.Tensor,
    sliding_window: int | None = None,
    logical_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restrict each canvas query to its clean prefix and own canvas segment."""
    if logical_ids is None:
        logical_ids = torch.arange(
            decoder_prefix_lengths.numel(), device=decoder_prefix_lengths.device
        )
    if logical_ids.shape != decoder_prefix_lengths.shape:
        raise ValueError("logical_ids must align with decoder_prefix_lengths")
    same_prompt = canvas_document_ids[:, :, None] == encoder_document_ids[:, None, :]
    prefix_by_document = torch.zeros_like(encoder_document_ids)
    for logical_id, prefix in zip(logical_ids, decoder_prefix_lengths, strict=True):
        prefix_by_document = torch.where(
            encoder_document_ids == logical_id, prefix, prefix_by_document
        )
    visible_prompt = encoder_position_ids[:, None, :] < prefix_by_document[:, None, :]
    same_canvas = canvas_document_ids[:, :, None] == canvas_document_ids[:, None, :]
    prompt_keys = same_prompt & visible_prompt & encoder_validity[:, None, :]
    canvas_keys = same_canvas & canvas_validity[:, None, :]
    if sliding_window is not None:
        prompt_keys = prompt_keys & (
            encoder_position_ids[:, None, :]
            >= prefix_by_document[:, None, :] - sliding_window + 1
        )
    mask = torch.cat((prompt_keys, canvas_keys), dim=-1) & canvas_validity[:, :, None]
    return mask.unsqueeze(1)


def _require_flex_attention() -> None:
    if create_block_mask is None:
        raise RuntimeError(
            "flex_attention requires a PyTorch build with BlockMask support"
        )


def encoder_flex_block_mask(
    document_ids: torch.Tensor,
    validity: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None = None,
):
    """Create a sparse native FlexAttention mask for packed causal documents."""
    _require_flex_attention()
    batch_size, length = document_ids.shape

    def mask_mod(batch, head, query, key):
        del head
        valid = validity[batch, query] & validity[batch, key]
        same_document = document_ids[batch, query] == document_ids[batch, key]
        distance = position_ids[batch, query] - position_ids[batch, key]
        visible = valid & same_document & (distance >= 0)
        if sliding_window is not None:
            visible = visible & (distance < sliding_window)
        return visible

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device=document_ids.device,
    )


def full_sequence_flex_block_mask(
    document_ids: torch.Tensor,
    semantic_validity: torch.Tensor,
    position_ids: torch.Tensor,
    sliding_window: int | None = None,
):
    """Create a sparse bidirectional mask for packed full-sequence rows."""
    _require_flex_attention()
    del position_ids
    batch_size, length = document_ids.shape

    def mask_mod(batch, head, query, key):
        del head
        return (
            semantic_validity[batch, query]
            & semantic_validity[batch, key]
            & (document_ids[batch, query] >= 0)
            & (document_ids[batch, query] == document_ids[batch, key])
        )

    if sliding_window is not None:
        raise ValueError("full-sequence diffusion does not support sliding attention")
    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device=document_ids.device,
    )


def decoder_flex_block_mask(
    encoder_document_ids: torch.Tensor,
    canvas_document_ids: torch.Tensor,
    encoder_validity: torch.Tensor,
    canvas_validity: torch.Tensor,
    decoder_prefix_lengths: torch.Tensor,
    encoder_position_ids: torch.Tensor,
    sliding_window: int | None = None,
    logical_ids: torch.Tensor | None = None,
):
    """Create a sparse decoder mask with document-local prefix and canvas access."""
    _require_flex_attention()
    if logical_ids is None:
        logical_ids = torch.arange(
            decoder_prefix_lengths.numel(), device=decoder_prefix_lengths.device
        )
    if logical_ids.shape != decoder_prefix_lengths.shape:
        raise ValueError("logical_ids must align with decoder_prefix_lengths")
    prefix_by_encoder = torch.zeros_like(encoder_document_ids)
    for logical_id, prefix in zip(logical_ids, decoder_prefix_lengths, strict=True):
        prefix_by_encoder = torch.where(
            encoder_document_ids == logical_id, prefix, prefix_by_encoder
        )
    batch_size, prompt_length = encoder_document_ids.shape
    canvas_length = canvas_document_ids.shape[1]

    def mask_mod(batch, head, query, key):
        del head
        query_document = canvas_document_ids[batch, query]
        query_valid = canvas_validity[batch, query]
        prompt_key = key < prompt_length
        safe_prompt_key = torch.where(prompt_key, key, 0)
        safe_canvas_key = torch.where(prompt_key, 0, key - prompt_length)
        key_document = torch.where(
            prompt_key,
            encoder_document_ids[batch, safe_prompt_key],
            canvas_document_ids[batch, safe_canvas_key],
        )
        key_valid = torch.where(
            prompt_key,
            encoder_validity[batch, safe_prompt_key],
            canvas_validity[batch, safe_canvas_key],
        )
        prefix = prefix_by_encoder[batch, safe_prompt_key]
        prompt_visible = encoder_position_ids[batch, safe_prompt_key] < prefix
        if sliding_window is not None:
            prompt_visible = prompt_visible & (
                encoder_position_ids[batch, safe_prompt_key]
                >= prefix - sliding_window + 1
            )
        return (
            query_valid
            & key_valid
            & (query_document == key_document)
            & torch.where(prompt_key, prompt_visible, torch.ones_like(prompt_visible))
        )

    return create_block_mask(
        mask_mod,
        B=batch_size,
        H=None,
        Q_LEN=canvas_length,
        KV_LEN=prompt_length + canvas_length,
        device=encoder_document_ids.device,
    )
