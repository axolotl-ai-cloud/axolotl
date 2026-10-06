"""Document-local THD metadata and public Torch variable-length attention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn

from axolotl.monkeypatch.utils import get_max_seqlen_in_batch


@dataclass(frozen=True)
class VarlenMetadata:
    """One-time packing metadata for document-local variable-length attention."""

    flat_indices: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int
    sequence_length: int

    @property
    def total_tokens(self) -> int:
        return int(self.flat_indices.numel())

    def to(self, device: torch.device) -> "VarlenMetadata":
        return VarlenMetadata(
            flat_indices=self.flat_indices.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            batch_size=self.batch_size,
            sequence_length=self.sequence_length,
        )


def build_varlen_metadata(
    document_ids: torch.Tensor,
    semantic_validity: torch.Tensor,
    *,
    max_seqlen: int | None = None,
) -> VarlenMetadata:
    """Build document-local metadata without materializing a dense attention mask.

    ``max_seqlen`` may carry the exact maximum from CPU collator metadata. The
    sequence width is otherwise a safe public-kernel upper bound.
    """
    if document_ids.ndim != 2 or semantic_validity.ndim != 2:
        raise ValueError("document_ids and semantic_validity must be [batch, sequence]")
    if document_ids.shape != semantic_validity.shape:
        raise ValueError("document_ids and semantic_validity must have matching shapes")
    if document_ids.device != semantic_validity.device:
        raise ValueError("document_ids and semantic_validity must share a device")
    if document_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("document_ids must use an integer dtype")
    if semantic_validity.dtype is not torch.bool:
        raise TypeError("semantic_validity must be bool")

    valid = semantic_validity & (document_ids >= 0)
    torch._assert_async(
        ~(semantic_validity & (document_ids < 0)).any(),
        "semantic tokens require nonnegative document IDs",
    )

    batch_size, sequence_length = document_ids.shape
    starts = valid.clone()
    starts[:, 1:] &= ~(valid[:, :-1] & (document_ids[:, 1:] == document_ids[:, :-1]))
    normalized_runs = torch.where(
        valid,
        torch.cumsum(starts, dim=1, dtype=torch.int64),
        0,
    )
    lengths = get_max_seqlen_in_batch(normalized_runs)
    flat_indices = valid.reshape(-1).nonzero(as_tuple=False).flatten().to(torch.long)
    total_tokens = flat_indices.numel()
    if total_tokens == 0:
        raise ValueError(
            "varlen attention requires at least one semantic document token"
        )
    if total_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("varlen attention supports at most int32 total tokens")

    flat_document_ids = document_ids.to(torch.int64).reshape(-1)
    start_indices = starts.reshape(-1).nonzero(as_tuple=False).flatten()
    pairs = torch.stack(
        (
            start_indices.div(sequence_length, rounding_mode="floor"),
            flat_document_ids[start_indices],
        ),
        dim=1,
    )
    _, counts = torch.unique(pairs, dim=0, return_counts=True)
    torch._assert_async(
        (counts <= 1).all(),
        "a document ID must occupy one contiguous run within each row",
    )

    if max_seqlen is None:
        max_seqlen = sequence_length
    if not 0 < max_seqlen <= sequence_length:
        raise ValueError("max_seqlen must be between one and the sequence width")
    torch._assert_async(
        (lengths <= max_seqlen).all(),
        "max_seqlen must cover every document run",
    )
    return VarlenMetadata(
        flat_indices=flat_indices,
        cu_seqlens=F.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0)),
        max_seqlen=max_seqlen,
        batch_size=batch_size,
        sequence_length=sequence_length,
    )


def gather_thd(value: torch.Tensor, metadata: VarlenMetadata) -> torch.Tensor:
    """Gather a [B, H, S, D] tensor to public varlen [T, H, D] layout."""
    if value.ndim != 4:
        raise ValueError("attention inputs must be [batch, heads, sequence, head_dim]")
    batch_size, _, sequence_length, _ = value.shape
    if (batch_size, sequence_length) != (
        metadata.batch_size,
        metadata.sequence_length,
    ):
        raise ValueError("attention input shape does not match varlen metadata")
    if value.device != metadata.flat_indices.device:
        raise ValueError("attention input and varlen metadata must share a device")
    flattened = value.transpose(1, 2).reshape(
        batch_size * sequence_length, *value.shape[1::2]
    )
    return flattened.index_select(0, metadata.flat_indices)


def scatter_thd(value: torch.Tensor, metadata: VarlenMetadata) -> torch.Tensor:
    """Scatter public varlen [T, H, D] output to zero-padded [B, S, H, D]."""
    if value.ndim != 3:
        raise ValueError("varlen output must be [tokens, heads, head_dim]")
    if value.shape[0] != metadata.total_tokens:
        raise ValueError("varlen output token count does not match metadata")
    if value.device != metadata.flat_indices.device:
        raise ValueError("varlen output and metadata must share a device")
    flat = torch.zeros(
        (metadata.batch_size * metadata.sequence_length, *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    ).index_copy(0, metadata.flat_indices, value)
    return flat.reshape(metadata.batch_size, metadata.sequence_length, *value.shape[1:])


def varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    metadata: VarlenMetadata,
    *,
    scale: float | None = None,
    window_size: tuple[int, int] = (-1, -1),
    enable_gqa: bool | None = None,
) -> torch.Tensor:
    """Run document-local public Torch varlen attention and restore BSHD output."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "query, key, and value must be [batch, heads, sequence, head_dim]"
        )
    if key.shape != value.shape:
        raise ValueError("key and value must share shape")
    if query.shape[0] != key.shape[0] or query.shape[2:] != key.shape[2:]:
        raise ValueError(
            "query, key, and value must share batch, sequence, and head_dim"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must share a device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must share a dtype")

    if (
        metadata.flat_indices.device != query.device
        or metadata.cu_seqlens.device != query.device
    ):
        metadata = metadata.to(query.device)
    query_thd = gather_thd(query, metadata)
    key_thd = gather_thd(key, metadata)
    value_thd = gather_thd(value, metadata)
    if metadata.total_tokens == 0:
        return scatter_thd(query_thd, metadata)

    query_heads = query.shape[1]
    key_heads = key.shape[1]
    use_gqa = query_heads != key_heads if enable_gqa is None else enable_gqa
    if query_heads != key_heads and (not use_gqa or query_heads % key_heads):
        raise ValueError("GQA requires query heads divisible by key/value heads")
    output = varlen_attn(
        query_thd,
        key_thd,
        value_thd,
        metadata.cu_seqlens,
        metadata.cu_seqlens,
        metadata.max_seqlen,
        metadata.max_seqlen,
        scale=scale,
        window_size=window_size,
        enable_gqa=use_gqa,
    )
    if isinstance(output, tuple):
        output = output[0]
    return scatter_thd(output, metadata)
