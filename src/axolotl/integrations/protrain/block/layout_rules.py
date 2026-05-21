"""Placement rules for the interleaved block manager: swap-early, interleave CKPT, OFFLOAD, NONE-late."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode, BlockStrategyMap
from axolotl.integrations.protrain.types import BlockId
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# assign_modes
# ---------------------------------------------------------------------------


def assign_modes(
    n_swap: int,
    n_checkpoint: int,
    N_block: int,
    *,
    n_offload: int = 0,
) -> BlockStrategyMap:
    """Return per-block mode map: SWAP early, CKPT interleaved, OFFLOAD before NONE in the tail."""
    if N_block < 0:
        raise ValueError(f"N_block must be non-negative, got {N_block}")
    if n_swap < 0 or n_checkpoint < 0 or n_offload < 0:
        raise ValueError(
            f"n_swap, n_checkpoint, n_offload must be non-negative, got "
            f"n_swap={n_swap}, n_checkpoint={n_checkpoint}, "
            f"n_offload={n_offload}"
        )
    if n_swap + n_checkpoint + n_offload > N_block:
        raise ValueError(
            f"n_swap + n_checkpoint + n_offload "
            f"({n_swap} + {n_checkpoint} + {n_offload} = "
            f"{n_swap + n_checkpoint + n_offload}) exceeds N_block "
            f"({N_block})"
        )

    # NONE default; SWAP/CKPT/OFFLOAD overwrite specific slots.
    modes: BlockStrategyMap = {BlockId(i): BlockMode.NONE for i in range(N_block)}

    # SWAP early.
    for i in range(n_swap):
        modes[BlockId(i)] = BlockMode.SWAP

    # Centered CKPT placement via half-step offset; unique indices guaranteed by input validation.
    remaining = N_block - n_swap
    if n_checkpoint > 0 and remaining > 0:
        ckpt_positions = {
            n_swap + ((2 * k + 1) * remaining) // (2 * n_checkpoint)
            for k in range(n_checkpoint)
        }
        for idx in sorted(ckpt_positions):
            modes[BlockId(idx)] = BlockMode.CKPT

    # OFFLOAD fills the next n_offload NONE positions in ascending order.
    if n_offload > 0:
        placed = 0
        for i in range(N_block):
            if placed >= n_offload:
                break
            if modes[BlockId(i)] is BlockMode.NONE:
                modes[BlockId(i)] = BlockMode.OFFLOAD
                placed += 1

    # Post-condition: counts match the request.
    _assert_counts(
        modes,
        n_swap=n_swap,
        n_checkpoint=n_checkpoint,
        n_offload=n_offload,
        N_block=N_block,
    )
    return modes


def _assert_counts(
    modes: BlockStrategyMap,
    *,
    n_swap: int,
    n_checkpoint: int,
    n_offload: int,
    N_block: int,
) -> None:
    """Invariant check. Raises ``ValueError`` if counts diverge."""
    counts = {
        BlockMode.NONE: 0,
        BlockMode.CKPT: 0,
        BlockMode.SWAP: 0,
        BlockMode.OFFLOAD: 0,
    }
    for m in modes.values():
        counts[m] = counts[m] + 1
    expected_none = N_block - n_swap - n_checkpoint - n_offload
    if (
        counts[BlockMode.SWAP] != n_swap
        or counts[BlockMode.CKPT] != n_checkpoint
        or counts[BlockMode.OFFLOAD] != n_offload
        or counts[BlockMode.NONE] != expected_none
    ):
        raise ValueError(
            f"assign_modes invariant violation: got counts={counts}, "
            f"expected SWAP={n_swap}, CKPT={n_checkpoint}, "
            f"OFFLOAD={n_offload}, NONE={expected_none}"
        )


# ---------------------------------------------------------------------------
# discover_blocks
# ---------------------------------------------------------------------------


# Dotted paths checked in order. Order rationale: GPT-2 style first (the
# project's canonical test target), then Llama/Mistral style (most common
# HF LLM layout), then less-common transformer variants, then the base_model
# layout used by PEFT-wrapped models. Encoder-decoder paths come last and are
# handled specially by ``discover_blocks`` (it walks the encoder/decoder pair
# together when both resolve, rather than returning the first match).
_KNOWN_BLOCK_PATHS: tuple[str, ...] = (
    "transformer.h",  # GPT-2, GPT-Neo, GPT-J (some), Falcon (some)
    "model.layers",  # Llama, Mistral, Qwen, most modern HF LLMs
    "transformer.layers",  # MPT, some GPT-NeoX variants
    "base_model.layers",  # PEFT / LoRA-wrapped models (short form)
    "base_model.model.model.layers",  # PEFT + LlamaForCausalLM (LoraModel wraps CausalLM)
    "base_model.model.transformer.h",  # PEFT + GPT-2
    "base_model.model.encoder.block",  # PEFT + T5 / FLAN-T5 encoder tree
    "base_model.model.decoder.block",  # PEFT + T5 / FLAN-T5 decoder tree
    "base_model.model.encoder.layers",  # PEFT + BART / mBART encoder tree
    "base_model.model.decoder.layers",  # PEFT + BART / mBART decoder tree
    "encoder.block",  # T5 / FLAN-T5 encoder tree
    "decoder.block",  # T5 / FLAN-T5 decoder tree
    "encoder.layers",  # BART / mBART encoder tree
    "decoder.layers",  # BART / mBART decoder tree
)


# Encoder-decoder dotted-path pairs. Each tuple is
# ``(encoder_path, decoder_path)``; both must resolve to non-empty
# ``nn.ModuleList`` for the model to be classified as encoder-decoder.
# When matched, ``discover_blocks`` returns two ``BlockTree`` entries —
# the encoder (forward_order=0) runs first; the decoder (forward_order=1)
# consumes the encoder's last-layer hidden state via cross-attention.
# PEFT/LoRA-wrapped enc-dec models route through ``base_model.model.*``
# (LoraModel wraps the original model under ``.base_model.model``); without
# the wrapped variants here, ``discover_blocks`` falls back to the heuristic
# and silently drops the decoder tree from block numbering.
_ENC_DEC_PATH_PAIRS: tuple[tuple[str, str], ...] = (
    ("encoder.block", "decoder.block"),  # T5 / FLAN-T5
    ("encoder.layers", "decoder.layers"),  # BART / mBART
    (
        "base_model.model.encoder.block",
        "base_model.model.decoder.block",
    ),  # PEFT + T5 / FLAN-T5
    (
        "base_model.model.encoder.layers",
        "base_model.model.decoder.layers",
    ),  # PEFT + BART / mBART
)


@dataclass(frozen=True)
class BlockTree:
    """One transformer-block sequence (encoder=0, decoder=1, or single-tree=0)."""

    name: str
    blocks: list[nn.Module]
    forward_order: int
    parent_path: str = ""


def flatten_block_trees(trees: list[BlockTree]) -> list[nn.Module]:
    """Flatten BlockTree list into forward-ordered global BlockIds (encoder first)."""
    out: list[nn.Module] = []
    for tree in sorted(trees, key=lambda t: t.forward_order):
        out.extend(tree.blocks)
    return out


def _resolve(root: nn.Module, dotted: str) -> nn.Module | None:
    obj: object = root
    for part in dotted.split("."):
        if not hasattr(obj, part):
            return None
        obj = getattr(obj, part)
    if isinstance(obj, nn.Module):
        return obj
    return None


def _looks_like_block(m: nn.Module) -> bool:
    """Heuristic for transformer block recognition; T5 nested-layer case via .layer ModuleList."""
    if hasattr(m, "attention") or hasattr(m, "self_attn"):
        return True
    if hasattr(m, "_protrain_wrapped_mode"):
        return True
    # CheckpointedBlock stores the original in .block.
    inner = getattr(m, "block", None)
    if inner is not None and (
        hasattr(inner, "attention") or hasattr(inner, "self_attn")
    ):
        return True
    # T5Block-style nested .layer ModuleList.
    nested = getattr(m, "layer", None)
    if isinstance(nested, nn.ModuleList) and len(nested) > 0:
        for child in nested:
            if (
                hasattr(child, "attention")
                or hasattr(child, "self_attn")
                or hasattr(child, "SelfAttention")
                or hasattr(child, "EncDecAttention")
            ):
                return True
    return False


def _iter_module_lists(root: nn.Module) -> Iterable[nn.ModuleList]:
    for m in root.modules():
        if isinstance(m, nn.ModuleList):
            yield m


def _iter_module_lists_with_path(
    root: nn.Module,
) -> Iterable[tuple[str, nn.ModuleList]]:
    for name, m in root.named_modules():
        if isinstance(m, nn.ModuleList):
            yield name, m


def discover_blocks(model: nn.Module) -> list[BlockTree]:
    """Return transformer-block trees: enc-dec pair → single-tree dotted → ModuleList heuristic."""
    # 1. Encoder-decoder pairs.
    for enc_path, dec_path in _ENC_DEC_PATH_PAIRS:
        enc = _resolve(model, enc_path)
        dec = _resolve(model, dec_path)
        if (
            isinstance(enc, nn.ModuleList)
            and isinstance(dec, nn.ModuleList)
            and len(enc) > 0
            and len(dec) > 0
        ):
            LOG.debug(
                "discover_blocks: enc-dec match %s+%s (n_enc=%d n_dec=%d)",
                enc_path,
                dec_path,
                len(enc),
                len(dec),
            )
            # Pick encoder/decoder name segment to handle PEFT-prefixed paths.
            enc_segments = enc_path.split(".")
            dec_segments = dec_path.split(".")
            enc_name = next(
                (s for s in enc_segments if s == "encoder"), enc_segments[0]
            )
            dec_name = next(
                (s for s in dec_segments if s == "decoder"), dec_segments[0]
            )
            return [
                BlockTree(
                    name=enc_name,
                    blocks=list(enc),
                    forward_order=0,
                    parent_path=enc_path,
                ),
                BlockTree(
                    name=dec_name,
                    blocks=list(dec),
                    forward_order=1,
                    parent_path=dec_path,
                ),
            ]

    # Single-tree dotted paths (skip enc-dec ones which only match in pairs).
    enc_dec_paths = {p for pair in _ENC_DEC_PATH_PAIRS for p in pair}
    for dotted in _KNOWN_BLOCK_PATHS:
        if dotted in enc_dec_paths:
            continue
        candidate = _resolve(model, dotted)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            LOG.debug("discover_blocks: matched %s (n=%d)", dotted, len(candidate))
            return [
                BlockTree(
                    name="",
                    blocks=list(candidate),
                    forward_order=0,
                    parent_path=dotted,
                ),
            ]

    # Fallback ModuleList scan; reject nested .layer lists under indexed-block ancestors.
    for path, mlist in _iter_module_lists_with_path(model):
        if len(mlist) == 0:
            continue
        skip = False
        ancestor_path = path
        while "." in ancestor_path:
            ancestor_path, _, _ = ancestor_path.rpartition(".")
            ancestor = _resolve(model, ancestor_path)
            ancestor_leaf = ancestor_path.rsplit(".", 1)[-1]
            if (
                isinstance(ancestor, nn.Module)
                and ancestor_leaf.isdigit()
                and _looks_like_block(ancestor)
            ):
                skip = True
                break
        if skip:
            continue
        if all(_looks_like_block(child) for child in mlist):
            LOG.debug(
                "discover_blocks: matched ModuleList via attention heuristic "
                "(n=%d, path=%r)",
                len(mlist),
                path,
            )
            return [
                BlockTree(
                    name="",
                    blocks=list(mlist),
                    forward_order=0,
                    parent_path=path,
                ),
            ]

    raise RuntimeError(
        "discover_blocks: no transformer-block ModuleList found on model. "
        f"Tried dotted paths {_KNOWN_BLOCK_PATHS} and the "
        "attention/self_attn attribute heuristic."
    )


def block_id_path_map(model: nn.Module, trees: list[BlockTree]) -> dict[str, BlockId]:
    """Map each block's dotted module path to its global BlockId; empty if any block missing."""
    flat = flatten_block_trees(trees)
    if not flat:
        return {}
    # id-indexed named_modules walk: O(N_modules) vs naive O(N_block * N_modules).
    path_by_id: dict[int, str] = {}
    for name, mod in model.named_modules():
        path_by_id[id(mod)] = name
    out: dict[str, BlockId] = {}
    for global_idx, block in enumerate(flat):
        path = path_by_id.get(id(block))
        if path is None or path == "":
            return {}
        out[path] = BlockId(global_idx)
    return out


__all__ = [
    "BlockTree",
    "assign_modes",
    "block_id_path_map",
    "discover_blocks",
    "flatten_block_trees",
]
