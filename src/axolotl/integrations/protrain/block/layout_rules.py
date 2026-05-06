"""Placement rules for the interleaved block manager (§3.1.2).

Given ``n_swap``, ``n_checkpoint``, and ``N_block``, decide which block
index gets which ``BlockMode`` under ProTrain's three placement rules:

1. **Swap-early** — the first ``n_swap`` blocks get SWAP. Earlier blocks
   have more forward compute after them to hide the CPU->GPU prefetch.
2. **Interleave CKPT among the remaining blocks** — flattens peak memory
   by preventing activation accumulation in a contiguous run.
3. **Unopt-late** — blocks with NONE sit in the late tail so their
   activations are consumed first in backward, freeing PCIe bandwidth
   for the earlier swap-block prefetches.

Also ships ``discover_blocks`` — the heuristic that finds the
transformer-block ``nn.ModuleList`` inside a user model without needing
a central registry. Returns a ``list[BlockTree]`` so encoder-decoder
models (T5, FLAN-T5) can surface both encoder and decoder block trees;
single-tree causal-LM models return a single-element list.
"""

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
    """Return the per-block mode map under the four placement rules.

    Placement order, applied in sequence (later rules fill positions left
    free by earlier rules):

    1. **Swap-early** — the first ``n_swap`` block ids are SWAP. Earlier
       blocks have more forward compute after them to hide the CPU->GPU
       prefetch.
    2. **Interleave CKPT among the remaining blocks** — picks
       ``n_checkpoint`` positions from ``[n_swap, N_block)`` via the
       math-based even distribution, flattening peak memory by preventing
       activation accumulation in a contiguous run.
    3. **OFFLOAD in the unopt-late tail, before NONE** — the next
       ``n_offload`` free positions (in index order, after SWAP/CKPT
       removal) become OFFLOAD. OFFLOAD blocks share NONE's "unopt-late"
       placement intent — they free their PCIe budget on the forward side
       and their backward gather competes with reduce-offload in the same
       backward window CKPT recompute would have. Placed before NONE so
       that with mixed configs the tail-most positions stay NONE.
    4. **NONE fills the remainder** — any positions not assigned by
       rules 1-3 are NONE.

    Parameters
    ----------
    n_swap:
        Number of blocks that should use ``BlockMode.SWAP``. Must be
        non-negative and ``n_swap + n_checkpoint + n_offload <= N_block``.
    n_checkpoint:
        Number of blocks that should use ``BlockMode.CKPT``.
    N_block:
        Total number of transformer blocks in the model.
    n_offload:
        Number of blocks that should use ``BlockMode.OFFLOAD`` (the
        param-offload-aware NONE-equivalent for non-persistent chunks;
        see ``BLOCK_MODE_OFFLOAD_DESIGN.md`` §3.6). Defaults to 0 for
        backward compatibility — the legacy 3-knob (SWAP/CKPT/NONE)
        signature ``assign_modes(n_swap, n_checkpoint, N_block)``
        continues to produce identical maps.

    Returns
    -------
    BlockStrategyMap
        ``dict`` keyed ``0 .. N_block-1`` mapping to exactly ``n_swap``
        SWAP entries, ``n_checkpoint`` CKPT entries, ``n_offload``
        OFFLOAD entries, and ``N_block - n_swap - n_checkpoint -
        n_offload`` NONE entries.

    Raises
    ------
    ValueError
        If any input is negative or
        ``n_swap + n_checkpoint + n_offload > N_block``.
    """
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

    # Initialise everything to NONE (unopt-late default — positions that
    # do not receive SWAP/CKPT/OFFLOAD just stay NONE, and by construction
    # those positions land in the tail).
    modes: BlockStrategyMap = {BlockId(i): BlockMode.NONE for i in range(N_block)}

    # Rule 1: swap-early. First n_swap block ids are SWAP.
    for i in range(n_swap):
        modes[BlockId(i)] = BlockMode.SWAP

    # Rule 2: interleave CKPT evenly among the remaining (N_block - n_swap)
    # positions so checkpoint and non-checkpoint blocks alternate, flattening
    # peak memory. Strategy: pick n_checkpoint positions from [n_swap, N_block)
    # using a math-based even distribution.
    remaining = N_block - n_swap
    if n_checkpoint > 0 and remaining > 0:
        # Centered math-based placement: for k in 0..n_checkpoint-1, place at
        # position n_swap + ((2k + 1) * remaining) // (2 * n_checkpoint). The
        # half-step offset (2k + 1)/(2 * n_checkpoint) targets the midpoint
        # of each of n_checkpoint equal-width sub-intervals across the tail,
        # rather than its left edge — so CKPT slots are centered rather than
        # front-loaded. Concretely: with remaining=5, n_checkpoint=3 this
        # yields {0, 2, 4} (offset by n_swap) instead of the front-loaded
        # {0, 1, 3} the left-edge formula `(k * remaining) // n_checkpoint`
        # produced. Indices are unique whenever remaining >= n_checkpoint
        # (guaranteed by input validation: n_swap + n_checkpoint <= N_block),
        # and the maximum index is bounded by n_swap + ((2 * n_checkpoint - 1)
        # * remaining) // (2 * n_checkpoint) < N_block, preserving the
        # unopt-late tail for rule 3.
        ckpt_positions = {
            n_swap + ((2 * k + 1) * remaining) // (2 * n_checkpoint)
            for k in range(n_checkpoint)
        }
        for idx in sorted(ckpt_positions):
            modes[BlockId(idx)] = BlockMode.CKPT

    # Rule 3: OFFLOAD fills the next n_offload positions still bearing
    # NONE, in ascending index order. This places OFFLOAD "before NONE"
    # in the unopt-late tail — see §3.6 of ``BLOCK_MODE_OFFLOAD_DESIGN.md``.
    # When n_offload=0 (default / legacy callers), this loop is a no-op
    # and the map is identical to the pre-Option-B output.
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
    """One transformer-block sequence in a model's forward graph.

    Causal-LM models surface a single tree (e.g. ``"layers"`` on Llama,
    ``"h"`` on GPT-2). Encoder-decoder models surface two: an encoder
    (``forward_order=0``) and a decoder (``forward_order=1``). The
    decoder's forward consumes the encoder's last-layer hidden state via
    cross-attention; that cross-tree dependency is captured at the cost-
    model layer, not here — this dataclass only carries the topology.

    Attributes
    ----------
    name:
        Human-readable identifier for the tree (``""`` for single-tree
        models, ``"encoder"`` / ``"decoder"`` for T5).
    blocks:
        Ordered list of block ``nn.Module`` instances inside this tree.
        Order matches the underlying ``nn.ModuleList``, which is forward
        execution order by construction.
    forward_order:
        Position of this tree in the model's overall forward pass.
        Encoder=0, decoder=1; single-tree models always use 0.
    parent_path:
        Dotted module path on the root model that resolves to the
        underlying ``nn.ModuleList`` (e.g. ``"encoder.block"``,
        ``"model.layers"``). Used by the model wrapper to swap in
        wrapped blocks; ``""`` when the tree was found via the attention
        heuristic and no dotted path applies.
    """

    name: str
    blocks: list[nn.Module]
    forward_order: int
    parent_path: str = ""


def flatten_block_trees(trees: list[BlockTree]) -> list[nn.Module]:
    """Flatten ``BlockTree`` list into a single forward-ordered block list.

    Trees are sorted by ``forward_order`` ascending. Within each tree
    blocks are emitted in their existing list order (already forward
    order by construction). The returned position of each block IS its
    global ``BlockId`` — encoder blocks occupy ids ``[0, n_enc)``,
    decoder blocks occupy ids ``[n_enc, n_enc + n_dec)``. This global
    numbering is the source of truth used by hooks, the scheduler, and
    the trace's path -> block_id resolver, so every consumer agrees on
    which block a given id refers to.
    """
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
    """Heuristic: transformer blocks expose an ``attention`` or ``self_attn``
    attribute. Blocks wrapped by ProTrain's dispatcher expose
    ``_protrain_wrapped_mode``. Fall-back path when no known dotted path
    matches.

    Extends one level deeper for T5-style nested layouts: T5Block hides
    its attention + FFN inside a ``.layer`` ``nn.ModuleList`` whose
    elements are ``T5LayerSelfAttention`` / ``T5LayerCrossAttention`` /
    ``T5LayerFF``. We accept a module whose ``.layer`` ModuleList
    contains at least one element exposing ``EncDecAttention``,
    ``SelfAttention``, ``attention``, or ``self_attn`` as a direct
    attribute. This is only consulted on the fallback scan path —
    T5 models are normally caught by the ``encoder.block`` /
    ``decoder.block`` dotted paths.
    """
    if hasattr(m, "attention") or hasattr(m, "self_attn"):
        return True
    if hasattr(m, "_protrain_wrapped_mode"):
        return True
    # CheckpointedBlock stores the original in ``.block``; check one level in.
    inner = getattr(m, "block", None)
    if inner is not None and (
        hasattr(inner, "attention") or hasattr(inner, "self_attn")
    ):
        return True
    # T5Block-style nested layer ModuleList. T5LayerSelfAttention exposes
    # ``SelfAttention``; T5LayerCrossAttention exposes ``EncDecAttention``;
    # both are common attribute names on the inner ``.layer`` children.
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
    """Return the transformer-block trees on ``model``.

    Resolution order:

    1. Encoder-decoder dotted-path pairs. If both ``encoder.block`` AND
       ``decoder.block`` resolve to non-empty ``nn.ModuleList`` (T5,
       FLAN-T5), return two ``BlockTree`` entries. Other future enc-dec
       models (BART's ``encoder.layers`` / ``decoder.layers``) can be
       added to ``_ENC_DEC_PATH_PAIRS`` when needed.
    2. Single-tree dotted paths. Try each known causal-LM path
       (``transformer.h``, ``model.layers``, etc.). Return a single
       ``BlockTree`` for the first one that resolves.
    3. Fallback heuristic. Scan every ``nn.ModuleList`` under ``model``
       and return the first whose children all look like transformer
       blocks. T5Block-style nested-layer modules are recognised here
       too via ``_looks_like_block``'s ``.layer`` recursion.

    Returns
    -------
    list[BlockTree]
        Non-empty list. Single-tree models return one element with
        ``name=""`` and ``forward_order=0``. Encoder-decoder models
        return two elements: encoder first (``forward_order=0``), then
        decoder (``forward_order=1``).

    Raises
    ------
    RuntimeError
        If no match is found. The error message names the paths tried.
    """
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
            # Tree name is the first dotted segment ("encoder", "decoder").
            enc_name = enc_path.split(".")[0]
            dec_name = dec_path.split(".")[0]
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

    # 2. Single-tree dotted paths. Skip the enc-dec ones; those only
    # match in a pair.
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

    # 3. Fallback: scan for a ModuleList of block-shaped children.
    for path, mlist in _iter_module_lists_with_path(model):
        if len(mlist) == 0:
            continue
        # Reject ModuleLists nested inside a block-shaped ancestor that is
        # itself an indexed ModuleList entry (e.g. ``T5Block``'s inner
        # ``.layer`` ModuleList, where the ancestor at ``encoder.block.0``
        # is the block instance). Without this guard the T5Block's inner
        # list of T5LayerSelfAttention / T5LayerCrossAttention / T5LayerFF
        # — all of which can superficially satisfy ``_looks_like_block`` —
        # would be picked up as the block sequence. Restricting the reject
        # to ancestors whose final path segment is numeric leaves
        # non-indexed wrappers (e.g. ``bert.encoder`` is a ``BertEncoder``
        # that itself looks block-shaped but is the right intermediate)
        # untouched.
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
    """Map each block's dotted module path to its global ``BlockId``.

    Walked across ``flatten_block_trees(trees)`` so the returned ids
    match exactly the global numbering every other consumer sees. Used
    by the profiler to disambiguate encoder vs decoder block 0 (which
    would otherwise collide under naive
    ``_infer_block_id`` path-fragment parsing).

    Returns ``{}`` if any block can't be located inside the model
    (defensive — should not happen for well-formed BlockTree inputs).
    """
    flat = flatten_block_trees(trees)
    if not flat:
        return {}
    # Build an identity index over named_modules so we can locate each
    # block's path in O(N_modules) total instead of O(N_block * N_modules).
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
    "assign_modes",
    "discover_blocks",
    "BlockTree",
    "flatten_block_trees",
    "block_id_path_map",
]
