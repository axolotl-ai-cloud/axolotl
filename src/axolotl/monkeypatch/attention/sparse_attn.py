"""Native/Flash Sparse Attention (NSA/FSA) for MLA models.

Unlike sage/xformers (registered as drop-in attention functions), NSA/FSA is a
full attention block with its own projections, compression branch and learned
gate — it cannot be expressed through transformers' ``(module, q, k, v, ...)``
interface. We therefore swap the model's full-attention module for the
upstream :class:`fsa.module.fsa.FlashSparseAttention` module.

Both ``nsa`` and ``fsa`` route through the Flash-Sparse-Attention package
(https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention), an efficient
implementation of the NSA algorithm (https://arxiv.org/abs/2502.11089).

NOTE: the swapped module is randomly initialised — the pretrained MLA attention
weights are discarded. This is a "train sparse attention" path, not a
weight-preserving one.
"""

import weakref

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Full-attention module class names per supported model. Only these layers are
# swapped; linear-attention layers (e.g. Kimi's KimiDeltaAttention) are left
# untouched.
_FULL_ATTN_CLASSES = {
    "kimi_linear": ("KimiMLAAttention",),
    "deepseek_v2": ("DeepseekV2Attention",),
    "deepseek_v3": ("DeepseekV3Attention",),
}


def _is_fsa_available() -> bool:
    try:
        import fsa  # noqa: F401 # pylint: disable=unused-import

        return True
    except ImportError:
        return False


def _check_fsa_imported():
    if not _is_fsa_available():
        raise ImportError(
            "Flash-Sparse-Attention is not installed. Install the extra:\n"
            "  pip install 'axolotl[fsa]'\n"
            "(this also builds flash-attn, which requires a CUDA toolkit and an "
            "Ampere+ GPU)."
        )


def sparse_attention_stub(*args, **kwargs):  # pylint: disable=unused-argument
    """Registered so transformers accepts ``attn_implementation=nsa/fsa`` at
    load time. The real computation is the module swap; this must never run."""
    raise NotImplementedError(
        "nsa/fsa run as a model-specific module swap; the attention-function "
        "stub was invoked, which means the sparse-attention patch did not run "
        "for this model."
    )


def _enforce_min_segment(cu, min_len):
    """Drop boundaries that would create a segment shorter than ``min_len``.

    FSA's compression branch needs every sequence to be at least ``kernel_size``
    long; a packed remainder or padding tail can be much shorter, which makes the
    compressed-attention softmax degenerate and produces NaN gradients. Merging a
    short segment into its predecessor is safe under causal attention (a trailing
    pad/short tail only attends backward, and pad positions are loss-masked).
    """
    if min_len <= 1 or cu.numel() <= 2:
        return cu
    # skip the host scan entirely. greedy merge below is sequential
    # (each kept boundary depends on the previous), so it cannot be expressed as a
    # pure mask; we only pay the `tolist()` host sync when a short segment exists.
    seg = cu[1:] - cu[:-1]
    if bool((seg >= min_len).all()):
        return cu
    bounds = cu.tolist()
    total = bounds[-1]
    kept = [bounds[0]]
    for b in bounds[1:-1]:
        if b - kept[-1] >= min_len:
            kept.append(b)
    if total - kept[-1] < min_len and len(kept) > 1:
        kept.pop()  # merge the short tail into the previous segment
    kept.append(total)
    return cu.new_tensor(kept)


def _build_cu_seqlens(bsz, q_len, position_ids, kwargs, device, min_seg=1):
    """Cumulative sequence lengths over the flattened ``(bsz * q_len)`` stream.

    Prefers varlen boundaries already computed for packing; otherwise derives
    them from ``position_ids`` resets (each ``0`` marks a sequence start), and
    finally falls back to one sequence per batch row. Segments shorter than
    ``min_seg`` (the FSA compression kernel size) are merged into their neighbour.
    """
    total = bsz * q_len
    cu = kwargs.get("cu_seq_lens_q")
    if cu is None:
        cu = kwargs.get("cu_seqlens_q")
    if cu is not None:
        # `x or y` is unsafe here: bool() on a multi-element tensor raises. Some
        # transformers paths also hand back a (cu_q, cu_k) pair; q suffices.
        if isinstance(cu, (tuple, list)):
            cu = cu[0]
        cu = cu.to(device=device, dtype=torch.int32)
    elif position_ids is not None:
        # The FSA input is hidden_states flattened as bsz contiguous rows of
        # q_len, so the stream length is bsz*q_len. position_ids is shaped
        # (bsz, q_len) or broadcast as (1, q_len); each row is its own sequence
        # and a ``0`` marks an inner (packed) sub-sequence start. Building from
        # position_ids.numel() alone undercounts the stream when bsz > 1.
        pos = position_ids.view(1, -1) if position_ids.dim() == 1 else position_ids
        if pos.shape[0] == 1 and bsz > 1:
            pos = pos.expand(bsz, -1)
        resets = pos == 0
        resets[:, 0] = True  # every row begins a new sequence
        rows, cols = resets.nonzero(as_tuple=True)
        starts = (rows * q_len + cols).to(torch.int64).sort().values
        total_t = torch.tensor([total], device=starts.device, dtype=starts.dtype)
        cu = torch.cat([starts, total_t]).to(device=device, dtype=torch.int32)
    else:
        cu = torch.arange(0, total + 1, q_len, device=device, dtype=torch.int32)
    return _enforce_min_segment(cu, min_seg)


class _CuSeqlensCache:
    """One ``cu_seqlens`` build per forward step, shared by every swapped layer.

    transformers computes ``position_ids`` / ``cu_seq_lens`` once and threads the
    same tensor through all decoder layers, so without sharing each adapter would
    rebuild (and host-sync) identical boundaries N times per step. We key on the
    identity of that shared tensor; a ``weakref`` guards against ``id()`` reuse
    once it is freed, so a stale hit can never be returned across steps.
    """

    def __init__(self):
        self._key_id = None
        self._key_ref = None
        self._value = None

    def get(self, key_obj, build):
        if key_obj is None:
            return build()
        key_id = id(key_obj)
        ref = self._key_ref
        if key_id == self._key_id and ref is not None and ref() is key_obj:
            return self._value
        value = build()
        self._key_id = key_id
        self._key_ref = weakref.ref(key_obj)
        self._value = value
        return value


class SparseAttentionAdapter(nn.Module):
    """Adapt :class:`FlashSparseAttention` to a transformers attention module.

    FSA consumes flattened hidden states + ``cu_seqlens`` and returns the
    projected output (``hidden_size``), so we flatten in and reshape out.
    """

    def __init__(self, fsa: nn.Module, returns_tuple: bool, cu_cache: _CuSeqlensCache):
        super().__init__()
        self.fsa = fsa
        self.returns_tuple = returns_tuple
        self.cu_cache = cu_cache

    def forward(self, hidden_states, *args, position_ids=None, **kwargs):  # noqa: D102
        bsz, q_len, _ = hidden_states.shape
        # The shared tensor that determines the boundaries is identical across all
        # layers within a forward; key the per-step cache on it.
        key_obj = kwargs.get("cu_seq_lens_q")
        if key_obj is None:
            key_obj = kwargs.get("cu_seqlens_q")
        if key_obj is None:
            key_obj = position_ids
        cu_seqlens = self.cu_cache.get(
            key_obj,
            lambda: _build_cu_seqlens(
                bsz,
                q_len,
                position_ids,
                kwargs,
                hidden_states.device,
                min_seg=getattr(self.fsa, "kernel_size", 1),
            ),
        )
        out = self.fsa(hidden_states.reshape(-1, hidden_states.shape[-1]), cu_seqlens)
        out = out.view(bsz, q_len, -1)
        return (out, None) if self.returns_tuple else out


def _build_fsa_module(model_config, cfg):
    from fsa.module.fsa import FlashSparseAttention, RopeConfig

    mc = getattr(model_config, "text_config", model_config)
    hidden_size = mc.hidden_size
    num_q_heads = mc.num_attention_heads
    num_kv_heads = getattr(mc, "num_key_value_heads", None) or num_q_heads
    # FSA needs a single square head_dim (<=256) across q/k/v. MLA's value dim
    # is the natural choice and is well within the limit.
    head_dim = (
        getattr(mc, "v_head_dim", None)
        or getattr(mc, "head_dim", None)
        or hidden_size // num_q_heads
    )
    max_pos = getattr(mc, "max_position_embeddings", 131072)

    return FlashSparseAttention(
        hidden_size=hidden_size,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        kernel_size=cfg.nsa_kernel_size,
        kernel_stride=cfg.nsa_kernel_stride,
        block_size=cfg.nsa_block_size,
        topk=cfg.nsa_topk,
        init_blocks=cfg.nsa_init_blocks,
        local_blocks=cfg.nsa_local_blocks,
        window_size=cfg.nsa_window_size,
        rope_config=RopeConfig(max_position_embeddings=max_pos, head_dim=head_dim),
    )


def patch_sparse_attention(model, cfg, model_config):
    """Swap full-attention modules of an MLA model for FSA/NSA."""
    _check_fsa_imported()

    full_attn_classes = _FULL_ATTN_CLASSES[cfg.model_config_type]
    # Kimi's decoder layer uses the attention return directly; transformers'
    # DeepSeek layers unpack ``(attn_output, attn_weights)``.
    returns_tuple = cfg.model_config_type != "kimi_linear"

    # Shared so cu_seqlens is built once per forward and reused by every layer.
    cu_cache = _CuSeqlensCache()
    swapped = 0
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is None or type(attn).__name__ not in full_attn_classes:
            continue
        ref = next(attn.parameters(), None)
        adapter = SparseAttentionAdapter(
            _build_fsa_module(model_config, cfg), returns_tuple, cu_cache
        )
        if ref is not None:
            float_ref = next(
                (p for p in attn.parameters() if p.is_floating_point()), None
            )
            dtype = float_ref.dtype if float_ref is not None else cfg.torch_dtype
            adapter = adapter.to(device=ref.device, dtype=dtype)
        module.self_attn = adapter
        swapped += 1

    LOG.info(
        "Swapped %d full-attention layer(s) to %s (sparse attention)",
        swapped,
        cfg.attn_implementation,
    )
    if swapped == 0:
        # The stub registered for transformers' loader raises if ever called, so a
        # no-op swap would only surface as a confusing error at the first forward.
        raise RuntimeError(
            f"attn_implementation={cfg.attn_implementation!r} but no "
            f"{full_attn_classes} layers were found to swap. The sparse-attention "
            "patch did not apply; refusing to train with the unusable stub."
        )
