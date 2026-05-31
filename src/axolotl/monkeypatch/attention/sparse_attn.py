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
            "Flash-Sparse-Attention is not installed. Install it from source: "
            "`pip install git+https://github.com/Relaxed-System-Lab/Flash-Sparse-Attention.git`"
        )


def sparse_attention_stub(*args, **kwargs):  # pylint: disable=unused-argument
    """Registered so transformers accepts ``attn_implementation=nsa/fsa`` at
    load time. The real computation is the module swap; this must never run."""
    raise NotImplementedError(
        "nsa/fsa run as a model-specific module swap; the attention-function "
        "stub was invoked, which means the sparse-attention patch did not run "
        "for this model."
    )


def _build_cu_seqlens(bsz, q_len, position_ids, kwargs, device):
    """Cumulative sequence lengths over the flattened ``(bsz * q_len)`` stream.

    Prefers varlen boundaries already computed for packing; otherwise derives
    them from ``position_ids`` resets (each ``0`` marks a sequence start), and
    finally falls back to one sequence per batch row.
    """
    cu = kwargs.get("cu_seq_lens_q") or kwargs.get("cu_seqlens_q")
    if cu is not None:
        return cu.to(device=device, dtype=torch.int32)
    if position_ids is not None:
        flat = position_ids.reshape(-1)
        starts = (flat == 0).nonzero(as_tuple=True)[0]
        total = torch.tensor([flat.numel()], device=flat.device)
        return torch.cat([starts, total]).to(device=device, dtype=torch.int32)
    return torch.arange(0, (bsz + 1) * q_len, q_len, device=device, dtype=torch.int32)


class SparseAttentionAdapter(nn.Module):
    """Adapt :class:`FlashSparseAttention` to a transformers attention module.

    FSA consumes flattened hidden states + ``cu_seqlens`` and returns the
    projected output (``hidden_size``), so we flatten in and reshape out.
    """

    def __init__(self, fsa: nn.Module, returns_tuple: bool):
        super().__init__()
        self.fsa = fsa
        self.returns_tuple = returns_tuple

    def forward(self, hidden_states, *args, position_ids=None, **kwargs):  # noqa: D102
        bsz, q_len, _ = hidden_states.shape
        cu_seqlens = _build_cu_seqlens(
            bsz, q_len, position_ids, kwargs, hidden_states.device
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

    swapped = 0
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is None or type(attn).__name__ not in full_attn_classes:
            continue
        ref = next(attn.parameters(), None)
        adapter = SparseAttentionAdapter(
            _build_fsa_module(model_config, cfg), returns_tuple
        )
        if ref is not None:
            adapter = adapter.to(device=ref.device, dtype=ref.dtype)
        module.self_attn = adapter
        swapped += 1

    LOG.info(
        "Swapped %d full-attention layer(s) to %s (sparse attention)",
        swapped,
        cfg.attn_implementation,
    )
    if swapped == 0:
        LOG.warning(
            "attn_implementation=%s but no %s layers were found to swap.",
            cfg.attn_implementation,
            full_attn_classes,
        )
