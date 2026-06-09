"""Fused NVFP4 lm_head + cross-entropy: skip the [M, V] logit tensor.

When ``quantize_lm_head`` makes the output projection an NVFP4 base module, the
default path still materializes the full ``[batch*seq, vocab]`` bf16 logit tensor
(~1 GiB at vocab 152k, seq 8k) before cross-entropy. This module fuses the
projection and the loss the way Cut Cross-Entropy does — tiling over the vocab,
computing a logit block at a time, accumulating logsumexp in fp32, gathering the
label logit — but it consumes the NVFP4-packed lm_head weight directly:
each vocab tile is dequantized FP4->bf16 on read (a row-slice of the packed
qdata/scale, bit-identical to dequantizing the whole table) instead of reading a
dense bf16 ``[V, H]`` weight (which an FP4 lm_head does not have).

What this buys, honestly:
  * MEMORY: the ``[M, V]`` logits (and their fp32 upcast) are never materialized;
    only one ``[M, V_BLOCK]`` tile lives at a time. This is the real win and it
    grows with seq length and vocab.
  * COMPUTE: the default path dequantizes each tile before a bf16 GEMM. The
    experimental ``fp4_matmul=True`` / ``AXOLOTL_NVFP4_FUSED_CE_FP4_MM=1`` path
    uses FP4 ``torch._scaled_mm`` for the lm_head logits, but still emits one
    ``[M, V_BLOCK]`` tile and runs CE/logsumexp separately. It is not a native CE
    epilogue, and backward still uses the dequantized weight tile for dgrad.

lm_head is frozen under LoRA, so only ``dL/dhidden`` is returned (no weight grad).
"""

from __future__ import annotations

import os

import torch
from torch import nn

# Vocab tile width. The transient logit tile is [M, _VOCAB_BLOCK] in fp32; 4096
# keeps it small (16 MiB at M=4096) while giving the bf16 tile-matmul enough work
# to stay efficient. Tunable; not load-bearing for correctness.
_VOCAB_BLOCK = 4096


def _fp4_scaled_mm_enabled(fp4_matmul: bool | None) -> bool:
    if fp4_matmul is not None:
        return fp4_matmul
    return os.environ.get("AXOLOTL_NVFP4_FUSED_CE_FP4_MM") == "1"


def _nvfp4_lm_head_store(module: nn.Module):
    """Return a row-sliceable ``[V, H]`` NVFP4Tensor for an FP4 lm_head, or None.

    Each supported module stores the lm_head weight packed in FP4 along the
    hidden dim, so every vocab row is a self-contained slice of qdata/scale. The
    store layout differs per module:

    - storage / tied: ``w_q`` is already the ``[V, H]`` NVFP4Tensor.
    - compute: ``w_fprop`` is ``_quantize(W)`` ([V, H] blocked along H); the GEMM
      transposes it on read, but the stored tensor is already the [V, H] store.

    MSLK-fast modules keep swizzled e4m3 scales (not safe to row-slice) and a
    high-precision store can't be tiled either — both return None so the caller
    falls back to the materialized path.
    """
    # Import lazily: torchao (and these classes) are optional at file import.
    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FrozenBaseLinear,
        NVFP4TiedLMHead,
    )

    if isinstance(module, (NVFP4FrozenBaseLinear, NVFP4TiedLMHead)):
        store = module.w_q
    elif isinstance(module, NVFP4ComputeBaseLinear):
        store = module.w_fprop
    else:
        return None  # MSLK-fast (swizzled), hp (NVFP4Linear), or non-FP4

    # Row-slicing the packed buffers is only bit-exact when scales are row-major.
    try:
        _, ctx = store.__tensor_flatten__()
    except Exception:
        return None
    if ctx.get("is_swizzled_scales"):
        return None
    return store


def _nvfp4_lm_head_fp4_store(module: nn.Module):
    """Return a ``[V, H]`` FP4 store usable by tiled ``torch._scaled_mm``."""
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    from axolotl.utils.nvfp4_training import (
        NVFP4ComputeBaseLinear,
        NVFP4FastComputeBaseLinear,
        NVFP4FastFrozenBaseLinear,
        NVFP4FrozenBaseLinear,
        NVFP4TiedLMHead,
    )

    if isinstance(module, (NVFP4FrozenBaseLinear, NVFP4TiedLMHead)):
        return module.w_q
    if isinstance(module, NVFP4ComputeBaseLinear):
        return module.w_fprop
    if isinstance(module, NVFP4FastFrozenBaseLinear):
        return NVFP4Tensor(
            module.wq,
            module.wsc,
            16,
            torch.bfloat16,
            per_tensor_scale=module.w_inv.to(torch.float32),
            is_swizzled_scales=True,
        )
    if isinstance(module, NVFP4FastComputeBaseLinear):
        return NVFP4Tensor(
            module.wq_f,
            module.wsc_f,
            16,
            torch.bfloat16,
            per_tensor_scale=module.w_inv_f.to(torch.float32),
            is_swizzled_scales=True,
        )
    return None


def _dequant_vocab_tile(store, lo: int, hi: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize vocab rows ``[lo, hi)`` of the packed lm_head to ``[hi-lo, H]``.

    Row-slices the packed qdata/scale (bit-identical to slicing a full dequant —
    NVFP4 blocks lie along the hidden dim, so rows are independent) so only one
    tile is ever upcast to bf16, never the whole ``[V, H]`` table.
    """
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    _, ctx = store.__tensor_flatten__()
    sub = NVFP4Tensor.__tensor_unflatten__(
        {
            "qdata": store.qdata[lo:hi],
            "scale": store.scale[lo:hi],
            "per_tensor_scale": store.per_tensor_scale,
        },
        ctx,
        None,
        None,
    )
    return sub.dequantize(dtype)


def _slice_vocab_tile(store, lo: int, hi: int):
    try:
        return store[lo:hi]
    except Exception:
        return None


def _weight_scale_for_scaled_mm(tile) -> torch.Tensor:
    if tile.is_swizzled_scales:
        return tile.scale.view(torch.float8_e4m3fn)
    from torchao.prototype.mx_formats.utils import to_blocked

    rows, hidden = tile.shape
    return to_blocked(tile.scale.view(rows, hidden // 16)).view(torch.float8_e4m3fn)


def _fp4_logits_tile(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    tile,
    dtype: torch.dtype,
) -> torch.Tensor:
    logits = torch._scaled_mm(
        hidden_q,
        tile.qdata.view(torch.float4_e2m1fn_x2).t(),
        bias=None,
        out_dtype=dtype,
        scale_a=hidden_scale,
        scale_b=_weight_scale_for_scaled_mm(tile),
    )
    if tile.per_tensor_scale is not None:
        logits = logits * tile.per_tensor_scale.to(dtype)
    return logits


def _quantize_hidden_sl(
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    try:
        from axolotl.utils.nvfp4_training import _mslk_quantize_sl

        return _mslk_quantize_sl(hidden)
    except Exception:
        return None


class _FusedFP4CrossEntropy(torch.autograd.Function):
    """Tiled lm_head(FP4) -> logsumexp/gather -> CE, no ``[M, V]`` logits.

    forward accumulates, per vocab tile: the running fp32 logsumexp (numerically
    stable, max-shifted) and the gathered label logit. backward recomputes the
    softmax tile-by-tile from the saved logsumexp and accumulates
    ``dL/dhidden = (softmax - onehot) @ W`` — lm_head is frozen, so no wgrad.

    ``grad_scale`` is the per-token weight already folded into the reduction
    (1/num_items for grad-accum, else 1/valid_count), so backward stays a pure
    function of the saved tensors.
    """

    @staticmethod
    def forward(ctx, hidden, store, labels, ignore_index, scale, grad_scale):
        # hidden: [M, H] (2D, contiguous), labels: [M]
        M, H = hidden.shape
        V = store.shape[0]
        device = hidden.device
        dtype = hidden.dtype

        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))

        running_max = torch.full(
            (M,), float("-inf"), device=device, dtype=torch.float32
        )
        running_sum = torch.zeros(M, device=device, dtype=torch.float32)
        label_logit = torch.zeros(M, device=device, dtype=torch.float32)

        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            w_tile = _dequant_vocab_tile(store, lo, hi, dtype)  # [Vb, H]
            # bf16 tile-matmul (the heavy FLOPs), fp32 only for the logsumexp.
            logits = (hidden @ w_tile.t()).float() * scale  # [M, Vb]

            tile_max = logits.max(dim=1).values
            new_max = torch.maximum(running_max, tile_max)
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                logits - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            gathered = logits.gather(1, cols.unsqueeze(1)).squeeze(1)
            label_logit = torch.where(in_tile, gathered, label_logit)

        lse = running_max + torch.log(running_sum)
        per_token = (lse - label_logit) * valid.float()  # CE per token, masked

        loss = per_token.sum() * grad_scale

        ctx.save_for_backward(hidden, lse, safe_labels, valid)
        ctx.store = store
        ctx.scale = scale
        ctx.grad_scale = grad_scale
        ctx.V = V
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, lse, safe_labels, valid = ctx.saved_tensors
        store = ctx.store
        scale = ctx.scale
        V = ctx.V
        M, H = hidden.shape
        dtype = hidden.dtype

        # d(loss)/d(logit_v) = grad_loss * grad_scale * mask * (softmax_v - onehot_v) * scale
        coef = (grad_loss * ctx.grad_scale * valid.float() * scale).unsqueeze(
            1
        )  # [M,1]
        rows = torch.arange(M, device=hidden.device)

        grad_hidden = torch.zeros(M, H, device=hidden.device, dtype=dtype)
        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            w_tile = _dequant_vocab_tile(store, lo, hi, dtype)  # [Vb, H]
            logits = (hidden @ w_tile.t()).float() * scale  # [M, Vb]
            sm = torch.exp(logits - lse.unsqueeze(1))  # softmax tile [M, Vb]

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            sm[rows, cols] -= in_tile.float()  # subtract onehot in place

            # bf16 dz @ w_tile (the heavy backward FLOPs); accumulate in bf16.
            grad_hidden += ((sm * coef).to(dtype)) @ w_tile

        return grad_hidden, None, None, None, None, None


class _FusedFP4ScaledMMCrossEntropy(torch.autograd.Function):
    """Tiled CE whose lm_head logits are produced by FP4 ``torch._scaled_mm``."""

    @staticmethod
    def forward(ctx, hidden, store, labels, ignore_index, scale, grad_scale):
        M, _ = hidden.shape
        V = store.shape[0]
        device = hidden.device
        dtype = hidden.dtype

        hidden_q = _quantize_hidden_sl(hidden)
        if hidden_q is None:
            raise RuntimeError(
                "MSLK single-level NVFP4 activation quant is unavailable"
            )
        hidden_qdata, hidden_scale = hidden_q

        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))

        running_max = torch.full(
            (M,), float("-inf"), device=device, dtype=torch.float32
        )
        running_sum = torch.zeros(M, device=device, dtype=torch.float32)
        label_logit = torch.zeros(M, device=device, dtype=torch.float32)

        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            tile = _slice_vocab_tile(store, lo, hi)
            if tile is None:
                raise RuntimeError("FP4 lm_head store is not vocab-tile sliceable")
            logits = _fp4_logits_tile(hidden_qdata, hidden_scale, tile, dtype).float()
            logits = logits * scale

            tile_max = logits.max(dim=1).values
            new_max = torch.maximum(running_max, tile_max)
            running_sum = running_sum * torch.exp(running_max - new_max) + torch.exp(
                logits - new_max.unsqueeze(1)
            ).sum(dim=1)
            running_max = new_max

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            gathered = logits.gather(1, cols.unsqueeze(1)).squeeze(1)
            label_logit = torch.where(in_tile, gathered, label_logit)

        lse = running_max + torch.log(running_sum)
        per_token = (lse - label_logit) * valid.float()
        loss = per_token.sum() * grad_scale

        ctx.save_for_backward(hidden, lse, safe_labels, valid)
        ctx.store = store
        ctx.scale = scale
        ctx.grad_scale = grad_scale
        ctx.V = V
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, lse, safe_labels, valid = ctx.saved_tensors
        store = ctx.store
        scale = ctx.scale
        V = ctx.V
        M, H = hidden.shape
        dtype = hidden.dtype

        hidden_qdata, hidden_scale = _quantize_hidden_sl(hidden)
        coef = (grad_loss * ctx.grad_scale * valid.float() * scale).unsqueeze(1)
        rows = torch.arange(M, device=hidden.device)

        grad_hidden = torch.zeros(M, H, device=hidden.device, dtype=dtype)
        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            tile = _slice_vocab_tile(store, lo, hi)
            logits = _fp4_logits_tile(hidden_qdata, hidden_scale, tile, dtype).float()
            logits = logits * scale
            sm = torch.exp(logits - lse.unsqueeze(1))

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            sm[rows, cols] -= in_tile.float()

            w_tile = tile.dequantize(dtype)
            grad_hidden += ((sm * coef).to(dtype)) @ w_tile

        return grad_hidden, None, None, None, None, None


def fused_fp4_cross_entropy(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch=None,
    shift: bool = True,
    logit_scale: float = 1.0,
    fp4_matmul: bool | None = None,
) -> torch.Tensor | None:
    """Fused FP4-lm_head + cross-entropy, or None if the head isn't tile-able.

    Mirrors ``ForCausalLMLoss``: shifts labels by one (predict next token),
    flattens, and reduces by sum/num_items (grad-accum) or mean over the unmasked
    tokens. Returns None when the lm_head is not a row-sliceable NVFP4 store
    (MSLK-swizzled / hp / bf16) or carries a bias, so the caller falls back to the
    materialized path.
    """
    if getattr(lm_head, "bias", None) is not None:
        return None  # bias-folding not implemented; rare on lm_head

    if shift:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    hidden2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    labels1d = labels.reshape(-1).to(hidden.device)

    valid = labels1d != ignore_index
    if num_items_in_batch is not None:
        denom = num_items_in_batch
        grad_scale = 1.0 / denom if torch.is_tensor(denom) else 1.0 / float(denom)
    else:
        grad_scale = 1.0 / valid.sum().clamp(min=1).float()

    if _fp4_scaled_mm_enabled(fp4_matmul):
        fp4_store = _nvfp4_lm_head_fp4_store(lm_head)
        if fp4_store is not None:
            try:
                return _FusedFP4ScaledMMCrossEntropy.apply(
                    hidden2d, fp4_store, labels1d, ignore_index, logit_scale, grad_scale
                )
            except Exception:
                if fp4_matmul is True:
                    raise
                pass

    store = _nvfp4_lm_head_store(lm_head)
    if store is None:
        return None

    return _FusedFP4CrossEntropy.apply(
        hidden2d, store, labels1d, ignore_index, logit_scale, grad_scale
    )


# --- model forward wiring -----------------------------------------------------
#
# The default HF forward computes ``logits = lm_head(hidden)`` then CE, so the
# logits are already materialized by the time ``loss_function`` runs. Like CCE,
# we wrap the ForCausalLM forward to skip the lm_head GEMM and call the fused
# kernel directly when labels are present and the head is an FP4 store. The
# forward still returns logits=None in the fused branch (training only reads
# ``loss``); any path that needs logits (no labels / generation) falls through to
# the original forward unchanged.

import functools  # noqa: E402
import logging  # noqa: E402

LOG = logging.getLogger(__name__)

_PATCHED_FORWARDS: set = set()


def _make_fused_forward(orig_forward, fp4_matmul: bool | None):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    # Preserve the original forward's signature: the Trainer inspects it via
    # _remove_unused_columns to decide which dataset columns to keep, and a bare
    # *args/**kwargs wrapper would hide input_ids/labels and drop every column.
    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        lm_head = self.get_output_embeddings()
        has_fp4_ce = _nvfp4_lm_head_store(lm_head) is not None or (
            _fp4_scaled_mm_enabled(fp4_matmul)
            and _nvfp4_lm_head_fp4_store(lm_head) is not None
        )
        # Only intercept the training path with an FP4, tile-able head. Anything
        # else (generation, non-FP4 head, logits_to_keep slicing) -> original.
        if labels is None or kwargs.get("logits_to_keep") or not has_fp4_ce:
            return orig_forward(self, *args, **kwargs)

        # Run the base model to get hidden states (mirror the HF forward prologue).
        labels = kwargs.pop("labels")
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        base = getattr(self, "model", None)
        if base is None:
            kwargs["labels"] = labels
            return orig_forward(self, *args, **kwargs)
        outputs = base(*args, **kwargs)
        hidden = outputs.last_hidden_state

        loss = fused_fp4_cross_entropy(
            hidden,
            lm_head,
            labels,
            num_items_in_batch=num_items_in_batch,
            shift=True,
            fp4_matmul=fp4_matmul,
        )
        if loss is None:  # store became non-tileable mid-run -> safe fallback
            kwargs["labels"] = labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return orig_forward(self, *args, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward


def patch_model_fused_fp4_ce(
    model: nn.Module,
    fp4_matmul: bool | None = None,
) -> bool:
    """Patch ``model``'s ForCausalLM forward to use the fused FP4 cross-entropy.

    Returns True if a patch was installed (the lm_head is a tile-able FP4 store),
    False otherwise. Idempotent per ForCausalLM class. The PEFT wrapper delegates
    its forward to the base model, so patching the underlying ForCausalLM class is
    enough whether or not LoRA is in use.
    """
    # Find the actual ForCausalLM module (unwrap PEFT) that owns get_output_embeddings.
    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:
            causal = model
    lm_head = causal.get_output_embeddings()
    has_fp4_ce = _nvfp4_lm_head_store(lm_head) is not None or (
        _fp4_scaled_mm_enabled(fp4_matmul)
        and _nvfp4_lm_head_fp4_store(lm_head) is not None
    )
    if not has_fp4_ce:
        LOG.warning(
            "fused_fp4_cross_entropy: lm_head is not supported by the fused FP4 "
            "CE paths (row-sliceable store, or FP4 matmul store with "
            "AXOLOTL_NVFP4_FUSED_CE_FP4_MM=1); keeping the materialized CE path."
        )
        return False

    cls = causal.__class__
    if cls in _PATCHED_FORWARDS:
        return True
    cls.forward = _make_fused_forward(cls.forward, fp4_matmul)
    _PATCHED_FORWARDS.add(cls)
    LOG.info(
        "fused_fp4_cross_entropy: patched %s.forward (logits not materialized%s)",
        cls.__name__,
        ", fp4_matmul" if _fp4_scaled_mm_enabled(fp4_matmul) else "",
    )
    return True
