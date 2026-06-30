"""Fused NVFP4 lm_head + cross-entropy that never materializes the [M, V] logits.

Tiles over the vocab (CCE-style), dequantizing each NVFP4-packed weight tile
FP4->bf16 on read. lm_head is frozen, so only ``dL/dhidden`` is returned.
"""

from __future__ import annotations

import os

import torch
from torch import nn

# Vocab tile width: a loss-invariant speed<->VRAM dial (4096 balanced, 8192 max
# throughput). Env var always wins; resolved/written back in patch_model_fused_fp4_ce.
_VOCAB_BLOCK_DEFAULT = 4096
_VOCAB_BLOCK = int(
    os.environ.get("AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK", str(_VOCAB_BLOCK_DEFAULT))
)


def _resolve_vocab_block(vocab_block: int | None) -> int:
    """Effective vocab tile width: env var (if set) > ``vocab_block`` arg > 4096."""
    env = os.environ.get("AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK")
    if env is not None:
        return int(env)
    if vocab_block is not None:
        return int(vocab_block)
    return _VOCAB_BLOCK_DEFAULT


def _nvfp4_lm_head_store(module: nn.Module):
    """Return a row-sliceable ``[V, H]`` NVFP4Tensor for an FP4 lm_head, or None.

    None for MSLK-fast (swizzled scales, not row-sliceable) and hp stores, so the
    caller falls back to the materialized path.
    """
    # torchao (and these classes) are optional at file import.
    from axolotl.integrations.nvfp4.nvfp4_training import (
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

    # Row-slicing is only bit-exact with row-major (non-swizzled) scales.
    if getattr(store, "is_swizzled_scales", False):
        return None
    return store


def _dequant_vocab_tile(store, lo: int, hi: int, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize vocab rows ``[lo, hi)`` of the packed lm_head to ``[hi-lo, H]``.

    Row-slice via NVFP4Tensor's ``aten.slice`` dispatch (compile-traceable, unlike a
    manual flatten round-trip); bit-exact only for the non-swizzled store callers pass.
    """
    return store[lo:hi].dequantize(dtype)


class _FusedFP4CrossEntropy(torch.autograd.Function):
    """Tiled lm_head(FP4) -> fp32 logsumexp/gather -> CE, no ``[M, V]`` logits.

    backward recomputes the softmax tile-by-tile from the saved logsumexp.
    ``grad_scale`` (1/num_items for grad-accum, else 1/valid_count) is pre-folded so
    backward is a pure function of the saved tensors. lm_head is frozen, no wgrad.
    """

    @staticmethod
    def forward(ctx, hidden, store, labels, ignore_index, scale, grad_scale):
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
            w_tile = _dequant_vocab_tile(store, lo, hi, dtype)
            logits = (hidden @ w_tile.t()).float() * scale

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

        coef = (grad_loss * ctx.grad_scale * valid.float() * scale).unsqueeze(1)
        rows = torch.arange(M, device=hidden.device)

        # fp32 accumulator: bf16 cross-tile summation drifts ~3% over the vocab tiles.
        grad_hidden = torch.zeros(M, H, device=hidden.device, dtype=torch.float32)
        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            w_tile = _dequant_vocab_tile(store, lo, hi, dtype)
            logits = (hidden @ w_tile.t()).float() * scale
            sm = torch.exp(logits - lse.unsqueeze(1))

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            sm[rows, cols] -= in_tile.float()

            grad_hidden += (((sm * coef).to(dtype)) @ w_tile).float()

        return grad_hidden.to(dtype), None, None, None, None, None


def fused_fp4_cross_entropy(
    hidden: torch.Tensor,
    lm_head: nn.Module,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch=None,
    shift: bool = True,
    logit_scale: float = 1.0,
) -> torch.Tensor | None:
    """Fused FP4-lm_head + cross-entropy, or None if the head isn't tile-able.

    Mirrors ``ForCausalLMLoss`` (shift, flatten, sum/num_items or mean). None for a
    non-row-sliceable store or a biased head, so the caller falls back.
    """
    if getattr(lm_head, "bias", None) is not None:
        return None  # bias-folding not implemented

    if shift:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    hidden2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    labels1d = labels.reshape(-1).to(hidden.device)

    valid = labels1d != ignore_index
    if num_items_in_batch is not None:
        denom = num_items_in_batch
        if torch.is_tensor(denom):
            grad_scale = denom.to(
                device=hidden.device, dtype=torch.float32
            ).reciprocal()
        else:
            grad_scale = 1.0 / float(denom)
    else:
        grad_scale = 1.0 / valid.sum().clamp(min=1).float()

    store = _nvfp4_lm_head_store(lm_head)
    if store is None:
        return None

    return _FusedFP4CrossEntropy.apply(
        hidden2d, store, labels1d, ignore_index, logit_scale, grad_scale
    )


# --- model forward wiring -----------------------------------------------------
# Wrap the ForCausalLM forward to call the fused kernel (and return logits=None)
# when labels are present and the head is a row-sliceable FP4 store; otherwise fall
# through to the original forward (generation / no-labels paths need logits).

import functools  # noqa: E402
import logging  # noqa: E402

LOG = logging.getLogger(__name__)

_PATCHED_FORWARDS: set = set()


def _make_fused_forward(orig_forward):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    # Preserve the signature: Trainer._remove_unused_columns inspects it; a bare
    # *args/**kwargs wrapper would hide input_ids/labels and drop every column.
    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        lm_head = self.get_output_embeddings()
        has_fp4_ce = _nvfp4_lm_head_store(lm_head) is not None
        # Only intercept training with an FP4, tile-able head.
        if (
            labels is None
            or not self.training
            or kwargs.get("logits_to_keep")
            or kwargs.get("return_dict") is False
            or not has_fp4_ce
        ):
            return orig_forward(self, *args, **kwargs)

        labels = kwargs.pop("labels")
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        base = getattr(self, "model", None)
        if base is None:
            kwargs["labels"] = labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return orig_forward(self, *args, **kwargs)
        outputs = base(*args, **kwargs)
        hidden = outputs.last_hidden_state

        loss = fused_fp4_cross_entropy(
            hidden,
            lm_head,
            labels,
            num_items_in_batch=num_items_in_batch,
            shift=True,
        )
        if loss is None:  # store became non-tileable mid-run
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
    vocab_block: int | None = None,
) -> bool:
    """Patch ``model``'s ForCausalLM forward to use the fused FP4 cross-entropy.

    Returns True if a patch was installed (tile-able FP4 store). Idempotent per
    ForCausalLM class; patching the base class covers the PEFT-wrapped case too.
    ``vocab_block`` (env > arg > 4096) is written back to ``_VOCAB_BLOCK`` once at
    load time; safe since the block width is loss-invariant.
    """
    global _VOCAB_BLOCK
    effective_block = _resolve_vocab_block(vocab_block)
    _VOCAB_BLOCK = effective_block

    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:
            causal = model
    lm_head = causal.get_output_embeddings()
    if _nvfp4_lm_head_store(lm_head) is None:
        LOG.warning(
            "fused_fp4_cross_entropy: lm_head is not a row-sliceable NVFP4 store; "
            "keeping the materialized CE path."
        )
        return False

    cls = causal.__class__
    if cls in _PATCHED_FORWARDS:
        return True
    cls.forward = _make_fused_forward(cls.forward)
    _PATCHED_FORWARDS.add(cls)
    LOG.info(
        "fused_fp4_cross_entropy: patched %s.forward (logits not materialized, "
        "vocab_block=%d)",
        cls.__name__,
        effective_block,
    )
    return True
