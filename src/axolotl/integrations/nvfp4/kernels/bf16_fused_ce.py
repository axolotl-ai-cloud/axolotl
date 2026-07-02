"""Chunked bf16 lm_head + cross-entropy that never materializes the [M, V] logits.

CCE-style vocab tiling, but the tile GEMM is plain bf16 (bit-for-bit the
materialized ``hidden @ W.t()``, no quantization). A memory/backward-traffic win,
not an FP4 throughput win. Frozen bias-free lm_head only (no weight grad).

Numerical safety (the prior CCE/Liger path went non-finite under NVFP4 SR grads +
max_grad_norm AMP unscale): exact CE gradient (no vocab skipping); logsumexp/softmax
max-shifted in fp32; ``grad_hidden`` accumulated in fp32, cast to bf16 once.
"""

from __future__ import annotations

import functools
import os

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Vocab tile width; env var wins over the patch fn.
_VOCAB_BLOCK = int(os.environ.get("AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK", "4096"))


def _set_vocab_block(vocab_block: int | None) -> None:
    """Effective vocab tile width: env var (if set) > ``vocab_block`` arg > 4096."""
    global _VOCAB_BLOCK
    env = os.environ.get("AXOLOTL_NVFP4_FUSED_CE_VOCAB_BLOCK")
    if env is not None:
        _VOCAB_BLOCK = int(env)
    elif vocab_block is not None:
        _VOCAB_BLOCK = int(vocab_block)


_PATCHED_FORWARDS: set[type] = set()


class _BF16FusedCrossEntropy(torch.autograd.Function):
    """Tiled bf16 lm_head -> fp32 logsumexp/gather -> CE, no ``[M, V]`` logits.

    backward recomputes the softmax tile-by-tile from the saved logsumexp, in fp32.
    ``grad_scale`` (1/num_items for grad-accum, else 1/valid_count) is pre-folded so
    backward is a pure function of the saved tensors. lm_head is frozen, no wgrad.
    """

    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, logit_scale, grad_scale):
        M = hidden.shape[0]
        V = weight.shape[0]
        device = hidden.device

        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))

        running_max = torch.full(
            (M,), float("-inf"), device=device, dtype=torch.float32
        )
        running_sum = torch.zeros(M, device=device, dtype=torch.float32)
        label_logit = torch.zeros(M, device=device, dtype=torch.float32)

        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            logits = (hidden @ weight[lo:hi].t()).float()
            if logit_scale != 1.0:
                logits = logits * logit_scale

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
        loss = ((lse - label_logit) * valid.float()).sum() * grad_scale

        ctx.save_for_backward(hidden, weight, lse, safe_labels, valid)
        ctx.logit_scale = logit_scale
        ctx.grad_scale = grad_scale
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden, weight, lse, safe_labels, valid = ctx.saved_tensors
        V = weight.shape[0]
        M, H = hidden.shape
        rows = torch.arange(M, device=hidden.device)

        coef = (
            grad_loss.float() * ctx.grad_scale * valid.float() * ctx.logit_scale
        ).unsqueeze(1)

        grad_hidden = torch.zeros(M, H, device=hidden.device, dtype=torch.float32)
        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            logits = (hidden @ weight[lo:hi].t()).float()
            if ctx.logit_scale != 1.0:
                logits = logits * ctx.logit_scale
            sm = torch.exp(logits - lse.unsqueeze(1))

            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            sm[rows, cols] -= in_tile.float()

            grad_hidden += (sm * coef) @ weight[lo:hi].float()

        return grad_hidden.to(hidden.dtype), None, None, None, None, None


def bf16_lm_head_cross_entropy(
    hidden: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch=None,
    shift: bool = True,
    logit_scale: float = 1.0,
) -> torch.Tensor | None:
    """Chunked bf16 lm_head + CE, or None if the head isn't a plain frozen Linear.

    Mirrors ``ForCausalLMLoss`` (shift, flatten, sum/num_items or mean). None for a
    non-plain / trainable / biased lm_head so the caller falls back.
    """
    if type(lm_head) is not nn.Linear:
        return None
    if lm_head.bias is not None or lm_head.weight.requires_grad:
        return None
    if hidden.device.type != "cuda":
        return None

    if shift:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    hidden2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    labels1d = labels.reshape(-1).to(hidden.device)

    valid = labels1d != ignore_index
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            grad_scale = num_items_in_batch.to(
                device=hidden.device, dtype=torch.float32
            ).reciprocal()
        else:
            grad_scale = 1.0 / float(num_items_in_batch)
    else:
        grad_scale = 1.0 / valid.sum().clamp(min=1).float()

    return _BF16FusedCrossEntropy.apply(
        hidden2d, lm_head.weight, labels1d, ignore_index, logit_scale, grad_scale
    )


def _make_fused_forward(orig_forward):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    # Preserve the signature: Trainer._remove_unused_columns inspects it; a bare
    # *args/**kwargs wrapper would hide input_ids/labels and drop every column.
    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if (
            labels is None
            or not getattr(self, "_axolotl_bf16_lm_head_ce_enabled", False)
            or not self.training
            or kwargs.get("logits_to_keep")
            or kwargs.get("return_dict") is False
        ):
            return orig_forward(self, *args, **kwargs)

        lm_head = self.get_output_embeddings()
        labels = kwargs.pop("labels")
        num_items_in_batch = kwargs.pop("num_items_in_batch", None)
        base = getattr(self, "model", None)
        if base is None:
            kwargs["labels"] = labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return orig_forward(self, *args, **kwargs)

        outputs = base(*args, **kwargs)
        loss = bf16_lm_head_cross_entropy(
            outputs.last_hidden_state,
            lm_head,
            labels,
            num_items_in_batch=num_items_in_batch,
            shift=True,
        )
        if loss is None:  # head became non-plain mid-run
            kwargs["labels"] = labels
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            return orig_forward(self, *args, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    return forward


def patch_model_bf16_lm_head_cross_entropy(
    model: nn.Module, vocab_block: int | None = None
) -> bool:
    """Patch ``model``'s ForCausalLM forward to use the chunked bf16 CE.

    Returns True if a patch was installed (frozen bias-free nn.Linear lm_head).
    Idempotent per ForCausalLM class; patching the base class covers PEFT too.
    """
    _set_vocab_block(vocab_block)
    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:
            causal = model

    try:
        lm_head = causal.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        LOG.warning("bf16_lm_head_cross_entropy: model has no output embeddings")
        return False

    if type(lm_head) is not nn.Linear:
        LOG.warning(
            "bf16_lm_head_cross_entropy: output embedding is %s, not a plain "
            "nn.Linear (NVFP4-quantized or LoRA-wrapped lm_head is not supported "
            "here; keeping the materialized CE path).",
            type(lm_head).__name__,
        )
        return False
    if lm_head.bias is not None or lm_head.weight.requires_grad:
        LOG.warning(
            "bf16_lm_head_cross_entropy: requires a frozen bias-free lm_head; "
            "keeping the materialized CE path."
        )
        return False

    causal._axolotl_bf16_lm_head_ce_enabled = True
    cls = causal.__class__
    if cls in _PATCHED_FORWARDS:
        return True
    cls.forward = _make_fused_forward(cls.forward)
    _PATCHED_FORWARDS.add(cls)
    LOG.info(
        "bf16_lm_head_cross_entropy: patched %s.forward (logits not materialized)",
        cls.__name__,
    )
    return True
