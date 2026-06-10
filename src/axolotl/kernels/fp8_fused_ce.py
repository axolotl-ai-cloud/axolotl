"""FP8 lm_head + cross-entropy without materializing full logits."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
from torch import nn

from axolotl.kernels.fp8_lm_head import (
    FP8Granularity,
    FP8LMHeadWeight,
    _packed_cache_key,
    _quantize_e4m3,
    _scale_from_amax,
    _scaled_mm,
    _to_col_major_for_scaled_mm,
    prepack_lm_head_weight_fp8,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_VOCAB_BLOCK = 4096
_PATCHED_FORWARDS: set[type] = set()


@dataclass(frozen=True)
class FP8FusedCEWeight:
    fprop: FP8LMHeadWeight
    dgrad_weight: torch.Tensor
    dgrad_scale: torch.Tensor


def prepack_lm_head_weight_fp8_ce(
    weight: torch.Tensor,
    *,
    granularity: FP8Granularity = "rowwise",
) -> FP8FusedCEWeight:
    if weight.ndim != 2:
        raise ValueError(f"lm_head weight must be 2D, got {tuple(weight.shape)}")
    vocab, hidden = weight.shape
    if vocab % 16 or hidden % 16:
        raise ValueError("FP8 fused CE requires vocab and hidden dims divisible by 16")

    fprop = prepack_lm_head_weight_fp8(weight, granularity=granularity)
    weight = weight.detach()
    if granularity == "tensorwise":
        dgrad_scale = _scale_from_amax(weight.abs().max())
    else:
        dgrad_scale = _scale_from_amax(weight.abs().amax(dim=0, keepdim=True))
    dgrad_weight = _to_col_major_for_scaled_mm(_quantize_e4m3(weight, dgrad_scale))
    return FP8FusedCEWeight(
        fprop=fprop,
        dgrad_weight=dgrad_weight,
        dgrad_scale=dgrad_scale,
    )


def _weight_cache_key(weight: torch.Tensor, granularity: FP8Granularity):
    return (_packed_cache_key(weight), granularity)


def _fp8_ce_packed_weight(lm_head: nn.Linear, granularity: FP8Granularity):
    key = _weight_cache_key(lm_head.weight, granularity)
    cached = getattr(lm_head, "_axolotl_fp8_fused_ce_packed", None)
    if cached is None or cached[0] != key:
        packed = prepack_lm_head_weight_fp8_ce(
            lm_head.weight,
            granularity=granularity,
        )
        lm_head._axolotl_fp8_fused_ce_packed = (key, packed)
    else:
        packed = cached[1]
    return packed


def _fprop_scale_tile(scale: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    return scale if scale.ndim == 0 else scale[:, lo:hi]


def _activation_scale(x: torch.Tensor, granularity: FP8Granularity) -> torch.Tensor:
    if granularity == "tensorwise":
        return _scale_from_amax(x.abs().max())
    return _scale_from_amax(x.abs().amax(dim=1, keepdim=True))


class _FP8FusedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, labels, packed, ignore_index, logit_scale, grad_scale):
        M = hidden.shape[0]
        V = packed.fprop.out_features
        valid = labels != ignore_index
        safe_labels = torch.where(valid, labels, labels.new_zeros(()))

        granularity = packed.fprop.granularity
        x_scale = _activation_scale(hidden, granularity)
        hidden_fp8 = _quantize_e4m3(hidden, x_scale)

        running_max = torch.full(
            (M,), float("-inf"), device=hidden.device, dtype=torch.float32
        )
        running_sum = torch.zeros(M, device=hidden.device, dtype=torch.float32)
        label_logit = torch.zeros(M, device=hidden.device, dtype=torch.float32)

        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            logits = _scaled_mm(
                hidden_fp8,
                packed.fprop.weight_t[:, lo:hi],
                scale_a=x_scale,
                scale_b=_fprop_scale_tile(packed.fprop.scale, lo, hi),
                out_dtype=hidden.dtype,
            ).float()
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

        ctx.save_for_backward(hidden_fp8, x_scale, lse, safe_labels, valid)
        ctx.packed = packed
        ctx.hidden_dtype = hidden.dtype
        ctx.logit_scale = logit_scale
        ctx.grad_scale = grad_scale
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        hidden_fp8, x_scale, lse, safe_labels, valid = ctx.saved_tensors
        packed = ctx.packed
        V = packed.fprop.out_features
        H = packed.fprop.in_features
        M = hidden_fp8.shape[0]
        rows = torch.arange(M, device=hidden_fp8.device)
        grad_hidden = torch.zeros(M, H, device=hidden_fp8.device, dtype=torch.float32)
        coef = (
            grad_loss.float() * ctx.grad_scale * valid.float() * ctx.logit_scale
        ).unsqueeze(1)

        for lo in range(0, V, _VOCAB_BLOCK):
            hi = min(lo + _VOCAB_BLOCK, V)
            logits = _scaled_mm(
                hidden_fp8,
                packed.fprop.weight_t[:, lo:hi],
                scale_a=x_scale,
                scale_b=_fprop_scale_tile(packed.fprop.scale, lo, hi),
                out_dtype=ctx.hidden_dtype,
            ).float()
            if ctx.logit_scale != 1.0:
                logits = logits * ctx.logit_scale

            dz = torch.exp(logits - lse.unsqueeze(1))
            in_tile = (safe_labels >= lo) & (safe_labels < hi)
            cols = (safe_labels - lo).clamp(0, hi - lo - 1)
            dz[rows, cols] -= in_tile.float()
            dz = dz * coef

            dz_scale = _activation_scale(dz, packed.fprop.granularity)
            dz_fp8 = _quantize_e4m3(dz, dz_scale)
            grad_hidden += _scaled_mm(
                dz_fp8,
                packed.dgrad_weight[lo:hi, :],
                scale_a=dz_scale,
                scale_b=packed.dgrad_scale,
                out_dtype=ctx.hidden_dtype,
            ).float()

        return grad_hidden.to(ctx.hidden_dtype), None, None, None, None, None


def fp8_lm_head_cross_entropy(
    hidden: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    num_items_in_batch=None,
    shift: bool = True,
    logit_scale: float = 1.0,
    granularity: FP8Granularity = "rowwise",
) -> torch.Tensor | None:
    if type(lm_head) is not nn.Linear:
        return None
    if lm_head.bias is not None or lm_head.weight.requires_grad:
        return None
    if hidden.device.type != "cuda" or lm_head.weight.device.type != "cuda":
        return None
    if not hasattr(torch, "_scaled_mm"):
        return None
    if lm_head.weight.shape[0] % 16 or lm_head.weight.shape[1] % 16:
        return None

    if shift:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    hidden2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    labels1d = labels.reshape(-1).to(hidden.device)
    valid = labels1d != ignore_index
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            denom = num_items_in_batch.to(
                device=hidden.device,
                dtype=torch.float32,
            )
            grad_scale = denom.reciprocal()
        else:
            grad_scale = 1.0 / float(num_items_in_batch)
    else:
        grad_scale = 1.0 / valid.sum().clamp(min=1).float()

    packed = _fp8_ce_packed_weight(lm_head, granularity)
    return _FP8FusedCrossEntropy.apply(
        hidden2d, labels1d, packed, ignore_index, logit_scale, grad_scale
    )


def _make_fused_forward(orig_forward):
    from transformers.modeling_outputs import CausalLMOutputWithPast

    @functools.wraps(orig_forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.get("labels")
        if (
            labels is None
            or not getattr(self, "_axolotl_fp8_lm_head_ce_enabled", False)
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
        loss = fp8_lm_head_cross_entropy(
            outputs.last_hidden_state,
            lm_head,
            labels,
            num_items_in_batch=num_items_in_batch,
            shift=True,
            granularity=getattr(self, "_axolotl_fp8_lm_head_ce_granularity", "rowwise"),
        )
        if loss is None:
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


def patch_model_fp8_lm_head_cross_entropy(
    model: nn.Module,
    *,
    granularity: FP8Granularity = "rowwise",
) -> bool:
    causal = model
    if hasattr(model, "get_base_model"):
        try:
            causal = model.get_base_model()
        except Exception:
            causal = model

    try:
        lm_head = causal.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        LOG.warning("fp8_lm_head_cross_entropy: model has no output embeddings")
        return False

    if type(lm_head) is not nn.Linear:
        LOG.warning(
            "fp8_lm_head_cross_entropy: output embedding is %s, not nn.Linear",
            type(lm_head).__name__,
        )
        return False
    if lm_head.bias is not None or lm_head.weight.requires_grad:
        LOG.warning("fp8_lm_head_cross_entropy: requires a frozen bias-free lm_head")
        return False
    if lm_head.weight.shape[0] % 16 or lm_head.weight.shape[1] % 16:
        LOG.warning("fp8_lm_head_cross_entropy: lm_head dims are not FP8-eligible")
        return False
    if not hasattr(torch, "_scaled_mm"):
        LOG.warning("fp8_lm_head_cross_entropy: torch._scaled_mm unavailable")
        return False

    causal._axolotl_fp8_lm_head_ce_granularity = granularity
    causal._axolotl_fp8_lm_head_ce_enabled = True
    cls = causal.__class__
    if cls in _PATCHED_FORWARDS:
        return True
    cls.forward = _make_fused_forward(cls.forward)
    _PATCHED_FORWARDS.add(cls)
    LOG.info(
        "fp8_lm_head_cross_entropy: patched %s.forward (logits not materialized)",
        cls.__name__,
    )
    return True
