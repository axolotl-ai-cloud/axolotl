"""Cut Cross-Entropy patch for the NCM tokenizer and generator.

Registered into ``cut_cross_entropy``'s ``PATCH_FNS`` by the NCM model-support hook,
so ``cut_cross_entropy: true`` routes both stages' losses through CCE without
materializing logits.
"""

from types import MethodType

import torch
from cut_cross_entropy import linear_cross_entropy as cce
from cut_cross_entropy.transformers.utils import PatchOptions
from transformers import PreTrainedModel

from .modeling import NCMGeneratorModel, NCMTokenizerModel

_PATCH_OPTS: PatchOptions | None = None


def linear_cross_entropy(
    e: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    targets: torch.Tensor,
    filter_c_grad: bool | None = None,
) -> torch.Tensor:
    """Per-position CE of ``e @ weight.T + bias``, shaped like ``targets``."""
    opts = _PATCH_OPTS
    if opts is None:
        raise RuntimeError("patch_ncm has not been applied")
    device = e.device.type
    dtype = (
        torch.get_autocast_dtype(device)
        if torch.is_autocast_enabled(device)
        else e.dtype
    )
    if dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("cut_cross_entropy needs bf16 or fp16 training (bf16: true)")
    # CCE takes 2D inputs; flattening also hands its backward a contiguous grad.
    loss = cce(
        e.reshape(-1, e.shape[-1]).to(dtype),
        weight.to(dtype),
        targets.reshape(-1),
        bias=bias,
        reduction="none",
        impl=opts.impl,
        filter_eps=opts.filter_eps,
        accum_e_fp32=opts.accum_e_fp32,
        accum_c_fp32=opts.accum_c_fp32,
        filter_e_grad=opts.filter_e_grad,
        filter_c_grad=opts.filter_c_grad if filter_c_grad is None else filter_c_grad,
        c_grad_chunk_size=opts.c_grad_chunk_size,
    )
    return loss.view(targets.shape)


def _recon_loss(self, z_hat: torch.Tensor, labels: torch.Tensor):
    if _PATCH_OPTS is None or not _PATCH_OPTS.use_lce(labels, self.training):
        return self._dense_recon_loss(z_hat, labels)
    ce = linear_cross_entropy(
        self.decoder.hidden(z_hat), self.decoder.output_layer.weight, None, labels
    )
    valid = labels != -100
    return (ce * valid).sum() / valid.sum().clamp(min=1), None


def _level_cross_entropy(self, h: torch.Tensor, targets: torch.Tensor):
    if _PATCH_OPTS is None or not _PATCH_OPTS.use_lce(targets, self.training):
        return self._dense_level_cross_entropy(h, targets)
    h_levels = h.split(self.pred_levels, dim=1)
    t_levels = targets.split(self.pred_levels, dim=1)
    ce = []
    for i, (hk, tg) in enumerate(zip(h_levels, t_levels, strict=True)):
        e, weight, bias = self._head_linear(i, hk)
        # Unfiltered: tied-head weights carry the learned temperature's gradient.
        ce.append(linear_cross_entropy(e, weight, bias, tg, filter_c_grad=False))
    return torch.cat(ce, dim=1), None


def patch_ncm(maybe_model, patch_options):
    global _PATCH_OPTS  # pylint: disable=global-statement
    _PATCH_OPTS = patch_options
    patches = (
        (NCMTokenizerModel, "_recon_loss", _recon_loss),
        (NCMGeneratorModel, "_level_cross_entropy", _level_cross_entropy),
    )
    if isinstance(maybe_model, PreTrainedModel):
        for cls, name, fn in patches:
            if isinstance(maybe_model, cls):
                setattr(maybe_model, name, MethodType(fn, maybe_model))
        return maybe_model
    for cls, name, fn in patches:
        setattr(cls, name, fn)
    return None
