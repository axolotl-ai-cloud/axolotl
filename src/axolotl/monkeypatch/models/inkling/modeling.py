"""Sample-packing patch for Inkling.

Every Inkling layer runs four short causal convs (k/v inside attention, and after
attention and the MLP) that never receive the layer kwargs, so ``seq_idx`` cannot
reach them. The text model turns ``position_ids`` into ``PackedSegments`` and each
decoder layer swaps the module-level conv for one that resets at document
boundaries while it runs, which a gradient-checkpointing recompute re-enters.

The full-attention log scaling reads the row offset instead of ``position_ids``,
so its queries and position bias are rescaled to each document's own positions.
"""

import functools
import importlib

import torch

from axolotl.monkeypatch.models.mamba.modeling import (
    PackedSegments,
    packed_causal_conv,
)
from axolotl.monkeypatch.models.mamba_utils import get_seq_idx, kernel_accepts
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _log_scale(config, positions):
    n_floor = config.log_scaling_n_floor
    effective_n = (positions + 1).float()
    return 1.0 + config.log_scaling_alpha * torch.log(
        (effective_n / n_floor).clamp(min=1.0)
    )


def _rescale_log_scaling(attn, position_ids):
    """Hooks turning the attention's row-offset log scale into per-document ones."""
    if attn.is_sliding or attn.config.log_scaling_n_floor is None:
        return []
    row = torch.arange(position_ids.shape[-1], device=position_ids.device)
    ratio = _log_scale(attn.config, position_ids) / _log_scale(attn.config, row)

    def scale(view):
        return lambda module, args, out: (
            out.float() * ratio.to(out.device).view(view)
        ).to(out.dtype)

    return [
        attn.q_norm.register_forward_hook(scale((*ratio.shape, 1, 1))),
        attn.rel_logits_proj.register_forward_hook(scale((ratio.shape[0], 1, -1, 1))),
    ]


def patch_inkling_modeling_packing() -> None:
    mod = importlib.import_module("transformers.models.inkling.modeling_inkling")
    if getattr(mod.InklingDecoderLayer.forward, "_axolotl_packing_patch", False):
        return
    model_forward = mod.InklingTextModel.forward
    layer_forward = mod.InklingDecoderLayer.forward

    @functools.wraps(model_forward)
    def patched_model_forward(self, *args, **kwargs):
        position_ids = kwargs.get("position_ids")
        if position_ids is not None and kwargs.get("past_key_values") is None:
            kwargs["packed_segments"] = PackedSegments(get_seq_idx(position_ids))
            kwargs["packed_position_ids"] = position_ids
        return model_forward(self, *args, **kwargs)

    @functools.wraps(layer_forward)
    def patched_layer_forward(
        self, *args, packed_segments=None, packed_position_ids=None, **kwargs
    ):
        if packed_segments is None or kwargs.get("past_key_values") is not None:
            return layer_forward(self, *args, **kwargs)
        conv = mod.causal_conv1d_fn

        def packed_conv(x, *conv_args, **conv_kwargs):
            if kernel_accepts(conv, "seq_idx") is True:
                conv_kwargs["seq_idx"] = packed_segments.seq_idx.to(x.device)
                return conv(x, *conv_args, **conv_kwargs)
            conv_kwargs.pop("seq_idx", None)
            return packed_causal_conv(
                conv, packed_segments, x, *conv_args, **conv_kwargs
            )

        # multipack attention masks carry one id per document; the convs only
        # multiply by it to zero padding
        if kwargs.get("conv_mask") is not None:
            kwargs["conv_mask"] = kwargs["conv_mask"] != 0
        hooks = _rescale_log_scaling(self.self_attn, packed_position_ids)
        mod.causal_conv1d_fn = packed_conv
        try:
            return layer_forward(self, *args, **kwargs)
        finally:
            mod.causal_conv1d_fn = conv
            for hook in hooks:
                hook.remove()

    patched_layer_forward._axolotl_packing_patch = True
    mod.InklingTextModel.forward = patched_model_forward
    mod.InklingDecoderLayer.forward = patched_layer_forward
    LOG.info("Applied inkling sample packing patch (per-document short convs)")
