"""Packed-sequence position ids for mRoPE VLMs.

transformers' mRoPE text models (Qwen2-VL family, GLM4V) only isolate packed
samples when given ``[4, B, L]`` position ids: a text row followed by the three
mRoPE grids. A plain ``(B, L)`` row is expanded to three identical grids, which
silently disables both the block-diagonal mask and ``get_rope_index``.
"""

from __future__ import annotations

import inspect

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def restart_positions_per_segment(
    position_ids: torch.Tensor, text_position_ids: torch.Tensor
) -> torch.Tensor:
    """Rebase ``(3, B, L)`` mRoPE positions so each packed segment starts from 0.

    Segments are read off the text row (a restart is any step that is not +1).
    ``get_rope_index`` is affine in its running offset on every axis, so the
    per-sample result is the whole-row result minus the segment's first-token
    position.
    """
    seq_len = text_position_ids.shape[-1]
    starts = torch.ones_like(text_position_ids, dtype=torch.bool)
    starts[:, 1:] = text_position_ids[:, 1:] != text_position_ids[:, :-1] + 1
    index = torch.arange(seq_len, device=text_position_ids.device).expand_as(
        text_position_ids
    )
    segment_start = torch.where(starts, index, 0).cummax(-1).values
    offsets = torch.gather(position_ids, 2, segment_start[None].expand_as(position_ids))
    return position_ids - offsets


def _fallback_mm_token_type_ids(module, input_ids: torch.Tensor) -> torch.Tensor:
    config = module.config
    token_types = torch.zeros_like(input_ids)
    image_token_id = getattr(config, "image_token_id", None)
    video_token_id = getattr(config, "video_token_id", None)
    if image_token_id is not None:
        token_types[input_ids == image_token_id] = 1
    if video_token_id is not None:
        token_types[input_ids == video_token_id] = 2
    return token_types


def build_packed_mrope_position_ids(
    module,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    **rope_kwargs,
) -> torch.Tensor:
    """Return ``[4, B, L]`` position ids for a packed batch from its ``(B, L)`` text positions."""
    if image_grid_thw is None and video_grid_thw is None:
        return position_ids[None].expand(4, -1, -1)
    if mm_token_type_ids is None:
        mm_token_type_ids = _fallback_mm_token_type_ids(module, input_ids)
    accepted = inspect.signature(module.get_rope_index).parameters
    rope_kwargs = {k: v for k, v in rope_kwargs.items() if k in accepted}
    with torch.no_grad():
        packed_positions, _ = module.get_rope_index(
            input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=None,
            **rope_kwargs,
        )
    packed_positions = restart_positions_per_segment(
        packed_positions.to(position_ids.device), position_ids
    )
    return torch.cat([position_ids[None], packed_positions], dim=0)


def _mrope_packing_pre_hook(module, args, kwargs):
    position_ids = kwargs.get("position_ids")
    input_ids = kwargs.get("input_ids")
    if position_ids is None or position_ids.ndim != 2 or input_ids is None:
        return None
    kwargs["position_ids"] = build_packed_mrope_position_ids(
        module,
        input_ids,
        position_ids,
        image_grid_thw=kwargs.get("image_grid_thw"),
        video_grid_thw=kwargs.get("video_grid_thw"),
        mm_token_type_ids=kwargs.get("mm_token_type_ids"),
        second_per_grid_ts=kwargs.get("second_per_grid_ts"),
    )
    return args, kwargs


def find_mrope_model(model) -> torch.nn.Module | None:
    for module in model.modules():
        if hasattr(module, "get_rope_index") and hasattr(module, "language_model"):
            return module
    return None


def patch_mrope_packing(model) -> bool:
    """Install the pre-hook; returns False for models without mRoPE (1-D RoPE VLMs need nothing)."""
    target = find_mrope_model(model)
    if target is None:
        return False
    if getattr(target, "_axolotl_mrope_packing_hooked", False):
        return True
    target.register_forward_pre_hook(_mrope_packing_pre_hook, with_kwargs=True)
    target._axolotl_mrope_packing_hooked = True
    LOG.info(
        "sample_packing: %s builds [4, B, L] mRoPE position ids for packed batches",
        type(target).__name__,
    )
    return True
