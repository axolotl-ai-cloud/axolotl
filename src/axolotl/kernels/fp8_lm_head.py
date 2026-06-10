"""Default-off FP8 lm_head projection for eval/logprob throughput."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Literal

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

FP8Granularity = Literal["tensorwise", "rowwise"]

_E4M3_MAX = 448.0


@dataclass(frozen=True)
class FP8LMHeadWeight:
    weight_t: torch.Tensor
    scale: torch.Tensor
    granularity: FP8Granularity
    out_features: int
    in_features: int

    @property
    def persistent_bytes(self) -> int:
        return (
            self.weight_t.numel() * self.weight_t.element_size()
            + self.scale.numel() * self.scale.element_size()
        )


def _scale_from_amax(amax: torch.Tensor) -> torch.Tensor:
    return (amax.float().clamp_min(1.0e-12) / _E4M3_MAX).contiguous()


def _to_col_major_for_scaled_mm(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.stride(0) == 1:
        return tensor
    return tensor.t().contiguous().t()


def _quantize_e4m3(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = (x.float() / scale).clamp(-_E4M3_MAX, _E4M3_MAX)
    return q.to(torch.float8_e4m3fn)


def _scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    out = torch._scaled_mm(
        a,
        b,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )
    return out[0] if isinstance(out, tuple) else out


def prepack_lm_head_weight_fp8(
    weight: torch.Tensor,
    *,
    granularity: FP8Granularity = "rowwise",
) -> FP8LMHeadWeight:
    """Prepack ``nn.Linear`` lm_head weight ``[vocab, hidden]`` to FP8."""
    if weight.ndim != 2:
        raise ValueError(f"lm_head weight must be 2D, got {tuple(weight.shape)}")
    if weight.device.type != "cuda":
        raise ValueError("FP8 lm_head requires a CUDA weight")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable in this PyTorch build")
    if granularity not in ("tensorwise", "rowwise"):
        raise ValueError(f"unsupported FP8 lm_head granularity: {granularity}")

    vocab, hidden = weight.shape
    weight_t = weight.detach().t()
    if granularity == "tensorwise":
        scale = _scale_from_amax(weight_t.abs().max())
    else:
        scale = _scale_from_amax(weight_t.abs().amax(dim=0, keepdim=True))
    weight_t_fp8 = _quantize_e4m3(weight_t, scale)
    return FP8LMHeadWeight(
        weight_t=_to_col_major_for_scaled_mm(weight_t_fp8),
        scale=scale,
        granularity=granularity,
        out_features=vocab,
        in_features=hidden,
    )


def fp8_lm_head(
    hidden: torch.Tensor,
    packed_weight: FP8LMHeadWeight,
    *,
    input_granularity: FP8Granularity | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``hidden @ lm_head.weight.T`` with FP8 operands."""
    if hidden.device.type != "cuda":
        raise ValueError("FP8 lm_head requires CUDA hidden states")
    if hidden.shape[-1] != packed_weight.in_features:
        raise ValueError(
            f"hidden last dim {hidden.shape[-1]} does not match packed weight "
            f"{packed_weight.in_features}"
        )

    granularity = input_granularity or packed_weight.granularity
    if granularity not in ("tensorwise", "rowwise"):
        raise ValueError(f"unsupported FP8 input granularity: {granularity}")

    lead = hidden.shape[:-1]
    hidden_2d = hidden.reshape(-1, hidden.shape[-1]).contiguous()
    if granularity == "tensorwise":
        x_scale = _scale_from_amax(hidden_2d.abs().max())
    else:
        x_scale = _scale_from_amax(hidden_2d.abs().amax(dim=1, keepdim=True))
    hidden_fp8 = _quantize_e4m3(hidden_2d, x_scale)

    out = _scaled_mm(
        hidden_fp8,
        packed_weight.weight_t,
        scale_a=x_scale,
        scale_b=packed_weight.scale,
        out_dtype=out_dtype or hidden.dtype,
    )
    return out.reshape(*lead, packed_weight.out_features)


def _packed_cache_key(
    weight: torch.Tensor,
) -> tuple[int, int, tuple[int, ...], torch.dtype]:
    return (weight.data_ptr(), weight._version, tuple(weight.shape), weight.dtype)


def _get_or_prepack_lm_head_weight_fp8(
    lm_head: nn.Linear,
    *,
    granularity: FP8Granularity,
) -> FP8LMHeadWeight:
    key = (_packed_cache_key(lm_head.weight), granularity)
    cached = getattr(lm_head, "_axolotl_fp8_lm_head_packed", None)
    if cached is None or cached[0] != key:
        packed = prepack_lm_head_weight_fp8(lm_head.weight, granularity=granularity)
        lm_head._axolotl_fp8_lm_head_packed = (key, packed)
    else:
        packed = cached[1]
    return packed


def _fp8_forward(self: nn.Linear, hidden: torch.Tensor) -> torch.Tensor:
    orig_forward = self._axolotl_fp8_lm_head_orig_forward
    if torch.is_grad_enabled() or self.training:
        return orig_forward(hidden)
    if hidden.device.type != "cuda" or self.weight.device.type != "cuda":
        return orig_forward(hidden)

    packed = _get_or_prepack_lm_head_weight_fp8(
        self,
        granularity=self._axolotl_fp8_lm_head_granularity,
    )
    out = fp8_lm_head(
        hidden,
        packed,
        input_granularity=self._axolotl_fp8_lm_head_granularity,
        out_dtype=hidden.dtype,
    )
    if self.bias is not None:
        out = out + self.bias
    return out


def patch_model_fp8_lm_head(
    model: nn.Module,
    *,
    granularity: FP8Granularity = "rowwise",
) -> bool:
    """Patch a model's existing lm_head Linear to use FP8 in eval/no-grad forward."""
    if granularity not in ("tensorwise", "rowwise"):
        raise ValueError(f"unsupported FP8 lm_head granularity: {granularity}")
    try:
        lm_head = model.get_output_embeddings()
    except (AttributeError, NotImplementedError):
        LOG.warning("fp8_lm_head: model has no output embeddings; skipping")
        return False
    if type(lm_head) is not nn.Linear:
        LOG.warning(
            "fp8_lm_head: output embedding is %s, not a plain nn.Linear; skipping",
            type(lm_head).__name__,
        )
        return False
    if not hasattr(torch, "_scaled_mm"):
        LOG.warning("fp8_lm_head: torch._scaled_mm is unavailable; skipping")
        return False
    if getattr(lm_head, "_axolotl_fp8_lm_head_patched", False):
        lm_head._axolotl_fp8_lm_head_granularity = granularity
        LOG.info("fp8_lm_head: updated existing patch (granularity=%s)", granularity)
        return True

    lm_head._axolotl_fp8_lm_head_orig_forward = lm_head.forward
    lm_head._axolotl_fp8_lm_head_granularity = granularity
    lm_head._axolotl_fp8_lm_head_patched = True
    lm_head.forward = MethodType(_fp8_forward, lm_head)
    LOG.info("fp8_lm_head: patched lm_head eval/no-grad forward (%s)", granularity)
    return True
