"""Native WeightConverter so Gemma-4 NVFP4 MoE experts load as NVFP4Tensor.

nvidia/Gemma-4-26B-A4B-NVFP4 ships per-expert weights under
  ``model.language_model.layers.N.experts.E.{gate_proj,up_proj,down_proj}.{weight,weight_scale,weight_scale_2}``
but ``quant_method: modelopt`` is not a recognized transformers quantizer, so
the model loads as a BF16 skeleton with the per-expert NVFP4 tensors landing
as UNEXPECTED and the fused ``gate_up_proj``/``down_proj`` remaining random BF16.

This module registers a ``WeightConverter`` for the ``gemma4_text`` model type
that fuses the per-expert raw uint8 qdata + E4M3 block scales + per-tensor
scalar into a single ``NVFP4Tensor`` (packed 4-bit) and assigns it in-place to
the ``Gemma4TextExperts`` module — exactly like ``Mxfp4Deserialize`` does for
MXFP4.  ``is_nvfp4_param(param)`` returns ``True`` on the result, activating
the scattermoe fused NVFP4 path.

The fusion logic mirrors ``nvfp4_moe_loading._build_expert_nvfp4`` exactly:
  gate_up qdata  = stack-experts then cat([gate, up], dim=1)  → [E, 2*I, H/2] uint8
  gate_up scale  = same                                        → [E, 2*I, H/16] e4m3
  gate_up pts    = scalar (shared between gate and up)
  down qdata     = stack-experts                               → [E, H, I/2] uint8
  down scale     = stack-experts                               → [E, H, I/16] e4m3

Registration is done via ``transformers.conversion_mapping.register_checkpoint_conversion_mapping``
— no site-packages edits.  The registration helper is gated: call it only when
the model is gemma4 + NVFP4 modelopt.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor
    except ImportError:
        return None


class Nvfp4ExpertsDeserialize:
    """ConversionOps that fuses per-expert NVFP4 tensors into a single NVFP4Tensor.

    For gate_up_proj, ``input_dict`` contains four keys (the source_patterns):
      - ``"experts.*.gate_proj.weight"``  → list of E uint8 tensors  [I, H/2]
      - ``"experts.*.up_proj.weight"``    → list of E uint8 tensors  [I, H/2]
      - ``"experts.*.gate_proj.weight_scale"``  → list of E e4m3 tensors [I, H/16]
      - ``"experts.*.up_proj.weight_scale"``    → list of E e4m3 tensors [I, H/16]
      - ``"experts.*.gate_proj.weight_scale_2"`` → list of E float32 scalars (use first)

    For down_proj, ``input_dict`` contains:
      - ``"experts.*.down_proj.weight"``         → list of E uint8 tensors  [H, I/2]
      - ``"experts.*.down_proj.weight_scale"``   → list of E e4m3 tensors [H, I/16]
      - ``"experts.*.down_proj.weight_scale_2"`` → list of E float32 scalars

    The op attaches the fused NVFP4Tensor to the module in-place and returns ``{}``
    so the loader does not try to materialize the original meta-parameter names.
    """

    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str] | None = None,
        target_patterns: list[str] | None = None,
        full_layer_name: str | None = None,
        model: nn.Module | None = None,
        missing_keys: set | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        from transformers.quantizers.quantizers_utils import get_module_from_name

        NVFP4Tensor = _nvfp4_cls()
        if NVFP4Tensor is None:
            raise RuntimeError(
                "torchao.prototype.mx_formats.nvfp4_tensor.NVFP4Tensor not found; "
                "install torchao with NVFP4 support"
            )

        if full_layer_name is None or "gate_up_proj" not in full_layer_name:
            proj = "down_proj"
        else:
            proj = "gate_up_proj"

        def _find(pat_suffix: str) -> list[torch.Tensor]:
            """Find the tensor list for a source pattern that ends with pat_suffix."""
            for key, tensors in input_dict.items():
                if key.endswith(pat_suffix):
                    return tensors
            raise KeyError(
                f"Nvfp4ExpertsDeserialize: could not find '{pat_suffix}' in "
                f"input_dict keys: {list(input_dict.keys())}"
            )

        # spawn_materialize casts all checkpoint tensors to the skeleton dtype (bf16) before
        # the converter sees them. uint8 qdata (0-255) and float8_e4m3fn scales both roundtrip
        # exactly through bf16, so recast back to the raw dtypes NVFP4Tensor needs.
        def _recast_weight(t: torch.Tensor) -> torch.Tensor:
            if t.dtype != torch.uint8:
                return t.to(torch.int32).to(torch.uint8)
            return t

        def _recast_scale(t: torch.Tensor) -> torch.Tensor:
            if t.dtype != torch.float8_e4m3fn:
                return t.to(torch.float8_e4m3fn)
            return t

        if proj == "gate_up_proj":
            gate_w = [_recast_weight(t) for t in _find("gate_proj.weight")]
            up_w = [_recast_weight(t) for t in _find("up_proj.weight")]
            gate_sc = [_recast_scale(t) for t in _find("gate_proj.weight_scale")]
            up_sc = [_recast_scale(t) for t in _find("up_proj.weight_scale")]
            pts_list = _find("gate_proj.weight_scale_2")

            gate_qd = torch.stack(gate_w, dim=0)  # [E, I, H/2]
            up_qd = torch.stack(up_w, dim=0)  # [E, I, H/2]
            qdata = torch.cat([gate_qd, up_qd], dim=1)  # [E, 2I, H/2]
            del gate_qd, up_qd

            gate_s = torch.stack(gate_sc, dim=0)  # [E, I, H/16]
            up_s = torch.stack(up_sc, dim=0)  # [E, I, H/16]
            scale = torch.cat([gate_s, up_s], dim=1)  # [E, 2I, H/16]
            del gate_s, up_s

            # Per-expert weight_scale_2 stacked to [E,1,1] (gate/up share it), not expert-0's scalar.
            pts = torch.stack([t.to(torch.float32) for t in pts_list]).view(-1, 1, 1)

        else:  # down_proj
            down_w = [_recast_weight(t) for t in _find("down_proj.weight")]
            down_sc = [_recast_scale(t) for t in _find("down_proj.weight_scale")]
            pts_list = _find("down_proj.weight_scale_2")

            qdata = torch.stack(down_w, dim=0)  # [E, H, I/2]
            scale = torch.stack(down_sc, dim=0)  # [E, H, I/16]
            pts = torch.stack([t.to(torch.float32) for t in pts_list]).view(-1, 1, 1)

        nvfp4 = NVFP4Tensor(qdata, scale, 16, torch.bfloat16, per_tensor_scale=pts)

        module, _ = get_module_from_name(model, full_layer_name)
        setattr(module, proj, nn.Parameter(nvfp4, requires_grad=False))

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        module._is_hf_initialized = True

        LOG.debug(
            "Nvfp4ExpertsDeserialize: set %s as NVFP4Tensor [%s]",
            full_layer_name,
            list(qdata.shape),
        )
        return {}

    # No meaningful reverse op (packed NVFP4 → checkpoint would need to unfuse).
    @property
    def reverse_op(self):
        from transformers.core_model_loading import _IdentityOp

        return _IdentityOp()


def nvfp4_experts_weight_converters() -> list:
    """Return the two WeightConverter instances for gemma4 NVFP4 experts.

    These are registered under ``"gemma4_text"`` in the transformers
    conversion_mapping cache so the loader finds and applies them during
    ``from_pretrained``.
    """
    from transformers.core_model_loading import WeightConverter

    op = Nvfp4ExpertsDeserialize()

    # Source patterns MUST be ordered longest-suffix-first. transformers compiles them into a
    # single ``(?P<g0>...)|(?P<g1>...)`` alternation and resolves a key with ``re.search`` +
    # first-non-None group (core_model_loading.py). The patterns are NOT end-anchored when the
    # converter is many-to-one (the ^...$ anchoring only runs for equal-length source/target
    # lists), so ``...weight`` would substring-match inside ``...weight_scale``/``...weight_scale_2``
    # and steal those keys unless the more specific suffixes appear first.
    gate_up_converter = WeightConverter(
        source_patterns=[
            # gate and up each ship their own weight_scale_2 scalar; claim BOTH so neither
            # lands as an UNEXPECTED key (the op only reads the first; they're identical).
            "experts.*.gate_proj.weight_scale_2",
            "experts.*.up_proj.weight_scale_2",
            "experts.*.gate_proj.weight_scale",
            "experts.*.up_proj.weight_scale",
            "experts.*.gate_proj.weight",
            "experts.*.up_proj.weight",
        ],
        target_patterns="experts.gate_up_proj",
        operations=[op],
    )

    down_converter = WeightConverter(
        source_patterns=[
            "experts.*.down_proj.weight_scale_2",
            "experts.*.down_proj.weight_scale",
            "experts.*.down_proj.weight",
        ],
        target_patterns="experts.down_proj",
        operations=[op],
    )

    return [gate_up_converter, down_converter]


def register_gemma4_nvfp4_converters() -> None:
    """Seed the transformers conversion_mapping cache with NVFP4 expert converters
    for ``gemma4_text``.

    Safe to call multiple times (idempotent via overwrite=True on re-entry).
    Does not touch DSV4, bf16 gemma4, or any other model type.
    """
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    converters = nvfp4_experts_weight_converters()
    try:
        register_checkpoint_conversion_mapping("gemma4_text", converters)
    except ValueError:
        # Already registered; overwrite to keep converters fresh.
        register_checkpoint_conversion_mapping(
            "gemma4_text", converters, overwrite=True
        )

    LOG.info(
        "Registered gemma4_text NVFP4 expert WeightConverters in transformers conversion_mapping"
    )
