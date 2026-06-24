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

The fusion itself (stack experts, cat gate/up on the N axis, reconcile the per-tensor scales)
is the shared ``nvfp4_moe_loading.fuse_nvfp4_experts`` core, so the WeightConverter (modelopt
skeleton load) and the post-load scale-attach path use one implementation instead of duplicating
the NVFP4-expert math.

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

        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            fuse_nvfp4_experts,
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

        def _proj_parts(proj_name: str) -> dict:
            return {
                "qd": [_recast_weight(t) for t in _find(f"{proj_name}.weight")],
                "sc": [_recast_scale(t) for t in _find(f"{proj_name}.weight_scale")],
                "pts": list(_find(f"{proj_name}.weight_scale_2")),
            }

        # gate/up fuse on the N axis (each ships its own per-tensor scale, reconciled in the core);
        # down is a single projection. Fusion + scale reconciliation live in fuse_nvfp4_experts.
        if proj == "gate_up_proj":
            projs = [_proj_parts("gate_proj"), _proj_parts("up_proj")]
        else:
            projs = [_proj_parts("down_proj")]
        nvfp4 = fuse_nvfp4_experts(projs)

        module, _ = get_module_from_name(model, full_layer_name)
        setattr(module, proj, nn.Parameter(nvfp4, requires_grad=False))

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        module._is_hf_initialized = True

        LOG.debug(
            "Nvfp4ExpertsDeserialize: set %s as NVFP4Tensor [%s]",
            full_layer_name,
            list(nvfp4.shape),
        )
        return {}

    # No meaningful reverse op (packed NVFP4 → checkpoint would need to unfuse).
    @property
    def reverse_op(self):
        from transformers.core_model_loading import _IdentityOp

        return _IdentityOp()


class Nvfp4LinearDequantize:
    """ConversionOps that dequantizes one NVFP4 ``nn.Linear`` weight to bf16 in place.

    Consumes a single module's NVFP4 triple — ``weight`` (uint8 qdata ``[out, in/2]``),
    ``weight_scale`` (e4m3 group-16 ``[out, in/16]``), ``weight_scale_2`` (per-tensor scalar) —
    and assigns ``dequantize(NVFP4Tensor(...))`` to the target ``.weight`` param. Used for whatever
    non-routed linears a given checkpoint quantizes (the caller decides which, from the index);
    this op makes no assumption about which modules those are. ``input_dict`` holds the three
    source tensors (each a 1-element list under a non-wildcard source pattern); ``full_layer_name``
    is the target ``....weight``.
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

        def _one(pat_suffix: str) -> torch.Tensor:
            for key, val in input_dict.items():
                if key.endswith(pat_suffix):
                    return val[0] if isinstance(val, (list, tuple)) else val
            raise KeyError(
                f"Nvfp4LinearDequantize: could not find '{pat_suffix}' in "
                f"input_dict keys: {list(input_dict.keys())}"
            )

        # spawn_materialize casts checkpoint tensors to the skeleton dtype (bf16); recast the
        # raw uint8 qdata / e4m3 scale back (both roundtrip exactly through bf16).
        w = _one(".weight")
        qdata = w if w.dtype == torch.uint8 else w.to(torch.int32).to(torch.uint8)
        sc = _one(".weight_scale")
        scale = sc if sc.dtype == torch.float8_e4m3fn else sc.to(torch.float8_e4m3fn)
        pts = _one(".weight_scale_2").to(torch.float32).view(1, 1)

        weight = NVFP4Tensor(
            qdata, scale, 16, torch.bfloat16, per_tensor_scale=pts
        ).dequantize(torch.bfloat16)

        module, param_name = get_module_from_name(model, full_layer_name)
        setattr(module, param_name, nn.Parameter(weight, requires_grad=False))
        if missing_keys is not None:
            missing_keys.discard(full_layer_name)
        module._is_hf_initialized = True
        return {}

    @property
    def reverse_op(self):
        from transformers.core_model_loading import _IdentityOp

        return _IdentityOp()


def _nvfp4_linear_dequant_converter(target_weight: str):
    """A WeightConverter that dequantizes one NVFP4 linear (``<target_weight>``) to bf16.

    ``target_weight`` is the distinctive suffix of the linear's ``.weight`` param (e.g.
    ``"shared_experts.gate_proj.weight"``). The three source patterns claim the NVFP4 triple so
    none lands UNEXPECTED; longest-suffix-first so ``.weight`` doesn't steal ``.weight_scale*``.
    """
    from transformers.core_model_loading import WeightConverter

    base = target_weight[: -len(".weight")]
    return WeightConverter(
        source_patterns=[
            f"{base}.weight_scale_2",
            f"{base}.weight_scale",
            f"{base}.weight",
        ],
        target_patterns=target_weight,
        operations=[Nvfp4LinearDequantize()],
    )


def nonrouted_dequant_converters(nonrouted_suffixes: list[str]) -> list:
    """Dequant converters for the non-routed NVFP4 linears a checkpoint actually quantizes.

    ``nonrouted_suffixes`` are the layer-relative module paths detected from the safetensors
    index (e.g. ``"mlp.shared_experts.gate_proj"``, ``"mlp.gate_proj"``) — NOT hardcoded — each
    pointing at a ``.weight`` to dequantize to bf16. Empty list -> no converters (e.g. a
    checkpoint that leaves all non-routed linears bf16)."""
    return [
        _nvfp4_linear_dequant_converter(f"{suf}.weight") for suf in nonrouted_suffixes
    ]


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
            # gate and up each ship their own weight_scale_2 scalar; claim BOTH so neither lands
            # as an UNEXPECTED key (the op reconciles them — equal in practice, folded if not).
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


def register_nvfp4_expert_converters(
    model_type: str, include_routed: bool = True, extra: list | None = None
) -> None:
    """Seed the transformers conversion_mapping cache with NVFP4 converters for ``model_type``.

    The routed-expert converters (``Nvfp4ExpertsDeserialize`` + the ``experts.*.{proj}.weight*``
    source patterns) fuse per-expert ``gate/up/down`` into the model's 3D expert params; they are
    registered when ``include_routed`` (gate this on the checkpoint actually exporting per-expert
    NVFP4). ``extra`` carries any per-checkpoint non-routed dequant converters (built from the
    detected index layout). The only per-model knob is which ``model_type`` the loader looks the
    mapping up under. Safe to call repeatedly (idempotent via overwrite on re-entry).
    """
    from transformers.conversion_mapping import register_checkpoint_conversion_mapping

    converters = (nvfp4_experts_weight_converters() if include_routed else []) + list(
        extra or []
    )
    if not converters:
        return
    try:
        register_checkpoint_conversion_mapping(model_type, converters)
    except ValueError:
        # Already registered; overwrite to keep converters fresh.
        register_checkpoint_conversion_mapping(model_type, converters, overwrite=True)

    LOG.info(
        "Registered %s NVFP4 WeightConverters (%d) in transformers conversion_mapping",
        model_type,
        len(converters),
    )


def register_gemma4_nvfp4_converters() -> None:
    """Register NVFP4 expert converters for ``gemma4_text`` (see register_nvfp4_expert_converters)."""
    register_nvfp4_expert_converters("gemma4_text")


def register_nvfp4_converters_for_layout(model_type: str, layout: dict) -> None:
    """Register NVFP4 converters built from a detected checkpoint ``layout`` (see
    :func:`...nvfp4_moe_loading.inspect_nvfp4_layout`): routed experts fused into packed
    NVFP4Tensor (when present), and each detected non-routed NVFP4 linear dequantized to bf16."""
    register_nvfp4_expert_converters(
        model_type,
        include_routed=layout.get("routed_present", False),
        extra=nonrouted_dequant_converters(layout.get("nonrouted_suffixes", [])),
    )
