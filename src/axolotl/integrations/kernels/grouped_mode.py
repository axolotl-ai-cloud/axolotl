"""Resolve ``moe_grouped_mode`` for the scattermoe grouped-expert path.

``auto`` (the default under ``use_scattermoe``) runs the routed experts in the format they load
in: NVFP4 experts on the NVFP4 grouped GEMM, 128x128 block-FP8 experts on the block-FP8 one, and
everything else on the plain scattermoe kernels. The format comes from the checkpoint's
safetensors headers (falling back to its ``quantization_config``) before any weight loads. An
explicit mode is checked against it: a lossy requant needs ``moe_grouped_requant: true``, and a
mode the experts can't run in is a config error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

GROUPED_MODES = ("nvfp4", "fp8_block", "mxfp8")

# routed-expert formats
NVFP4 = "nvfp4"
BLOCK_FP8 = "block_fp8"
BF16 = "bf16"
PACKED_FP4 = "packed_fp4"
MIXED = "mixed"
OTHER = "other"  # quantized in a format no grouped mode reads (bnb, mxfp4, per-tensor fp8, ...)
NO_EXPERTS = "no_experts"

_DESCRIBE = {
    NVFP4: "NVFP4",
    BLOCK_FP8: "128x128 block-FP8",
    BF16: "unquantized (bf16)",
    PACKED_FP4: "DeepSeek packed FP4",
    OTHER: "quantized in a format no grouped mode reads",
}
# the mode that runs each format as it loads
_NATIVE = {NVFP4: "nvfp4", BLOCK_FP8: "fp8_block"}
# lossy conversions an explicit mode may opt into, with their fidelity cost
_REQUANT = {
    (
        BLOCK_FP8,
        "nvfp4",
    ): "block-FP8 -> NVFP4: 4-bit experts, ~17x the fp8_block weight error",
    (
        BLOCK_FP8,
        "mxfp8",
    ): "block-FP8 -> mxfp8: every expert weight re-rounded to e4m3 with "
    "per-32 e8m0 scales",
    (BF16, "mxfp8"): "bf16 -> mxfp8: 8-bit experts with per-32 e8m0 scales",
}

_EXPERT_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:.+\.)?experts\.(.+)$")
_MTP_RE = re.compile(r"(?:^|\.)mtp[^.]*\.")
_PER_EXPERT_WEIGHT_RE = re.compile(r"\d+\.[^.]+\.(weight|weight_packed)")
_FUSED_WEIGHT_RE = re.compile(r"[A-Za-z_]\w*")
_FUSED_AUX = ("_scale_inv", "_scale", "_scales", "_scale_2", "_mx_scale", "_bias")
_SCALE_LEAVES = (
    "weight_scale_inv",
    "scale_inv",
    "scale",
    "qscale_weight",
    "weight_scale",
)


@dataclass(frozen=True)
class ExpertFormat:
    kind: str
    reason: str


@dataclass(frozen=True)
class GroupedModeChoice:
    mode: str | None
    explicit: bool
    reason: str
    requant: str | None = None


def _classify(dtype: str, shape: tuple, siblings: dict) -> str:
    dt = str(dtype).upper()
    if dt == "U8" and "weight_scale" in siblings:
        return NVFP4
    if dt.startswith("F8"):
        for leaf in _SCALE_LEAVES:
            if leaf in siblings:
                # block scales tile the weight; per-tensor ones are scalar or one per expert
                return BLOCK_FP8 if len(siblings[leaf][1]) == len(shape) else OTHER
        return OTHER
    if dt in ("I8", "U8") and "scale" in siblings:
        return PACKED_FP4
    if dt in ("BF16", "F16", "F32"):
        return BF16
    return OTHER


def _expert_layer_formats(meta: dict, num_layers: int | None) -> dict[int, set[str]]:
    """``{layer: {format, ...}}`` of every routed-expert weight in the checkpoint headers."""
    layers: dict[int, set[str]] = {}
    for name, (dtype, shape) in meta.items():
        m = _EXPERT_RE.search(name)
        if m is None:
            continue
        layer, rest = int(m.group(1)), m.group(2)
        # multi-token-prediction layers (``mtp.layers.N`` or past num_hidden_layers) don't load
        if _MTP_RE.search(name) or (num_layers is not None and layer >= num_layers):
            continue
        if _PER_EXPERT_WEIGHT_RE.fullmatch(rest):
            prefix = name[: name.rindex(".") + 1]
            siblings = {
                leaf: meta[prefix + leaf]
                for leaf in _SCALE_LEAVES
                if prefix + leaf in meta
            }
        elif _FUSED_WEIGHT_RE.fullmatch(rest) and not rest.endswith(_FUSED_AUX):
            if rest.endswith("_blocks"):  # gpt-oss mxfp4
                layers.setdefault(layer, set()).add(OTHER)
                continue
            siblings = {
                s.lstrip("_"): meta[name + s]
                for s in ("_scale_inv", "_scale")
                if name + s in meta
            }
        else:
            continue
        layers.setdefault(layer, set()).add(_classify(dtype, shape, siblings))
    return layers


def _ranges(layers: list[int]) -> str:
    out, start = [], None
    for i, layer in enumerate(layers):
        if start is None:
            start = layer
        if i + 1 == len(layers) or layers[i + 1] != layer + 1:
            out.append(str(start) if start == layer else f"{start}-{layer}")
            start = None
    return ",".join(out)


def _num_layers(model_config) -> int | None:
    if model_config is None:
        return None
    text = (
        model_config.get_text_config()
        if hasattr(model_config, "get_text_config")
        else model_config
    )
    return getattr(text, "num_hidden_layers", None)


def _qcfg_dict(model_config) -> dict | None:
    qcfg = getattr(model_config, "quantization_config", None)
    if qcfg is None or isinstance(qcfg, dict):
        return qcfg
    return qcfg.to_dict() if hasattr(qcfg, "to_dict") else vars(qcfg)


def _experts_excluded(modules_to_not_convert) -> bool:
    return any("experts" in str(m) for m in modules_to_not_convert or ())


def _from_quantization_config(model_config, why: str) -> ExpertFormat:
    """The experts' format from ``quantization_config`` alone (headers unavailable)."""
    qcfg = _qcfg_dict(model_config)
    if not qcfg:
        return ExpertFormat(BF16, f"no quantization_config ({why})")
    method = qcfg.get("quant_method")
    if method == "fp8":
        if str(qcfg.get("moe_quant_algo", "")).upper() == "NVFP4":
            return ExpertFormat(
                NVFP4, f"quantization_config moe_quant_algo NVFP4 ({why})"
            )
        expert_dtype = qcfg.get("expert_dtype") or getattr(
            model_config, "expert_dtype", None
        )
        if expert_dtype == "fp4":
            return ExpertFormat(
                PACKED_FP4, f"quantization_config expert_dtype fp4 ({why})"
            )
        if _experts_excluded(qcfg.get("modules_to_not_convert")):
            return ExpertFormat(BF16, f"experts in modules_to_not_convert ({why})")
        if not qcfg.get("weight_block_size"):
            return ExpertFormat(OTHER, f"per-tensor fp8 quantization_config ({why})")
        return ExpertFormat(
            BLOCK_FP8,
            f"quantization_config fp8 weight_block_size {qcfg['weight_block_size']} ({why})",
        )
    if method == "modelopt" and qcfg.get("quant_algo") == "NVFP4":
        return ExpertFormat(NVFP4, f"modelopt quant_algo NVFP4 ({why})")
    if method == "modelopt" and qcfg.get("quant_algo") == "MIXED_PRECISION":
        algos = {
            str(layer.get("quant_algo"))
            for name, layer in (qcfg.get("quantized_layers") or {}).items()
            if name.endswith("experts")
        }
        if algos == {"NVFP4"}:
            return ExpertFormat(
                NVFP4, f"modelopt quantized_layers experts NVFP4 ({why})"
            )
        if len(algos) > 1:
            return ExpertFormat(
                MIXED, f"modelopt expert quant_algos {sorted(algos)} ({why})"
            )
    return ExpertFormat(OTHER, f"quant_method {method!r} ({why})")


def _on_the_fly_fp8(cfg) -> ExpertFormat:
    from transformers import FineGrainedFP8Config

    qcfg = FineGrainedFP8Config(**(cfg.get("model_quantization_config_kwargs") or {}))
    src = "model_quantization_config: FineGrainedFP8Config"
    if qcfg.dequantize or _experts_excluded(qcfg.modules_to_not_convert):
        return ExpertFormat(BF16, f"{src} leaves the experts unquantized")
    if not qcfg.weight_block_size:
        return ExpertFormat(OTHER, f"{src} quantizes per tensor")
    return ExpertFormat(
        BLOCK_FP8,
        f"{src} quantizes the experts to {list(qcfg.weight_block_size)} block FP8 on load",
    )


def detect_expert_format(cfg) -> ExpertFormat:
    """The routed experts' format as the model will load them. Reads config and safetensors
    headers only."""
    if cfg.get("quantize_moe_experts"):
        return ExpertFormat(
            OTHER, "quantize_moe_experts quantizes the experts to bnb on load"
        )
    on_the_fly = cfg.get("model_quantization_config")
    if on_the_fly == "FineGrainedFP8Config":
        return _on_the_fly_fp8(cfg)
    if on_the_fly:
        return ExpertFormat(OTHER, f"model_quantization_config: {on_the_fly}")

    from axolotl.integrations.kernels.adapters import load_base_model_config
    from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
        _safetensors_metadata,
    )

    model_config = load_base_model_config(cfg)
    try:
        layers = _expert_layer_formats(
            _safetensors_metadata(cfg.base_model), _num_layers(model_config)
        )
    except Exception as exc:  # pylint: disable=broad-except
        return _from_quantization_config(model_config, f"headers unreadable: {exc}")
    if not layers:
        return ExpertFormat(NO_EXPERTS, "no routed-expert weights in the checkpoint")

    by_kind: dict[str, list[int]] = {}
    for layer in sorted(layers):
        kind = MIXED if len(layers[layer]) > 1 else next(iter(layers[layer]))
        by_kind.setdefault(kind, []).append(layer)
    if len(by_kind) > 1 or MIXED in by_kind:
        detail = "; ".join(
            f"{_DESCRIBE.get(kind, 'several formats within a layer')} in layers "
            f"{_ranges(ls)}"
            for kind, ls in by_kind.items()
        )
        return ExpertFormat(MIXED, f"routed experts differ across layers: {detail}")
    kind = next(iter(by_kind))
    return ExpertFormat(kind, f"the checkpoint's routed experts are {_DESCRIBE[kind]}")


def _valid_choices(kind: str) -> str:
    choices = ["auto", "none"]
    if kind in _NATIVE:
        choices.append(_NATIVE[kind])
    choices += [
        f"{mode} (with moe_grouped_requant: true)"
        for (src, mode) in _REQUANT
        if src == kind
    ]
    return ", ".join(choices)


def choose_grouped_mode(
    requested, fmt: ExpertFormat, allow_requant: bool = False
) -> GroupedModeChoice:
    """Decide the grouped mode for experts in ``fmt``. ``requested``: None/"auto", "none", or a
    mode. Raises ValueError for a combination that can't run or an unapproved lossy requant."""
    req = "auto" if requested is None else str(requested).lower()
    if req == "none":
        return GroupedModeChoice(None, True, "moe_grouped_mode: none")
    if fmt.kind == PACKED_FP4:
        raise ValueError(
            f"use_scattermoe has no grouped kernel for DeepSeek packed-FP4 experts ({fmt.reason}). "
            "Use the NVFP4 checkpoint (quantization_config.moe_quant_algo: NVFP4), set "
            "moe_grouped_mode: none to run them on transformers' FP8 experts forward without "
            "expert LoRA, or drop use_scattermoe."
        )
    if fmt.kind == MIXED:
        raise ValueError(
            f"use_scattermoe can't pick a grouped expert mode: {fmt.reason}. Every grouped mode "
            "needs one expert format; set moe_grouped_mode: none to run each layer on its own "
            "path, or use a checkpoint with uniform experts."
        )
    if req == "auto":
        mode = _NATIVE.get(fmt.kind)
        return GroupedModeChoice(mode, False, fmt.reason)
    if req not in GROUPED_MODES:
        raise ValueError(f"unknown moe_grouped_mode {requested!r}")
    if _NATIVE.get(fmt.kind) == req:
        return GroupedModeChoice(req, True, fmt.reason)
    if fmt.kind == NO_EXPERTS:
        LOG.warning("moe_grouped_mode: %s has nothing to act on: %s.", req, fmt.reason)
        return GroupedModeChoice(None, True, fmt.reason)
    cost = _REQUANT.get((fmt.kind, req))
    if cost is not None:
        if not allow_requant:
            raise ValueError(
                f"moe_grouped_mode: {req} requantizes the experts ({fmt.reason}): {cost}. Set "
                "moe_grouped_requant: true to accept that, or choose one of: "
                f"{_valid_choices(fmt.kind)}."
            )
        return GroupedModeChoice(req, True, fmt.reason, requant=cost)
    raise ValueError(
        f"moe_grouped_mode: {req} can't run these experts ({fmt.reason}). Valid choices for "
        f"this checkpoint: {_valid_choices(fmt.kind)}."
    )


def grouped_backend(mode: str) -> str | None:
    """The base-GEMM backend ``mode`` runs on this GPU (after the scattermoe runtime is
    configured), or None when none can."""
    if mode == "nvfp4":
        from axolotl.integrations.kernels.libs.scattermoe_lora.grouped_train import (
            _train_backend,
            grouped_fp4_available,
        )

        return _train_backend(mode) if grouped_fp4_available(mode) else None
    if mode == "fp8_block":
        from axolotl.integrations.kernels.libs.scattermoe_lora.fp8_grouped import (
            _fp8_train_backend,
        )

        return _fp8_train_backend()
    if mode == "mxfp8":
        from axolotl.integrations.kernels.libs.scattermoe_lora.mxfp8_grouped import (
            _mx_backend,
        )

        return _mx_backend()
    return None


def configure_moe_grouped_mode(cfg) -> GroupedModeChoice:
    """Resolve ``cfg.moe_grouped_mode`` against the experts' format and this GPU, apply the
    scattermoe runtime, and log the outcome once."""
    import torch

    from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
        RUNTIME,
        configure_scattermoe_runtime,
    )

    fmt = detect_expert_format(cfg)
    requested = cfg.get("moe_grouped_mode") or cfg.get("dsv4_fp4_grouped_mode")
    choice = choose_grouped_mode(requested, fmt, bool(cfg.get("moe_grouped_requant")))
    cfg.moe_grouped_mode = choice.mode
    cfg.dsv4_fp4_grouped_mode = None
    configure_scattermoe_runtime(cfg)

    gpu, backend = "none", None
    if torch.cuda.is_available():
        gpu = "sm%d%d" % torch.cuda.get_device_capability()
        if choice.mode is not None:
            backend = grouped_backend(choice.mode)
            if backend is None:
                if choice.explicit:
                    raise RuntimeError(
                        f"moe_grouped_mode: {choice.mode} has no grouped backend on this GPU "
                        f"({gpu}). Set moe_grouped_mode: none or run on a supported GPU."
                    )
                LOG.warning(
                    "moe_grouped_mode auto: no %s grouped backend on %s; the experts run on the "
                    "plain scattermoe kernels.",
                    choice.mode,
                    gpu,
                )
                choice = replace(
                    choice,
                    mode=None,
                    reason=f"{choice.reason}; no {choice.mode} backend",
                )
                cfg.moe_grouped_mode = None
                RUNTIME.grouped_mode = None
    if choice.requant:
        LOG.warning(
            "moe_grouped_mode: %s requantizes the experts at load: %s.",
            choice.mode,
            choice.requant,
        )
    LOG.info(
        "moe_grouped_mode=%s (%s: %s) | gpu=%s | backend=%s | requant=%s",
        choice.mode or "none",
        "explicit" if choice.explicit else "auto",
        choice.reason,
        gpu,
        backend or "-",
        "yes" if choice.requant else "no",
    )
    return choice
