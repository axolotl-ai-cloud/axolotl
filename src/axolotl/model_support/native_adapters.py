"""Native diffusion-model PEFT guards and DiffusionGemma tied-LoRA handling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from axolotl.model_support.diffusion import is_native_diffusion
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_GEMMA_TEXT_PREFIXES = (
    "base_model.model.model.encoder.language_model.",
    "base_model.model.model.decoder.",
)
_FORBIDDEN_TARGET_WORDS = ("vision", "router")
# 3D routed-expert tensors PEFT can wrap with ``ParamWrapper`` (``target_parameters``).
_GEMMA_EXPERT_PARAMETERS = ("experts.gate_up_proj", "experts.down_proj")
_NATIVE_DIFFUSION_ATTN_IMPLS = {"eager", "sdpa", "flex_attention"}


def validate_native_diffusion_lora(
    cfg: Any,
    *,
    model_name: str,
    allow_4bit: bool = False,
    allow_fsdp: bool = False,
) -> None:
    """Reject native adapter modes whose correctness has not been established."""

    if not getattr(cfg, "diffusion_lm", None):
        return
    attn_implementation = getattr(cfg, "attn_implementation", None)
    supported_attn_implementations = _NATIVE_DIFFUSION_ATTN_IMPLS | (
        {"varlen"} if model_name.startswith("Nemotron") else set()
    )
    if (
        is_native_diffusion(cfg)
        and attn_implementation is not None
        and attn_implementation not in supported_attn_implementations
    ):
        raise ValueError(
            f"{model_name} native diffusion supports attn_implementation values "
            f"{sorted(supported_attn_implementations)}; got {attn_implementation!r}."
        )
    adapter = getattr(cfg, "adapter", None)
    if adapter not in {"lora", "qlora"}:
        raise ValueError(
            f"{model_name} native diffusion training currently requires adapter: lora; "
            f"got {adapter!r}."
        )
    if getattr(cfg, "load_in_8bit", False):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support 8-bit quantization."
        )
    if not allow_4bit and (adapter == "qlora" or getattr(cfg, "load_in_4bit", False)):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support quantized adapters."
        )
    if not allow_fsdp and getattr(cfg, "fsdp_config", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support FSDP yet."
        )
    if getattr(cfg, "peft_use_dora", False):
        raise ValueError(f"{model_name} native diffusion LoRA does not support DoRA.")
    if getattr(cfg, "peft_layer_replication", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA does not support layer replication."
        )
    target_parameters = getattr(cfg, "lora_target_parameters", None) or []
    if isinstance(target_parameters, str):
        target_parameters = [target_parameters]
    if target_parameters and (
        model_name != "DiffusionGemma"
        or not all(
            str(name).endswith(_GEMMA_EXPERT_PARAMETERS) for name in target_parameters
        )
    ):
        raise ValueError(
            f"{model_name} native diffusion LoRA supports linear module targets"
            + (
                " plus routed-expert parameters "
                f"{list(_GEMMA_EXPERT_PARAMETERS)} (the router stays frozen)"
                if model_name == "DiffusionGemma"
                else ""
            )
            + f"; got lora_target_parameters={list(target_parameters)!r}."
        )
    if getattr(cfg, "lora_target_linear", False):
        raise ValueError(
            f"{model_name} native diffusion LoRA requires explicit text projection targets; "
            "lora_target_linear could select vision or router projections."
        )

    targets = getattr(cfg, "lora_target_modules", None) or []
    if not targets:
        raise ValueError(
            f"{model_name} native diffusion LoRA requires explicit text attention or "
            "dense-MLP target paths."
        )
    values = [targets] if isinstance(targets, str) else list(targets)
    names = [str(target) for target in values]
    if model_name == "DiffusionGemma" and (
        any(word in name.lower() for name in names for word in _FORBIDDEN_TARGET_WORDS)
        or not any("encoder" in name and "language_model" in name for name in names)
        or not any("decoder" in name for name in names)
    ):
        raise ValueError(
            "DiffusionGemma native LoRA targets must cover both tied encoder "
            "language-model and decoder text attention/dense-MLP projections; "
            "vision and router targets are frozen."
        )


def _lora_layers(model: Any) -> Iterable[tuple[str, Any]]:
    for name, module in model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "base_layer")
        ):
            yield name, module


def _validate_text_layers(layers: Iterable[tuple[str, Any]]) -> list[tuple[str, Any]]:
    result = list(layers)
    for name, _ in result:
        if not name.startswith(_GEMMA_TEXT_PREFIXES):
            raise ValueError(
                "DiffusionGemma native LoRA found a non-text adapter target "
                f"{name!r}; vision and router projections must remain frozen."
            )
    return result


def _is_param_wrapper(module: Any) -> bool:
    return getattr(module, "parameter_name", None) is not None


def _innermost_base(module: Any) -> Any:
    base = module
    while hasattr(base, "base_layer"):
        base = base.base_layer
    return base


def _share_adapter_factors(
    canonical_name: str,
    canonical: Any,
    tied_layers: list[tuple[str, Any]],
    *,
    validate_saved: bool,
) -> None:
    adapter_names = set(canonical.lora_A) | set(canonical.lora_B)
    if len(adapter_names) != 1:
        raise ValueError(
            "DiffusionGemma native LoRA supports exactly one adapter per tied projection."
        )
    for name, alias in tied_layers:
        if alias is canonical:
            continue
        if set(alias.lora_A) != adapter_names or set(alias.lora_B) != adapter_names:
            raise ValueError(
                "DiffusionGemma tied projection adapters disagree on adapter names: "
                f"{canonical_name!r} and {name!r}."
            )
        for adapter_name in adapter_names:
            canonical_a = canonical.lora_A[adapter_name]
            canonical_b = canonical.lora_B[adapter_name]
            alias_a = alias.lora_A[adapter_name]
            alias_b = alias.lora_B[adapter_name]
            if (
                canonical_a.weight.shape != alias_a.weight.shape
                or canonical_b.weight.shape != alias_b.weight.shape
                or canonical.r.get(adapter_name) != alias.r.get(adapter_name)
                or getattr(canonical, "use_dora", {}).get(adapter_name, False)
                or getattr(alias, "use_dora", {}).get(adapter_name, False)
            ):
                raise ValueError(
                    "DiffusionGemma tied projection adapters disagree on LoRA rank or mode."
                )
            if (
                validate_saved
                and canonical_a.weight.data_ptr() != alias_a.weight.data_ptr()
                and not canonical_a.weight.detach().equal(alias_a.weight.detach())
            ) or (
                validate_saved
                and canonical_b.weight.data_ptr() != alias_b.weight.data_ptr()
                and not canonical_b.weight.detach().equal(alias_b.weight.detach())
            ):
                raise ValueError(
                    "DiffusionGemma tied projection checkpoint contains unequal "
                    f"LoRA factors for {canonical_name!r} and {name!r}."
                )
            if canonical.scaling[adapter_name] != alias.scaling[
                adapter_name
            ] or getattr(canonical.lora_dropout[adapter_name], "p", None) != getattr(
                alias.lora_dropout[adapter_name], "p", None
            ):
                raise ValueError(
                    "DiffusionGemma tied projection adapters disagree on scaling or dropout."
                )
            alias.lora_A[adapter_name] = canonical_a
            alias.lora_B[adapter_name] = canonical_b
        alias.merged_adapters = canonical.merged_adapters


def _pick_encoder_canonical(
    tied_layers: list[tuple[str, Any]],
) -> tuple[str, Any]:
    for pair in tied_layers:
        if pair[0].startswith("base_model.model.model.encoder.language_model."):
            return pair
    return tied_layers[0]


def _share_tied_expert_lora(
    model: Any, layers: list[tuple[str, Any]], *, validate_saved: bool
) -> int:
    """Share ``ParamWrapper`` factors between tied encoder/decoder expert tensors.

    PEFT ``target_parameters`` wraps each ``experts`` module once per targeted 3D
    tensor (nested wrappers). The encoder and decoder experts hold the *same*
    ``gate_up_proj`` / ``down_proj`` Parameters, so their adapters must be one set
    of factors, exactly like the tied attention/dense-MLP linears.
    """
    wrapped: dict[tuple[int, str], list[tuple[str, Any]]] = defaultdict(list)
    for name, layer in layers:
        if not _is_param_wrapper(layer):
            continue
        parameter_name = layer.parameter_name
        param = getattr(_innermost_base(layer), parameter_name)
        # HF ties encoder/decoder experts by sharing one Parameter object.
        wrapped[(id(param), parameter_name)].append((name, layer))
    # Every experts module that holds one of these tensors must be wrapped.
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    holders: dict[tuple[int, str], set[str]] = defaultdict(set)
    for name, module in base_model.named_modules():
        if not name.startswith(("model.encoder.language_model.", "model.decoder.")):
            continue
        if hasattr(module, "base_layer"):
            continue
        for parameter_name in ("gate_up_proj", "down_proj"):
            param = module._parameters.get(parameter_name)
            if param is not None and param.dim() == 3:
                holders[(id(param), parameter_name)].add(name)
    for key, tied_layers in wrapped.items():
        if len(tied_layers) != len(holders.get(key, ())):
            raise ValueError(
                "DiffusionGemma expert LoRA must wrap every experts module sharing "
                f"{key[1]!r}; got {[n for n, _ in tied_layers]!r} for "
                f"{sorted(holders.get(key, ()))!r}."
            )
        if len(tied_layers) < 2:
            continue
        canonical_name, canonical = _pick_encoder_canonical(tied_layers)
        _share_adapter_factors(
            canonical_name, canonical, tied_layers, validate_saved=validate_saved
        )
    return sum(len(group) > 1 for group in wrapped.values())


def share_diffusion_gemma_tied_lora(
    model: Any, *, validate_saved: bool = False
) -> None:
    """Share LoRA factors and merge ledger for tied DiffusionGemma text weights."""

    all_layers = _validate_text_layers(_lora_layers(model))
    _share_tied_expert_lora(model, all_layers, validate_saved=validate_saved)
    layers = [
        (name, layer) for name, layer in all_layers if not _is_param_wrapper(layer)
    ]
    wrapped_raw_names = {name.removeprefix("base_model.model.") for name, _ in layers}
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    tied_raw: dict[int, list[str]] = defaultdict(list)
    for name, module in base_model.named_modules():
        if not name.startswith(("model.encoder.language_model.", "model.decoder.")):
            continue
        # PEFT replaces one member of a tied pair with a LoRA wrapper.  Its
        # weight lives on ``base_layer``, while an unwrapped peer exposes it
        # directly.  Inspect each logical projection once in either form.
        if name.endswith(".base_layer"):
            continue
        weight = getattr(module, "weight", None)
        if weight is None:
            weight = getattr(getattr(module, "base_layer", None), "weight", None)
        if weight is not None:
            tied_raw[weight.data_ptr()].append(name)
    for names in tied_raw.values():
        targeted = [name for name in names if name in wrapped_raw_names]
        if targeted and len(targeted) != len(names):
            raise ValueError(
                "DiffusionGemma LoRA must target every projection sharing a text "
                f"weight; got {targeted!r} from tied group {names!r}."
            )
    by_weight: dict[int, list[tuple[str, Any]]] = defaultdict(list)
    for name, layer in layers:
        weight = getattr(getattr(layer, "base_layer", None), "weight", None)
        if weight is not None:
            by_weight[weight.data_ptr()].append((name, layer))

    for tied_layers in by_weight.values():
        if len(tied_layers) < 2:
            continue
        canonical_name, canonical = _pick_encoder_canonical(tied_layers)
        _share_adapter_factors(
            canonical_name, canonical, tied_layers, validate_saved=validate_saved
        )
