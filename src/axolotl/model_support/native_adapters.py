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
    if getattr(cfg, "lora_target_parameters", None):
        raise ValueError(
            f"{model_name} native diffusion LoRA supports linear module targets only."
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


def share_diffusion_gemma_tied_lora(
    model: Any, *, validate_saved: bool = False
) -> None:
    """Share LoRA factors and merge ledger for tied DiffusionGemma text weights."""

    layers = _validate_text_layers(_lora_layers(model))
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
        canonical_name, canonical = tied_layers[0]
        if not canonical_name.startswith(
            "base_model.model.model.encoder.language_model."
        ):
            encoder = next(
                (
                    pair
                    for pair in tied_layers
                    if pair[0].startswith(
                        "base_model.model.model.encoder.language_model."
                    )
                ),
                None,
            )
            if encoder is not None:
                canonical_name, canonical = encoder
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
                    or canonical_a.weight.shape[0] != canonical_b.weight.shape[1]
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
                ] or getattr(
                    canonical.lora_dropout[adapter_name], "p", None
                ) != getattr(alias.lora_dropout[adapter_name], "p", None):
                    raise ValueError(
                        "DiffusionGemma tied projection adapters disagree on scaling or dropout."
                    )
                alias.lora_A[adapter_name] = canonical_a
                alias.lora_B[adapter_name] = canonical_b
            alias.merged_adapters = canonical.merged_adapters
