"""Multimodal CPT helpers (image-token autodetection + processor compat).

Only the streaming `pretraining_dataset` route is wired in v1; the
non-streaming `datasets:` route (strategy class + `load()`) is deferred to a
follow-on PR that also wires `build_collator` to route MM CPT batches outside
the `training_args.pretraining` branch.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import ProcessorMixin

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def load(*_args, **_kwargs):
    raise ValueError(
        "multimodal_pretrain is only supported via pretraining_dataset "
        "with streaming: true — see docs/multimodal.qmd"
    )


def _get_incompatible_processor_classes() -> tuple[type, ...]:
    import importlib

    classes: list[type] = []
    for mod_path, name in (
        ("transformers.models.mllama", "MllamaProcessor"),
        ("transformers.models.pixtral", "PixtralProcessor"),
        ("transformers.models.internvl", "InternVLProcessor"),
    ):
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, name, None)
            if cls is not None:
                classes.append(cls)
        except ImportError:
            continue
    return tuple(classes)


_KNOWN_IMAGE_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "[IMG]",
    "<IMG_CONTEXT>",
)

# Without masking these in labels, loss blows up ~10x on Qwen/SmolVLM.
_IMAGE_FAMILY_TOKEN_CANDIDATES: tuple[str, ...] = (
    "<image>",
    "<|image|>",
    "<|image_pad|>",
    "<image_soft_token>",
    "<start_of_image>",
    "<end_of_image>",
    "<|vision_start|>",
    "<|vision_end|>",
    "[IMG]",
    "[IMG_END]",
    "<IMG_CONTEXT>",
)

# Substrings that mark a *registered special token* as processor-inserted image
# layout (tile/grid/global markers). Used to extend the family set dynamically
# so model-specific structural tokens (e.g. SmolVLM/Idefics `<row_i_col_j>`,
# `<fake_token_around_image>`, `<global-img>`) don't leak into the CPT loss.
# Matched only against special/added tokens, so plain text and bos/eos/pad are
# never touched.
_IMAGE_STRUCTURAL_KEYWORDS: tuple[str, ...] = (
    "image",
    "img",
    "vision",
    "patch",
    "tile",
)


def _is_image_structural_token(surface: str) -> bool:
    core = surface.strip("<>| \t").lower()
    if any(kw in core for kw in _IMAGE_STRUCTURAL_KEYWORDS):
        return True
    # Tile-grid cells like `<row_2_col_3>`; require both so `<arrow_*>` etc. miss.
    if "row" in core and "col" in core:
        return True
    return False


_INCOMPATIBLE_PROCESSOR_REASONS: dict[str, str] = {
    "MllamaProcessor": (
        "Llama-3.2-Vision (Mllama) uses cross-attention image injection, not "
        "in-stream placeholder tokens. Multimodal CPT is incompatible with "
        "this architecture; use chat-template SFT instead."
    ),
    "PixtralProcessor": (
        "Pixtral's tokenizer goes through mistral_common with a different "
        "API surface than AutoProcessor. Multimodal CPT not supported in v1; "
        "use chat-template SFT or Mistral-Small-3.1."
    ),
    "InternVLProcessor": (
        "InternVL ships a custom processing pipeline (AutoProcessor returns "
        "text-only); no pixel_values are produced. Multimodal CPT not "
        "supported in v1."
    ),
}
_INCOMPATIBLE_PROCESSOR_CLASSES = _get_incompatible_processor_classes()


@dataclass
class ImageTokenSpec:
    image_token: str
    image_token_id: int
    image_family_token_ids: set[int]
    # Markers the model expects around each placeholder but the processor
    # doesn't insert (Qwen's vision_start/end); encoder wraps when set.
    marker_prefix: str | None = None
    marker_suffix: str | None = None


def build_image_token_spec(
    processor: ProcessorMixin, override: str | None = None
) -> ImageTokenSpec:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError(
            "Processor has no `tokenizer` attribute — multimodal CPT "
            "requires a processor with a text tokenizer (e.g. one produced "
            "by AutoProcessor.from_pretrained for a VLM)."
        )

    def resolve_id(tok: str) -> int | None:
        tid = tokenizer.convert_tokens_to_ids(tok)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            return None
        return tid

    known_special_tokens: set[str] = set()
    try:
        known_special_tokens |= set(tokenizer.get_added_vocab().keys())
    except Exception as exc:  # noqa: BLE001
        LOG.debug(
            "tokenizer.get_added_vocab() failed on %s: %s",
            type(tokenizer).__name__,
            exc,
        )
    known_special_tokens |= set(getattr(tokenizer, "all_special_tokens", None) or [])
    known_special_tokens |= set(
        getattr(tokenizer, "additional_special_tokens", None) or []
    )

    # special=True surfaces only — keyword-masking plain added tokens from
    # get_added_vocab() (e.g. a user-added "imagery") would -100 real text.
    special_surfaces: set[str] = set(
        getattr(tokenizer, "all_special_tokens", None) or []
    )
    special_surfaces |= set(getattr(tokenizer, "additional_special_tokens", None) or [])
    added_decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if added_decoder:
        try:
            special_surfaces |= {
                str(tok)
                for tok in added_decoder.values()
                if getattr(tok, "special", False)
            }
        except Exception as exc:  # noqa: BLE001
            LOG.debug(
                "tokenizer.added_tokens_decoder unusable on %s: %s",
                type(tokenizer).__name__,
                exc,
            )

    image_token: str | None = None
    image_token_id: int | None = None
    if override is not None:
        # Reject plain words that BPE-tokenize cleanly but aren't placeholders.
        if override not in known_special_tokens:
            raise ValueError(
                f"image_token override {override!r} is not a registered "
                f"special token on this tokenizer. Pick one of the model's "
                f"actual image tokens (e.g. '<image>', '<|image_pad|>', "
                f"'<start_of_image>'), or leave unset to autodetect."
            )
        image_token_id = resolve_id(override)
        if image_token_id is None:
            raise ValueError(
                f"image_token override {override!r} did not resolve to a "
                f"token id (unk). Remove the override to autodetect."
            )
        image_token = override
    else:
        proc_token = getattr(processor, "image_token", None)
        # Gemma-3-style only: `image_token` is the post-expansion soft token
        # (its name literally contains "soft_token"); the user-facing
        # placeholder is `boi_token`. Gemma-4 reverses this — `image_token`
        # IS the user-facing placeholder (`<|image|>`) and `boi_token`
        # (`<|image>`) is just a bracket marker, so don't blindly swap.
        boi_token = getattr(processor, "boi_token", None)
        if (
            boi_token
            and proc_token
            and boi_token != proc_token
            and boi_token in known_special_tokens
            and "soft_token" in proc_token
        ):
            proc_token = boi_token
        if proc_token is not None:
            image_token_id = resolve_id(proc_token)
            if image_token_id is not None:
                image_token = proc_token
        if image_token is None:
            for cand in _KNOWN_IMAGE_TOKEN_CANDIDATES:
                tid = resolve_id(cand)
                if tid is not None:
                    image_token = cand
                    image_token_id = tid
                    break
        if image_token is None:
            raise ValueError(
                "Could not autodetect the image placeholder token for this "
                "processor. Set `image_token: <token>` in the dataset config "
                "(e.g. '<image>' for LLaVA, '<|image_pad|>' for Qwen-VL, "
                "'<start_of_image>' for Gemma-3)."
            )

    # Filter to registered tokens so BPE-fallback ids don't get masked.
    family: set[int] = {image_token_id}  # type: ignore[arg-type]
    for cand in _IMAGE_FAMILY_TOKEN_CANDIDATES:
        if cand != image_token and cand not in known_special_tokens:
            continue
        tid = resolve_id(cand)
        if tid is not None:
            family.add(tid)
    # Mask model-specific layout tokens (tile/grid/global markers) missing
    # from the static candidate list; special=True only, so text is safe.
    for special in special_surfaces:
        if not _is_image_structural_token(special):
            continue
        tid = resolve_id(special)
        if tid is not None:
            family.add(tid)
    # Qwen's processor doesn't insert vision_start/end, but mrope needs them
    # to locate image spans — the encoder wraps bare placeholders.
    marker_prefix = marker_suffix = None
    if image_token == "<|image_pad|>":  # nosec B105
        if (
            resolve_id("<|vision_start|>") is not None
            and resolve_id("<|vision_end|>") is not None
        ):
            marker_prefix = "<|vision_start|>"
            marker_suffix = "<|vision_end|>"
    return ImageTokenSpec(
        image_token=image_token,
        image_token_id=image_token_id,  # type: ignore[arg-type]
        image_family_token_ids=family,
        marker_prefix=marker_prefix,
        marker_suffix=marker_suffix,
    )


def check_processor_compatibility(processor: ProcessorMixin) -> None:
    if _INCOMPATIBLE_PROCESSOR_CLASSES and isinstance(
        processor, _INCOMPATIBLE_PROCESSOR_CLASSES
    ):
        for cls in _INCOMPATIBLE_PROCESSOR_CLASSES:
            if isinstance(processor, cls):
                raise ValueError(
                    f"Multimodal CPT is not supported for {cls.__name__}: "
                    f"{_INCOMPATIBLE_PROCESSOR_REASONS.get(cls.__name__, '')}"
                )
    # MRO-name fallback for test fakes and unimportable concrete classes.
    for base_cls in type(processor).__mro__:
        reason = _INCOMPATIBLE_PROCESSOR_REASONS.get(base_cls.__name__)
        if reason is not None:
            raise ValueError(
                f"Multimodal CPT is not supported for {base_cls.__name__}: {reason}"
            )
