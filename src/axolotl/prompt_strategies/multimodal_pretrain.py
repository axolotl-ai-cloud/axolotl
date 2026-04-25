"""Multimodal CPT tokenization strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from transformers import BatchEncoding, PreTrainedTokenizerBase, ProcessorMixin

from axolotl.prompt_strategies.pretrain import (
    PretrainTokenizationStrategy,
    PretrainTokenizer,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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
        # Gemma-3-style: `image_token` is the post-expansion soft token; the
        # user-facing placeholder is `boi_token`.
        boi_token = getattr(processor, "boi_token", None)
        if (
            boi_token
            and proc_token
            and boi_token != proc_token
            and boi_token in known_special_tokens
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
    return ImageTokenSpec(
        image_token=image_token,
        image_token_id=image_token_id,  # type: ignore[arg-type]
        image_family_token_ids=family,
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


class MultimodalPretrainTokenizationStrategy(PretrainTokenizationStrategy):
    def __init__(
        self,
        *args: Any,
        image_token: str,
        image_token_id: int,
        image_column: str = "images",
        image_base_dir: str | None = None,
        image_token_spec: ImageTokenSpec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.image_token = image_token
        self.image_token_id = image_token_id
        self.image_column = image_column
        self.image_base_dir = image_base_dir
        self.image_token_spec = image_token_spec

    def _tokenize(
        self,
        prompt: str,
        add_eos_token: bool = True,
        strip_bos_token: bool = False,
    ) -> BatchEncoding:
        # No truncation: collator re-tokenizes the full text without truncation;
        # truncating here decouples the stored ids from what the model receives.
        res = self.tokenizer(prompt, add_special_tokens=True)
        ids = list(res["input_ids"])
        mask = list(res["attention_mask"])
        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if strip_bos_token and ids and bos_id is not None and ids[0] == bos_id:
            ids = ids[1:]
            mask = mask[1:]
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if add_eos_token and eos_id is not None:
            ids = ids + [eos_id]
            mask = mask + [1]
        res["input_ids"] = [ids]
        res["attention_mask"] = [mask]
        return res

    def tokenize_prompt(self, prompt: dict[str, Any]) -> dict[str, list]:
        text = prompt[self.text_column]
        raw_images = prompt.get(self.image_column)
        if raw_images is None:
            images: list = []
        elif isinstance(raw_images, (list, tuple)):
            images = list(raw_images)
        else:
            raise ValueError(
                f"Row's `{self.image_column}` must be a list of image paths, "
                f"got {type(raw_images).__name__}."
            )

        res = self._tokenize(text)
        ids = res["input_ids"][0]
        # Count by token id — `text.count` substring-matches `<image>` in `<image_soft_token>`.
        n_placeholders = sum(1 for t in ids if t == self.image_token_id)
        if n_placeholders != len(images):
            raise ValueError(
                f"Multimodal CPT row has {n_placeholders} occurrence(s) of "
                f"{self.image_token!r} in text but {len(images)} image path(s) "
                f"in `{self.image_column}`. They must match — the text column "
                f"must contain exactly one placeholder per image."
            )
        if len(ids) > self.max_length:
            raise ValueError(
                f"Multimodal CPT row tokenizes to {len(ids)} tokens which "
                f"exceeds sequence_len={self.max_length}. Pre-chunk your text "
                f"or raise sequence_len (image patch expansion at the "
                f"processor may push the final length even higher)."
            )

        # `_tokenize` produces exactly one chunk; the assert keeps that
        # invariant explicit so a future change there can't silently
        # mis-align `images` / `_mm_text` against `input_ids`.
        assert len(res["input_ids"]) == 1
        res["images"] = [list(images)]
        res["_mm_text"] = [text]
        return res


def load(
    tokenizer: PreTrainedTokenizerBase,
    cfg: Any,
    ds_cfg: dict | None = None,
    processor: ProcessorMixin | None = None,
) -> MultimodalPretrainTokenizationStrategy:
    if processor is None:
        raise ValueError(
            "multimodal_pretrain requires a processor. Set `processor_type: "
            "AutoProcessor` (or the concrete processor class) in your config "
            "so axolotl loads it at startup."
        )
    check_processor_compatibility(processor)

    ds_cfg = dict(ds_cfg or {})
    text_column = ds_cfg.get("text_column") or ds_cfg.get("field") or "text"
    image_column = ds_cfg.get("image_column") or "images"
    image_base_dir = ds_cfg.get("image_base_dir")
    image_token_override = ds_cfg.get("image_token")

    spec = build_image_token_spec(processor, override=image_token_override)
    LOG.info(
        f"multimodal_pretrain: placeholder={spec.image_token!r} "
        f"(id={spec.image_token_id}), masking {len(spec.image_family_token_ids)} "
        f"image-family token ids in labels"
    )

    strat = MultimodalPretrainTokenizationStrategy(
        PretrainTokenizer(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        text_column=text_column,
        image_column=image_column,
        image_base_dir=image_base_dir,
        image_token=spec.image_token,
        image_token_id=spec.image_token_id,
        max_length=cfg.sequence_len,
        image_token_spec=spec,
    )
    return strat
