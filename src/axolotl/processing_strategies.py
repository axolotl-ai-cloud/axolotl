"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch import Tensor, zeros_like
from transformers import ProcessorMixin
from transformers.image_utils import load_image
from transformers.models.internvl import InternVLProcessor
from transformers.models.smolvlm import SmolVLMProcessor
from transformers.models.voxtral import VoxtralProcessor

from axolotl.utils.dict import remove_none_values
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# One-shot warning dedupe so opt-out subclasses don't spam per-batch.
_ROLE_MASK_WARNED: set[str] = set()

# Supported values for ``train_on_eos`` — mirrors the text-only
# ChatTemplateStrategy (``turn`` = trainable turn ends only, ``all`` = every
# turn end, ``none`` = never, ``last`` = only the final trainable turn end).
_VALID_TRAIN_ON_EOS = ("turn", "all", "none", "last")


@dataclass(frozen=True)
class RoleBoundary:
    """One role's token-level span markers for the masking scanner.

    Empty ``end_tokens`` means end-of-sequence terminates the span.
    """

    role: str
    start_tokens: list[int]
    end_tokens: list[int] = field(default_factory=list)
    include_start: bool = False
    include_end: bool = True


class ProcessingStrategy:
    """Base Processing Strategy class.

    Subclasses opt in to role masking by overriding ``_build_role_boundaries``;
    otherwise only pad + media tokens are masked (legacy behavior, one-shot warned).
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None

        self.image_size = image_size
        self.image_resize_algorithm = (
            image_resize_algorithm or Image.Resampling.BILINEAR
        )

        # Defaults mirror the text-only ChatTemplateStrategy. An explicit
        # empty list is honored as "no trainable roles" (masks everything);
        # only ``None`` falls back to the default of assistant-only.
        self.train_on_inputs = bool(train_on_inputs)
        self.roles_to_train = (
            list(roles_to_train) if roles_to_train is not None else ["assistant"]
        )
        self.train_on_eos = train_on_eos if train_on_eos is not None else "turn"
        if self.train_on_eos not in _VALID_TRAIN_ON_EOS:
            raise ValueError(
                f"train_on_eos={self.train_on_eos!r} is not one of "
                f"{_VALID_TRAIN_ON_EOS}."
            )

        if hasattr(processor, "image_token"):
            self.image_token = processor.image_token
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
                self.image_token
            )

        built_in = self._build_role_boundaries()

        # Truthiness (not ``is not None``) — an empty list is treated the same
        # as an unset field: fall back to the strategy's built-in boundaries.
        # Rationale: ``role_boundaries`` is an opt-in user escape hatch for
        # unsupported / custom templates; writing ``role_boundaries: []`` in
        # YAML is almost always a typo or leftover, and honoring it literally
        # would produce all-masked labels (zero gradient). Users who truly
        # want "no role masking" should omit the field entirely.
        if role_boundaries_override:
            overridden = _resolve_role_boundary_override(
                role_boundaries_override, self.processor.tokenizer
            )
            LOG.info(
                "%s: overriding built-in role boundaries (%d decls) "
                "with cfg.role_boundaries (%d decls).",
                type(self).__name__,
                len(built_in),
                len(overridden),
            )
            self.role_boundaries: list[RoleBoundary] = overridden
            source = "override"
        else:
            self.role_boundaries = built_in
            source = "built-in"

        # Single-line, grep-friendly summary of the resolved masking config so
        # "why isn't masking firing?" is visible in training logs. For
        # overrides we include the fully resolved (role, start_ids, end_ids)
        # tuples; for built-ins we log a count (subclasses vary and logging
        # every id sequence would be noisy on, e.g., Llama3 with five roles).
        boundaries_repr: str | list[tuple[str, list[int], list[int]]]
        if source == "override":
            boundaries_repr = [
                (b.role, b.start_tokens, b.end_tokens) for b in self.role_boundaries
            ]
        else:
            boundaries_repr = f"{len(self.role_boundaries)} built-in"
        LOG.info(
            "ProcessingStrategy init: class=%s train_on_inputs=%s "
            "roles_to_train=%s train_on_eos=%s boundaries_source=%s "
            "boundaries=%s",
            type(self).__name__,
            self.train_on_inputs,
            self.roles_to_train,
            self.train_on_eos,
            source,
            boundaries_repr,
        )

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        """Subclasses declare role boundaries here; [] opts out of role masking."""
        return []

    def __call__(self, examples: list[dict]) -> list[dict]:
        """Normalize examples to OpenAI ``messages`` format (accepts legacy ``conversations``)."""
        role_mapping = {
            "human": "user",
            "gpt": "assistant",
        }

        def normalize_role(role: str) -> str:
            return role_mapping.get(role, role)

        def convert_legacy_format(example: dict) -> dict:
            messages = [
                {"role": normalize_role(convo["from"]), "content": convo["value"]}
                for convo in example["conversations"]
            ]
            result = deepcopy(example)
            result.pop("conversations")
            result["messages"] = messages
            return result

        def convert_messages_to_multimedia_messages(messages: list[dict]) -> list[dict]:
            new_messages = []
            for message in messages:
                if isinstance(message["content"], str):
                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": [
                                {
                                    "type": "text",
                                    "text": message["content"],
                                }
                            ],
                        }
                    )
                elif isinstance(message["content"], list):
                    content = message["content"]

                    new_messages.append(
                        {
                            "role": message["role"],
                            "content": content,
                        }
                    )

            return new_messages

        processed_examples = []
        for example in examples:
            if not ("messages" in example or "conversations" in example):
                raise ValueError(
                    "Only `messages` and `conversations` message keys are currently supported."
                )

            if "messages" in example and example["messages"] is not None:
                # Deepcopy for symmetry with convert_legacy_format (which
                # deepcopies internally) so downstream mutations of
                # processed_example don't leak back to the caller's input.
                processed_example = deepcopy(example)
            elif "conversations" in example:
                processed_example = convert_legacy_format(example)
            else:
                # `messages` is present but None, and no `conversations`
                # fallback exists — convert_legacy_format would KeyError on
                # ["conversations"]. Surface a clear validation error instead.
                raise ValueError(
                    "`messages` is present but None; provide non-null "
                    "`messages` or a `conversations` field."
                )

            # Required for apply_chat_template compatibility.
            processed_example["messages"] = convert_messages_to_multimedia_messages(
                processed_example["messages"]
            )

            possible_image_keys = ["images", "image"]
            image_key = None
            for key in possible_image_keys:
                if key in processed_example:
                    image_key = key
                    break

            if image_key is not None and processed_example[image_key] is not None:
                # TODO: support multi-image samples; for now we take the first.
                if len(processed_example[image_key]) > 1:
                    LOG.warning(
                        f"Found {len(processed_example[image_key])} images in a sample. Using the first one."
                        "If you are using a dataset with multiple images per sample, please convert it to use multi-content Messages."
                        "See https://docs.axolotl.ai/docs/multimodal.html#dataset-format"
                    )

                image_value = processed_example[image_key][0]

                image_value = load_image(image_value)

                if self.image_size is not None:
                    assert hasattr(image_value, "resize"), (
                        "Image does not have a resize method"
                    )

                    if isinstance(self.image_size, tuple):
                        image_value = image_value.resize(
                            self.image_size, self.image_resize_algorithm
                        )
                    else:
                        # Int image_size: preserve aspect ratio then pad to square (black) to avoid distortion.
                        padding_color = (0, 0, 0)
                        image_value = ImageOps.pad(
                            image_value,
                            (self.image_size, self.image_size),
                            method=self.image_resize_algorithm,
                            color=padding_color,
                        )

                msg_ind_to_add = None
                ind_to_add = None
                first_user_idx = None

                for msg_idx, msg_content in enumerate(processed_example["messages"]):
                    if first_user_idx is None and msg_content["role"] == "user":
                        first_user_idx = msg_idx
                    for i, content in enumerate(
                        processed_example["messages"][msg_idx]["content"]
                    ):
                        # Column-image datasets often leave a bare {type: "image"} placeholder.
                        if content["type"] == "image" and all(
                            k not in content for k in ["image", "url", "path", "base64"]
                        ):
                            msg_ind_to_add = msg_idx
                            ind_to_add = i
                            break

                if ind_to_add is not None and msg_ind_to_add is not None:
                    processed_example["messages"][msg_ind_to_add]["content"][
                        ind_to_add
                    ]["image"] = image_value
                else:
                    if first_user_idx is None:
                        first_user_idx = 0
                    processed_example["messages"][first_user_idx]["content"].append(
                        {
                            "type": "image",
                            "image": image_value,
                        }
                    )

            processed_examples.append(remove_none_values(processed_example))

        return processed_examples

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """Mask non-trainable role regions to -100 using ``self.role_boundaries``."""
        if self.train_on_inputs:
            return labels

        # Legacy no-op for boundary-less strategies; warn once so the miss shows up in logs.
        if not self.role_boundaries:
            key = type(self).__name__
            if key not in _ROLE_MASK_WARNED:
                _ROLE_MASK_WARNED.add(key)
                LOG.warning(
                    "%s has no built-in role boundaries; "
                    "cfg.train_on_inputs / cfg.roles_to_train / cfg.train_on_eos "
                    "will NOT restrict loss to assistant tokens for this "
                    "multimodal model — only pad and media tokens are masked, "
                    "every other token (system, user, assistant) contributes "
                    "to loss. To enable assistant-only masking, declare "
                    "per-role markers in YAML via cfg.role_boundaries — see "
                    "docs/multimodal_assistant_mask.md for the format and the "
                    "list of strategies on this fallback path.",
                    key,
                )
            return labels

        return _apply_role_boundaries(
            labels,
            self.role_boundaries,
            roles_to_train=set(self.roles_to_train),
            train_on_eos=self.train_on_eos,
        )

    def process_labels(self, input_ids: Tensor) -> Tensor:
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if self.image_token_id is not None:
            labels[labels == self.image_token_id] = -100
        return labels


def _apply_role_boundaries(
    labels: Tensor,
    role_boundaries: list[RoleBoundary],
    roles_to_train: set[str],
    train_on_eos: str,
) -> Tensor:
    """Mask tokens outside trainable role spans to -100.

    Scan is greedy-left with longest-prefix-wins on start_tokens to disambiguate
    nested markers (e.g. ``<|im_start|>assistant`` vs ``<|im_start|>``).
    ``train_on_eos`` accepts ``"turn"`` (end marker in loss on trainable turns
    only), ``"all"`` (always), ``"none"`` (never — overrides ``include_end``),
    ``"last"`` (only on the last trainable turn in the sequence).
    """
    mask = zeros_like(labels)
    # For "last": remember each trainable turn's end-marker span so we can
    # unmask only the final one after the scan finishes.
    last_trainable_end_span: list[Optional[tuple[int, int]]] = [None] * labels.shape[0]

    def _match_prefix(label: Tensor, start_pos: int, tok_seq: list[int]) -> bool:
        if not tok_seq or start_pos + len(tok_seq) > len(label):
            return False
        return label[start_pos : start_pos + len(tok_seq)].tolist() == tok_seq

    def _find_end(
        label: Tensor, start_pos: int, end_tok: list[int]
    ) -> tuple[int, bool]:
        # Empty end_tok means run to end-of-sequence.
        if not end_tok:
            return len(label), False
        k = start_pos
        while k < len(label):
            if _match_prefix(label, k, end_tok):
                return k + len(end_tok), True
            k += 1
        return k, False

    for i in range(labels.shape[0]):
        label = labels[i]
        j = 0
        n = len(label)
        while j < n:
            best_match: Optional[RoleBoundary] = None
            for b in role_boundaries:
                if _match_prefix(label, j, b.start_tokens):
                    if best_match is None or len(b.start_tokens) > len(
                        best_match.start_tokens
                    ):
                        best_match = b
            if best_match is None:
                j += 1
                continue

            start_of_content = j + len(best_match.start_tokens)
            end_after, found_end = _find_end(
                label, start_of_content, best_match.end_tokens
            )

            role_in_loss = best_match.role in roles_to_train

            if role_in_loss:
                if best_match.include_start:
                    mask[i][j:start_of_content] = 1
                content_end = (
                    end_after - len(best_match.end_tokens) if found_end else end_after
                )
                mask[i][start_of_content:content_end] = 1
                # train_on_eos="none"/"last" override include_end during main
                # loop; "last" is applied after the scan finishes.
                if (
                    found_end
                    and best_match.include_end
                    and train_on_eos not in ("none", "last")
                ):
                    mask[i][content_end:end_after] = 1
                if found_end and best_match.include_end and train_on_eos == "last":
                    last_trainable_end_span[i] = (content_end, end_after)
            else:
                # Non-trainable role: only the end marker can contribute, and only on train_on_eos="all".
                # Gate on include_end to mirror the trainable branch: a boundary
                # that declares include_end=False (e.g. Pixtral / Mistral V7
                # Tekken user, whose [/INST] end is shared with assistant-start)
                # must not leak its end marker into loss via the "all" path.
                if found_end and best_match.include_end and train_on_eos == "all":
                    content_end = end_after - len(best_match.end_tokens)
                    mask[i][content_end:end_after] = 1

            # When include_end=False, do not consume the end marker: back up so
            # the next iteration can re-match it as the next boundary's start
            # marker (Pixtral / Mistral V7 Tekken share [/INST] between
            # user-end and assistant-start). Requires end_tokens non-empty and
            # actually found.
            if found_end and not best_match.include_end and best_match.end_tokens:
                j = end_after - len(best_match.end_tokens)
            else:
                j = end_after

        if train_on_eos == "last" and (span := last_trainable_end_span[i]) is not None:
            s, e = span
            mask[i][s:e] = 1

        labels[i][mask[i] == 0] = -100

    return labels


def _encode_markers(tokenizer, marker_strs: list[str]) -> list[list[int]]:
    """Encode markers via ``encode(..., add_special_tokens=False)``; drops empty results."""
    result = []
    for s in marker_strs:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if toks:
            result.append(toks)
    return result


def _resolve_role_boundary_override(specs: list[dict], tokenizer) -> list[RoleBoundary]:
    """Resolve user ``cfg.role_boundaries`` specs into RoleBoundary objects.

    The sentinel ``end == "eos_token"`` resolves to ``eos_token_id`` (used by
    Pixtral/Mistral v7 templates). ``end`` null/omitted runs to end-of-sequence.
    """
    out: list[RoleBoundary] = []
    for i, spec in enumerate(specs):
        if hasattr(spec, "model_dump"):
            d = spec.model_dump()
        else:
            d = dict(spec)

        role = d.get("role")
        start_str = d.get("start")
        if not role or start_str is None:
            raise ValueError(
                f"cfg.role_boundaries[{i}] must have both 'role' and 'start' "
                f"(got {d!r})."
            )
        start_ids = tokenizer.encode(start_str, add_special_tokens=False)
        if not start_ids:
            raise ValueError(
                f"cfg.role_boundaries[{i}]: start marker {start_str!r} "
                f"tokenizes to an empty sequence; cannot match."
            )

        end_spec = d.get("end")
        if end_spec is None:
            end_ids: list[int] = []
        elif end_spec == "eos_token":
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is None:
                raise ValueError(
                    f"cfg.role_boundaries[{i}] requested end='eos_token' but "
                    "the tokenizer has no eos_token_id."
                )
            end_ids = [eos]
        else:
            end_ids = tokenizer.encode(end_spec, add_special_tokens=False)
            if not end_ids:
                raise ValueError(
                    f"cfg.role_boundaries[{i}]: end marker {end_spec!r} "
                    f"tokenizes to an empty sequence; cannot match. Use "
                    f"end=null to run to end-of-sequence or end='eos_token' "
                    f"to terminate at the tokenizer's EOS."
                )

        out.append(
            RoleBoundary(
                role=role,
                start_tokens=start_ids,
                end_tokens=end_ids,
                include_start=bool(d.get("include_start", False)),
                include_end=bool(d.get("include_end", True)),
            )
        )
    return out


class Qwen2VLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Qwen2-VL (ChatML ``<|im_start|>{role}\\n ... <|im_end|>``)."""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        self.image_token = "<|image_pad|>"  # nosec
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.image_token
        )

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|im_end|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant"):
            start = _encode_markers(tok, [f"<|im_start|>{role}\n"])
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class Qwen3_5ProcessingStrategy(Qwen2VLProcessingStrategy):
    """Processing Strategy class for Qwen3.5 (Qwen2-VL boundaries + ``<|video_pad|>`` mask)."""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        self.video_token = "<|video_pad|>"  # nosec
        self.video_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.video_token
        )

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        if self.video_token_id is not None:
            labels[labels == self.video_token_id] = -100
        return labels


class _GemmaTurnStrategy(ProcessingStrategy):
    """Gemma3/3n ``<start_of_turn>{role} ... <end_of_turn>`` (Gemma 4 uses different markers)."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<end_of_turn>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        # Template uses 'model'; external role knob stays 'assistant'. Gemma 3
        # and Gemma 3n jinja templates fold the system message into the first
        # user's content prefix and never emit '<start_of_turn>system', so we
        # don't declare a system boundary here.
        role_marker_pairs = [
            ("assistant", "model"),
            ("user", "user"),
        ]
        for external_role, template_role in role_marker_pairs:
            start = _encode_markers(tok, [f"<start_of_turn>{template_role}\n"])
            if start:
                boundaries.append(
                    RoleBoundary(
                        role=external_role,
                        start_tokens=start[0],
                        end_tokens=end_ids,
                    )
                )
        return boundaries


class Gemma3ProcessingStrategy(_GemmaTurnStrategy):
    """Processing Strategy class for Gemma3."""

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        # Gemma3 uses boi_token as the image placeholder. Real Gemma3
        # tokenizers expose it as a direct attribute (set from
        # tokenizer_config.json init_kwargs), not as a key in
        # ``special_tokens_map`` — that dict only holds HF's standard slots
        # (bos/eos/pad/unk/...). Verified against transformers
        # ``models/gemma3/processing_gemma3.py`` which reads ``tokenizer.boi_token``
        # directly.
        boi = getattr(processor.tokenizer, "boi_token", None)
        if boi is not None:
            self.image_token = boi
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(boi)

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        # Gemma3 soft image token. Resolve via tokenizer for robustness against
        # vocab shifts (custom fine-tunes, added specials, upstream retokenization).
        # Falls back to the known default id if the token isn't in vocab, so the
        # strategy still does the right thing on a stock checkpoint where the
        # string lookup returns unk. Mirrors Gemma4's convert_tokens_to_ids +
        # unk-id guard pattern.
        tok = self.processor.tokenizer
        soft_id = tok.convert_tokens_to_ids("<image_soft_token>")
        unk_id = getattr(tok, "unk_token_id", None)
        if soft_id is not None and soft_id != unk_id:
            labels[labels == soft_id] = -100
        else:
            labels[labels == 262144] = -100
        return labels


class Gemma3nProcessingStrategy(_GemmaTurnStrategy):
    """Gemma3n: same turn boundaries as Gemma3, additionally masks audio/delimiter tokens."""

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)
        tok = self.processor.tokenizer
        # Follows huggingface-gemma-recipes fine_tune_gemma3n_on_t4 notebook.
        for attr in (
            "image_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
        ):
            tok_id = getattr(tok, attr, None)
            if tok_id is not None:
                labels[labels == tok_id] = -100
        return labels


class Gemma4ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Gemma 4.

    Boundary markers ``<|turn>model ... <turn|>`` verified against
    google/gemma-4-E2B-it. boi/eoi/boa/eoa ids are resolved via
    ``convert_tokens_to_ids`` since only their string forms are on the processor.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<turn|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        role_marker_pairs = [
            ("assistant", "model"),
            ("user", "user"),
            ("system", "system"),
        ]
        for external_role, template_role in role_marker_pairs:
            # Include trailing ``\n`` for consistency with Qwen/Gemma3/Llama
            # markers; the newline is part of the marker in the real
            # google/gemma-4 tokenizer's chat template.
            start = _encode_markers(tok, [f"<|turn>{template_role}\n"])
            if start:
                boundaries.append(
                    RoleBoundary(
                        role=external_role,
                        start_tokens=start[0],
                        end_tokens=end_ids,
                    )
                )
        return boundaries

    def process_labels(self, input_ids):
        labels = super().process_labels(input_ids)

        tokenizer = self.processor.tokenizer
        unk_id = getattr(tokenizer, "unk_token_id", None)

        if getattr(tokenizer, "image_token_id", None) is not None:
            labels[labels == tokenizer.image_token_id] = -100
        if getattr(tokenizer, "audio_token_id", None) is not None:
            labels[labels == tokenizer.audio_token_id] = -100

        # boi/eoi/boa/eoa are only string attrs on the processor; resolve ids here.
        for attr in ("boi_token", "eoi_token", "boa_token", "eoa_token"):
            token_str = getattr(self.processor, attr, None)
            if token_str is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id == unk_id:
                continue
            labels[labels == token_id] = -100

        # Video id lives on the processor, not the tokenizer.
        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None and video_token_id != unk_id:
            labels[labels == video_token_id] = -100

        return labels


class Llama3_2VisionProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Llama-3.2 Vision (``<|start_header_id|>{role}<|end_header_id|>\\n\\n ... <|eot_id|>``)."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|eot_id|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant", "ipython", "tool"):
            start = _encode_markers(
                tok, [f"<|start_header_id|>{role}<|end_header_id|>\n\n"]
            )
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class Llama4ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Llama 4 (``<|header_start|>{role}<|header_end|>\\n\\n ... <|eot|>``)."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        end = _encode_markers(tok, ["<|eot|>"])
        if not end:
            return []
        end_ids = end[0]
        boundaries = []
        for role in ("system", "user", "assistant", "ipython", "tool"):
            start = _encode_markers(tok, [f"<|header_start|>{role}<|header_end|>\n\n"])
            if start:
                boundaries.append(
                    RoleBoundary(role=role, start_tokens=start[0], end_tokens=end_ids)
                )
        return boundaries


class PixtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Pixtral (``[INST] ... [/INST]`` user, assistant terminates at ``eos_token``).

    ``[/INST]`` is shared between user-end and assistant-start. We declare user
    with ``include_end=False`` so the scanner hands the ``[/INST]`` back to
    assistant's start match on the next iteration.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            return []
        boundaries = []
        inst_start = _encode_markers(tok, ["[INST]"])
        inst_end = _encode_markers(tok, ["[/INST]"])
        if inst_start and inst_end:
            boundaries.append(
                RoleBoundary(
                    role="user",
                    start_tokens=inst_start[0],
                    end_tokens=inst_end[0],
                    include_end=False,
                )
            )
            boundaries.append(
                RoleBoundary(
                    role="assistant",
                    start_tokens=inst_end[0],
                    end_tokens=[eos],
                )
            )
        return boundaries


class MistralV7TekkenProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral v7 Tekken (Pixtral-style plus ``[SYSTEM_PROMPT]...[/SYSTEM_PROMPT]``).

    Same ``[/INST]``-shared-marker treatment as :class:`PixtralProcessingStrategy`.
    """

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        eos = getattr(tok, "eos_token_id", None)
        if eos is None:
            return []
        boundaries = []
        sys_start = _encode_markers(tok, ["[SYSTEM_PROMPT]"])
        sys_end = _encode_markers(tok, ["[/SYSTEM_PROMPT]"])
        if sys_start and sys_end:
            boundaries.append(
                RoleBoundary(
                    role="system", start_tokens=sys_start[0], end_tokens=sys_end[0]
                )
            )
        inst_start = _encode_markers(tok, ["[INST]"])
        inst_end = _encode_markers(tok, ["[/INST]"])
        if inst_start and inst_end:
            boundaries.append(
                RoleBoundary(
                    role="user",
                    start_tokens=inst_start[0],
                    end_tokens=inst_end[0],
                    include_end=False,
                )
            )
            boundaries.append(
                RoleBoundary(
                    role="assistant",
                    start_tokens=inst_end[0],
                    end_tokens=[eos],
                )
            )
        return boundaries


class VoxtralProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Voxtral.

    Role boundaries NOT declared — mistral-common instruct tokenizer markers
    unverified. Falls back to pad+audio masking with a one-shot warning.
    """

    def __init__(
        self,
        processor: VoxtralProcessor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids
        )

        self.audio_token = special_ids.audio
        self.begin_audio_token = special_ids.begin_audio

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if self.audio_token is not None:
            labels[labels == self.audio_token] = -100
        if self.begin_audio_token is not None:
            labels[labels == self.begin_audio_token] = -100

        return labels


class SmolVLM2ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for SmolVLM2.

    Role boundaries NOT declared — SmolVLM2 chat_template varies per checkpoint
    (HuggingFaceTB ships multiple variants), so we opt out rather than mis-mask.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        self.image_token = "<image>"  # nosec

        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index(self.image_token)
        ]


class Mistral3ProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for Mistral3.

    Role boundaries NOT declared (mistral-common instruct tokenizer unverified);
    same fallback as VoxtralProcessingStrategy.
    """

    def __init__(
        self,
        processor,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.image_encoder.special_ids
        )

        self.image_token = special_ids.img
        self.image_break_token = special_ids.img_break
        self.image_end_token = special_ids.img_end

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        for tok_id in (self.image_token, self.image_break_token, self.image_end_token):
            if tok_id is not None:
                labels[labels == tok_id] = -100

        return labels


class InternVLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for InternVL.

    Role boundaries NOT declared (InternLM-style template unverified); falls
    back to pad + image-id masking with a one-shot warning.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )

        if not hasattr(processor, "image_ids"):
            raise ValueError("'image_ids' missing from InternVL Processor.")

        self.image_token_ids = processor.image_ids

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        for ids in self.image_token_ids:
            if ids is not None:
                labels[labels == ids] = -100

        # Video tokens get converted to image patches during media processing; masking may be redundant.
        return labels


class Glm4vProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for the GLM-4V family — covers both
    ``Glm4vProcessor`` (GLM-4V / GLM-4.1V) and ``Glm46VProcessor``
    (GLM-4.6V / GLM-4.7V). Both ship identical media-token markers
    (``<|image|>``, ``<|video|>``, ``<|begin_of_image|>``,
    ``<|end_of_image|>``, ``<|begin_of_video|>``, ``<|end_of_video|>``);
    the only upstream difference is the video-timestamp string format,
    which doesn't affect masking.

    Role boundaries NOT declared — GLM-4V role markers
    (``<|assistant|>`` / ``<|user|>``) are unverified against a real
    checkpoint. Users who need assistant-only masking should set
    ``cfg.role_boundaries`` in YAML.
    """

    def __init__(
        self,
        processor: ProcessorMixin,
        chat_template: Optional[str] = None,
        image_size: int | tuple[int, int] | None = None,
        image_resize_algorithm: Resampling | None = None,
        train_on_inputs: bool = False,
        roles_to_train: Optional[list[str]] = None,
        train_on_eos: Optional[str] = None,
        role_boundaries_override: Optional[list[dict]] = None,
    ):
        super().__init__(
            processor,
            chat_template,
            image_size,
            image_resize_algorithm,
            train_on_inputs=train_on_inputs,
            roles_to_train=roles_to_train,
            train_on_eos=train_on_eos,
            role_boundaries_override=role_boundaries_override,
        )

        self.tokenizer = getattr(processor, "tokenizer", processor)

        self.image_token = "<|image|>"  # nosec
        self.begin_image_token = "<|begin_of_image|>"  # nosec
        self.end_image_token = "<|end_of_image|>"  # nosec
        self.video_token = "<|video|>"  # nosec
        self.begin_video_token = "<|begin_of_video|>"  # nosec
        self.end_video_token = "<|end_of_video|>"  # nosec

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.begin_image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.begin_image_token
        )
        self.end_image_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_image_token
        )
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        self.begin_video_token_id = self.tokenizer.convert_tokens_to_ids(
            self.begin_video_token
        )
        self.end_video_token_id = self.tokenizer.convert_tokens_to_ids(
            self.end_video_token
        )

    def process_labels(self, input_ids):
        labels = input_ids.clone()
        labels = self._mask_non_assistant(labels)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100

        for tok_id in (
            self.image_token_id,
            self.begin_image_token_id,
            self.end_image_token_id,
            self.video_token_id,
            self.begin_video_token_id,
            self.end_video_token_id,
        ):
            if tok_id is not None:
                labels[labels == tok_id] = -100

        return labels


def get_processing_strategy(
    processor: ProcessorMixin,
    chat_template,
    chat_template_type,
    image_size: int | tuple[int, int] | None = None,
    image_resize_algorithm: Resampling | None = None,
    train_on_inputs: bool = False,
    roles_to_train: Optional[list[str]] = None,
    train_on_eos: Optional[str] = None,
    role_boundaries_override: Optional[list[dict]] = None,
):
    processing_kwargs = {
        "processor": processor,
        "chat_template": chat_template,
        "image_size": image_size,
        "image_resize_algorithm": image_resize_algorithm,
        "train_on_inputs": train_on_inputs,
        "roles_to_train": roles_to_train,
        "train_on_eos": train_on_eos,
        "role_boundaries_override": role_boundaries_override,
    }

    if chat_template_type in [None, "tokenizer_default"]:
        tokenizer = getattr(processor, "tokenizer", processor)
        if hasattr(tokenizer, "chat_template"):
            processing_kwargs["chat_template"] = tokenizer.chat_template

    if chat_template_type == "qwen2_vl":
        return Qwen2VLProcessingStrategy(**processing_kwargs)
    if chat_template_type == "qwen3_5":
        return Qwen3_5ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma3":
        return Gemma3ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma3n":
        return Gemma3nProcessingStrategy(**processing_kwargs)
    if chat_template_type == "gemma4":
        return Gemma4ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama3_2_vision":
        return Llama3_2VisionProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama4":
        return Llama4ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "pixtral":
        return PixtralProcessingStrategy(**processing_kwargs)
    if chat_template_type == "mistral_v7_tekken":
        return MistralV7TekkenProcessingStrategy(**processing_kwargs)

    if isinstance(processor, VoxtralProcessor):
        return VoxtralProcessingStrategy(**processing_kwargs)

    if isinstance(processor, SmolVLMProcessor):
        return SmolVLM2ProcessingStrategy(**processing_kwargs)

    # Lazy import: mistral_common is optional. Mirrors the Glm46V pattern below.
    try:
        from axolotl.utils.mistral.mistral3_processor import Mistral3Processor

        if isinstance(processor, Mistral3Processor):
            return Mistral3ProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug(
            "Mistral3Processor import failed; Mistral3 strategy will be unavailable: %r",
            exc,
        )

    # Register BOTH Glm4vProcessor (GLM-4V / GLM-4.1V) and Glm46VProcessor
    # (GLM-4.6V / GLM-4.7V) — they ship the same image/video markers, so one
    # strategy class covers both. Missing either registration would route a
    # genuine processor to the base fallback (pad + media-only masking with
    # a one-shot warning). Imports are independent try/except blocks so a
    # missing module on an older transformers build doesn't disable the other.
    try:
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor

        if isinstance(processor, Glm4vProcessor):
            return Glm4vProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug(
            "Glm4vProcessor import failed; Glm4v strategy will be unavailable "
            "for GLM-4V / GLM-4.1V: %r",
            exc,
        )

    try:
        from transformers.models.glm46v.processing_glm46v import Glm46VProcessor

        if isinstance(processor, Glm46VProcessor):
            return Glm4vProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug(
            "Glm46VProcessor import failed; Glm4v strategy will be unavailable "
            "for GLM-4.6V / GLM-4.7V: %r",
            exc,
        )

    if isinstance(processor, InternVLProcessor):
        return InternVLProcessingStrategy(**processing_kwargs)

    # Unregistered templates (llava, lfm2vl, mistral_v3_tekken, ...) use the
    # base strategy; it warns once when train_on_inputs=False.
    return ProcessingStrategy(**processing_kwargs)
