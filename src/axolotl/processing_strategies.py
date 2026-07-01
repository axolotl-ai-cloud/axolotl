"""Module containing ProcessingStrategy classes and its derivative for different MultiModal Model types"""

import bisect
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
    ):
        self.processor = processor
        self.chat_template = chat_template
        self.image_token = None
        self.image_token_id = None
        self.field_messages = self._normalize_field_messages(field_messages)

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

        # Truthiness check: empty list == unset (opt-in escape hatch), so
        # `role_boundaries: []` in YAML falls through to built-ins instead of
        # producing all-masked labels.
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

    @staticmethod
    def _normalize_field_messages(
        field_messages: str | list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        if field_messages is None:
            return ("messages",)
        if isinstance(field_messages, str):
            return (field_messages,)
        return tuple(name for name in field_messages if name)

    def _get_messages_field(self, example: dict) -> str | None:
        # Configured field wins so a stale `messages` column can't override it.
        for name in self.field_messages:
            if name in example and example[name] is not None:
                return name
        if "messages" in example and example["messages"] is not None:
            return "messages"
        return None

    @staticmethod
    def _is_legacy_schema(messages) -> bool:
        """Detect ShareGPT schema: first message has both ``from`` and ``value``."""
        return (
            isinstance(messages, list)
            and bool(messages)
            and isinstance(messages[0], dict)
            and "from" in messages[0]
            and "value" in messages[0]
        )

    def __call__(self, examples: list[dict]) -> list[dict]:
        """Normalize examples to OpenAI ``messages`` (accepts legacy ``conversations`` or custom ``field_messages``)."""
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
            messages_field = self._get_messages_field(example)
            # Re-route a custom field into the canonical key whose schema it matches;
            # the canonical "messages"/"conversations" branches below are unchanged.
            if messages_field and messages_field not in {"messages", "conversations"}:
                msgs = example[messages_field]
                target = "conversations" if self._is_legacy_schema(msgs) else "messages"
                example = dict(example)
                example[target] = msgs
                example.pop(messages_field, None)
                if target == "conversations":
                    example.pop("messages", None)

            if not ("messages" in example or "conversations" in example):
                raise ValueError(
                    "Only configured `field_messages`, `messages`, and `conversations` message keys are currently supported."
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

    def _mask_non_assistant_keep(self, input_ids: Tensor) -> Tensor:
        """Return a [B, L] bool keep tensor (True = contributes to loss)."""
        if self.train_on_inputs:
            return torch.ones_like(input_ids, dtype=torch.bool)

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
            return torch.ones_like(input_ids, dtype=torch.bool)

        # Vectorized path regresses 9-13x under multi-threaded torch.
        scanner = (
            _compute_role_keep_mask_vectorized
            if torch.get_num_threads() == 1
            else _compute_role_keep_mask
        )
        return scanner(
            input_ids,
            self.role_boundaries,
            roles_to_train=set(self.roles_to_train),
            train_on_eos=self.train_on_eos,
        )

    def _mask_non_assistant(self, labels: Tensor) -> Tensor:
        """Mask non-trainable role regions to -100 using ``self.role_boundaries``."""
        keep = self._mask_non_assistant_keep(labels)
        labels[~keep] = -100
        return labels

    def process_labels(self, input_ids: Tensor) -> Tensor:
        keep = self._mask_non_assistant_keep(input_ids)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.image_token_id is not None:
            keep = keep & (input_ids != self.image_token_id)
        labels = input_ids.clone()
        labels[~keep] = -100
        return labels


def _compute_role_keep_mask(
    labels: Tensor,
    role_boundaries: list[RoleBoundary],
    roles_to_train: set[str],
    train_on_eos: str,
) -> Tensor:
    """Return a [B, L] bool keep mask (True = contributes to loss)."""
    mask = zeros_like(labels)
    # For "last": remember each trainable turn's end-marker span so we can
    # unmask only the final one after the scan finishes.
    last_trainable_end_span: list[Optional[tuple[int, int]]] = [None] * labels.shape[0]

    # Work on a Python list per row — avoids O(n*boundaries) Tensor→list
    # conversions in the hot prefix-match loop.
    def _match_prefix(label: list[int], start_pos: int, tok_seq: list[int]) -> bool:
        if not tok_seq or start_pos + len(tok_seq) > len(label):
            return False
        return label[start_pos : start_pos + len(tok_seq)] == tok_seq

    def _find_end(
        label: list[int], start_pos: int, end_tok: list[int]
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
        label = labels[i].tolist()
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
                # Non-trainable role on train_on_eos="all": gate on include_end
                # so Pixtral / Mistral V7 Tekken shared [/INST] doesn't leak.
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

    return mask.bool()


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
    keep = _compute_role_keep_mask(
        labels, role_boundaries, roles_to_train, train_on_eos
    )
    labels[~keep] = -100
    return labels


def _compute_role_keep_mask_vectorized(
    labels: Tensor,
    role_boundaries: list[RoleBoundary],
    roles_to_train: set[str],
    train_on_eos: str,
) -> Tensor:
    """Vectorized variant of :func:`_compute_role_keep_mask`. Byte-identical output; faster under single-threaded torch."""
    if labels.numel() == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    if not role_boundaries:
        # Defensive: match reference all-mask semantics on empty boundaries.
        return torch.zeros_like(labels, dtype=torch.bool)

    B, L = labels.shape
    device = labels.device

    # Longer start_tokens first so longest-prefix-wins tie-break holds.
    indexed = list(enumerate(role_boundaries))
    indexed.sort(key=lambda ib: -len(ib[1].start_tokens))

    start_winner = torch.full((B, L), -1, dtype=torch.int64, device=device)
    # Track match length so equal-length ties keep the first writer (matches reference).
    start_winner_len = torch.zeros((B, L), dtype=torch.int64, device=device)
    end_match: list[Optional[Tensor]] = [None] * len(role_boundaries)

    for orig_idx, b in indexed:
        s_tok = b.start_tokens
        s_len = len(s_tok)
        if s_len == 0:
            continue  # defensive: empty start never matches

        if s_len > L:
            start_mask = torch.zeros((B, 0), dtype=torch.bool, device=device)
        else:
            s_tok_t = torch.tensor(s_tok, dtype=labels.dtype, device=device)
            windows = labels.unfold(1, s_len, 1)
            start_mask = (windows == s_tok_t).all(dim=-1)

        # Pad to L so absolute position j indexes into start_mask.
        pad_w = L - start_mask.shape[1]
        if pad_w > 0:
            start_mask = torch.cat(
                [
                    start_mask,
                    torch.zeros((B, pad_w), dtype=torch.bool, device=device),
                ],
                dim=1,
            )

        # Strictly-longer gate + longest-first iteration => first writer sticks.
        update = start_mask & (start_winner_len < s_len)
        start_winner = torch.where(
            update, torch.full_like(start_winner, orig_idx), start_winner
        )
        start_winner_len = torch.where(
            update, torch.full_like(start_winner_len, s_len), start_winner_len
        )

        e_tok = b.end_tokens
        e_len = len(e_tok)
        if e_len == 0:
            end_match[orig_idx] = None
        elif e_len > L:
            end_match[orig_idx] = torch.zeros((B, 0), dtype=torch.bool, device=device)
        else:
            e_tok_t = torch.tensor(e_tok, dtype=labels.dtype, device=device)
            ewindows = labels.unfold(1, e_len, 1)
            end_match[orig_idx] = (ewindows == e_tok_t).all(dim=-1)

    # Lift to CPU lists: hot per-row loop is Python, tensor access is slow.
    start_winner_cpu = start_winner.cpu().tolist()
    # Per boundary, per row: sorted end-match start positions, so the inner loop
    # bisects for the next end instead of scanning every position in the span.
    end_pos: list[Optional[list[list[int]]]] = []
    for em in end_match:
        if em is None:
            end_pos.append(None)
        else:
            nz = em.nonzero(as_tuple=False)
            rows: list[list[int]] = [[] for _ in range(B)]
            for r, c in nz.tolist():
                rows[r].append(c)
            end_pos.append(rows)

    bnd_start_len = [len(b.start_tokens) for b in role_boundaries]
    bnd_end_len = [len(b.end_tokens) for b in role_boundaries]
    bnd_include_start = [b.include_start for b in role_boundaries]
    bnd_include_end = [b.include_end for b in role_boundaries]
    bnd_role_in_loss = [b.role in roles_to_train for b in role_boundaries]

    mask = zeros_like(labels)
    ONE = b"\x01"

    for i in range(B):
        row_start = start_winner_cpu[i]
        n = L
        last_trainable_end_span: Optional[tuple[int, int]] = None
        # bytearray slice-assignment beats per-element writes; one tensor per row.
        row_mask = bytearray(n)

        j = 0
        while j < n:
            bidx = row_start[j]
            if bidx == -1:
                j += 1
                continue

            s_len = bnd_start_len[bidx]
            start_of_content = j + s_len
            e_len = bnd_end_len[bidx]
            include_start = bnd_include_start[bidx]
            include_end = bnd_include_end[bidx]
            role_in_loss = bnd_role_in_loss[bidx]

            if e_len == 0:
                end_after = n
                found_end = False
            else:
                positions = end_pos[bidx][i]  # type: ignore[index]
                limit = n - e_len
                idx = bisect.bisect_left(positions, start_of_content)
                if idx < len(positions) and positions[idx] <= limit:
                    found_end = True
                    end_after = positions[idx] + e_len
                else:
                    found_end = False
                    end_after = n

            if role_in_loss:
                if include_start:
                    row_mask[j:start_of_content] = ONE * (start_of_content - j)
                content_end = end_after - e_len if found_end else end_after
                row_mask[start_of_content:content_end] = ONE * (
                    content_end - start_of_content
                )
                if found_end and include_end and train_on_eos not in ("none", "last"):
                    row_mask[content_end:end_after] = ONE * (end_after - content_end)
                if found_end and include_end and train_on_eos == "last":
                    last_trainable_end_span = (content_end, end_after)
            else:
                if found_end and include_end and train_on_eos == "all":
                    content_end = end_after - e_len
                    row_mask[content_end:end_after] = ONE * (end_after - content_end)

            # include_end=False rewind: re-match the end as the next start.
            if found_end and not include_end and e_len:
                j = end_after - e_len
            else:
                j = end_after

        if train_on_eos == "last" and last_trainable_end_span is not None:
            s, e = last_trainable_end_span
            row_mask[s:e] = ONE * (e - s)

        # One tensor write per row — keep heavyweight op out of inner loop.
        if any(row_mask):
            mask[i] = torch.frombuffer(row_mask, dtype=torch.uint8).to(mask.dtype)

    return mask.bool()


def _apply_role_boundaries_vectorized(
    labels: Tensor,
    role_boundaries: list[RoleBoundary],
    roles_to_train: set[str],
    train_on_eos: str,
) -> Tensor:
    """Vectorized variant of :func:`_apply_role_boundaries`."""
    keep = _compute_role_keep_mask_vectorized(
        labels, role_boundaries, roles_to_train, train_on_eos
    )
    labels[~keep] = -100
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
        )
        self.video_token = "<|video_pad|>"  # nosec
        self.video_token_id = processor.tokenizer.convert_tokens_to_ids(
            self.video_token
        )

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.image_token_id is not None:
            keep = keep & (input_ids != self.image_token_id)
        if self.video_token_id is not None:
            keep = keep & (input_ids != self.video_token_id)
        labels = input_ids.clone()
        labels[~keep] = -100
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
        )
        # Real Gemma3 tokenizers expose boi_token as a direct attribute, not
        # via special_tokens_map (which only holds HF's standard slots).
        boi = getattr(processor.tokenizer, "boi_token", None)
        if boi is not None:
            self.image_token = boi
            self.image_token_id = processor.tokenizer.convert_tokens_to_ids(boi)

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.image_token_id is not None:
            keep = keep & (input_ids != self.image_token_id)
        # Resolve <image_soft_token> via tokenizer; fall back to default id
        # if not in vocab. Matches Gemma4's pattern.
        tok = self.processor.tokenizer
        soft_id = tok.convert_tokens_to_ids("<image_soft_token>")
        unk_id = getattr(tok, "unk_token_id", None)
        if soft_id is not None and soft_id != unk_id:
            keep = keep & (input_ids != soft_id)
        else:
            keep = keep & (input_ids != 262144)
        labels = input_ids.clone()
        labels[~keep] = -100
        return labels


class Gemma3nProcessingStrategy(_GemmaTurnStrategy):
    """Gemma3n: same turn boundaries as Gemma3, additionally masks audio/delimiter tokens."""

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)
        tok = self.processor.tokenizer
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.image_token_id is not None:
            keep = keep & (input_ids != self.image_token_id)
        # Follows huggingface-gemma-recipes fine_tune_gemma3n_on_t4 notebook.
        for attr in (
            "image_token_id",
            "audio_token_id",
            "boi_token_id",
            "eoi_token_id",
        ):
            tok_id = getattr(tok, attr, None)
            if tok_id is not None:
                keep = keep & (input_ids != tok_id)
        labels = input_ids.clone()
        labels[~keep] = -100
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
        keep = self._mask_non_assistant_keep(input_ids)

        tokenizer = self.processor.tokenizer
        unk_id = getattr(tokenizer, "unk_token_id", None)

        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.image_token_id is not None:
            keep = keep & (input_ids != self.image_token_id)

        if getattr(tokenizer, "image_token_id", None) is not None:
            keep = keep & (input_ids != tokenizer.image_token_id)
        if getattr(tokenizer, "audio_token_id", None) is not None:
            keep = keep & (input_ids != tokenizer.audio_token_id)

        # boi/eoi/boa/eoa are only string attrs on the processor; resolve ids here.
        for attr in ("boi_token", "eoi_token", "boa_token", "eoa_token"):
            token_str = getattr(self.processor, attr, None)
            if token_str is None:
                continue
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id == unk_id:
                continue
            keep = keep & (input_ids != token_id)

        # Video id lives on the processor, not the tokenizer.
        video_token_id = getattr(self.processor, "video_token_id", None)
        if video_token_id is not None and video_token_id != unk_id:
            keep = keep & (input_ids != video_token_id)

        labels = input_ids.clone()
        labels[~keep] = -100
        return labels


class Gemma4UnifiedProcessingStrategy(Gemma4ProcessingStrategy):
    """Processing Strategy for Gemma 4 Unified (encoder-free image/audio/video).

    The unified checkpoint shares Gemma 4's turn format and the same media
    placeholder/delimiter token set (image/audio/video, boi/eoi/boa/eoa), so
    boundary detection and label masking are inherited unchanged — both resolve
    ids dynamically from the processor/tokenizer rather than hard-coding them.
    The encoder-free raw pixel/waveform projection is handled entirely by the HF
    processor, so the strategy itself needs no audio/vision-specific logic.
    """


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


class PaddleOCRVLProcessingStrategy(ProcessingStrategy):
    """Processing Strategy class for PaddleOCR-VL."""

    def _build_role_boundaries(self) -> list[RoleBoundary]:
        tok = self.processor.tokenizer
        assistant_start = _encode_markers(tok, ["Assistant:\n"])
        eos = getattr(tok, "eos_token_id", None)
        if not assistant_start or eos is None:
            return []

        boundaries = []
        user_start = _encode_markers(tok, ["User: "])
        if user_start:
            boundaries.append(
                RoleBoundary(
                    role="user",
                    start_tokens=user_start[0],
                    end_tokens=assistant_start[0],
                    include_end=False,
                )
            )
        boundaries.append(
            RoleBoundary(
                role="assistant",
                start_tokens=assistant_start[0],
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.audio_encoder.special_ids
        )

        self.audio_token = special_ids.audio
        self.begin_audio_token = special_ids.begin_audio

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        if self.audio_token is not None:
            keep = keep & (input_ids != self.audio_token)
        if self.begin_audio_token is not None:
            keep = keep & (input_ids != self.begin_audio_token)

        labels = input_ids.clone()
        labels[~keep] = -100
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
        )
        special_ids = (
            processor.tokenizer.tokenizer.instruct_tokenizer.image_encoder.special_ids
        )

        self.image_token = special_ids.img
        self.image_break_token = special_ids.img_break
        self.image_end_token = special_ids.img_end

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)
        for tok_id in (self.image_token, self.image_break_token, self.image_end_token):
            if tok_id is not None:
                keep = keep & (input_ids != tok_id)

        labels = input_ids.clone()
        labels[~keep] = -100
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
        )

        if not hasattr(processor, "image_ids"):
            raise ValueError("'image_ids' missing from InternVL Processor.")

        self.image_token_ids = processor.image_ids

    def process_labels(self, input_ids):
        keep = self._mask_non_assistant_keep(input_ids)

        pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)

        for ids in self.image_token_ids:
            if ids is not None:
                keep = keep & (input_ids != ids)

        # Video tokens get converted to image patches during media processing; masking may be redundant.
        labels = input_ids.clone()
        labels[~keep] = -100
        return labels


class Glm4vProcessingStrategy(ProcessingStrategy):
    """Shared strategy for Glm4vProcessor (GLM-4V / GLM-4.1V) and
    Glm46VProcessor (GLM-4.6V / GLM-4.7V) — identical media-token markers.

    Role boundaries unverified; use cfg.role_boundaries to enable masking.
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
        field_messages: str | list[str] | tuple[str, ...] | None = None,
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
            field_messages=field_messages,
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
        keep = self._mask_non_assistant_keep(input_ids)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            keep = keep & (input_ids != pad_id)

        for tok_id in (
            self.image_token_id,
            self.begin_image_token_id,
            self.end_image_token_id,
            self.video_token_id,
            self.begin_video_token_id,
            self.end_video_token_id,
        ):
            if tok_id is not None:
                keep = keep & (input_ids != tok_id)

        labels = input_ids.clone()
        labels[~keep] = -100
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
    field_messages: str | list[str] | tuple[str, ...] | None = None,
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
        "field_messages": field_messages,
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
    if chat_template_type == "gemma4_unified":
        return Gemma4UnifiedProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama3_2_vision":
        return Llama3_2VisionProcessingStrategy(**processing_kwargs)
    if chat_template_type == "llama4":
        return Llama4ProcessingStrategy(**processing_kwargs)
    if chat_template_type == "pixtral":
        return PixtralProcessingStrategy(**processing_kwargs)
    if chat_template_type == "mistral_v7_tekken":
        return MistralV7TekkenProcessingStrategy(**processing_kwargs)
    if chat_template_type == "paddleocr_vl":
        return PaddleOCRVLProcessingStrategy(**processing_kwargs)

    try:
        from transformers.models.paddleocr_vl.processing_paddleocr_vl import (
            PaddleOCRVLProcessor,
        )

        if isinstance(processor, PaddleOCRVLProcessor):
            return PaddleOCRVLProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug("PaddleOCRVLProcessor import failed: %r", exc)

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

    # Both Glm4vProcessor and Glm46VProcessor share markers; route to the same
    # strategy. Independent try/except so either can be absent.
    try:
        from transformers.models.glm4v.processing_glm4v import Glm4vProcessor

        if isinstance(processor, Glm4vProcessor):
            return Glm4vProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug("Glm4vProcessor import failed: %r", exc)

    try:
        from transformers.models.glm46v.processing_glm46v import Glm46VProcessor

        if isinstance(processor, Glm46VProcessor):
            return Glm4vProcessingStrategy(**processing_kwargs)
    except (ImportError, ModuleNotFoundError) as exc:
        LOG.debug("Glm46VProcessor import failed: %r", exc)

    if isinstance(processor, InternVLProcessor):
        return InternVLProcessingStrategy(**processing_kwargs)

    # Unregistered templates (llava, lfm2vl, mistral_v3_tekken, ...) use the
    # base strategy; it warns once when train_on_inputs=False.
    return ProcessingStrategy(**processing_kwargs)
