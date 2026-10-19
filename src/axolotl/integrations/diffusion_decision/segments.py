"""Pre-tokenized canvas segments: re-ordered training draws without the tokenizer.

Every training draw but a record's first in epoch 0 re-permutes its questions and
options (``permutation.py``), so its prompt and answer template were re-tokenized
(about 20 ms a draw, mostly ``resolve_template``'s one encode per label). The
pieces a permutation only re-orders are tokenized once at preprocess instead:

    prompt = chat prefix | head | block(q) for q in order | tail | chat suffix
    head   = system preamble [+ "\\n" instructions "\\n"] + "\\n"
    block  = "Question {id}: {instructions}\\n" + option line per option, the last
             one carrying the block separator "\\n"
    option = "  {label}" + rest, rest = ": {name} ({description})\\n" (or
             ": {name}\\n", ": {criterion}\\n", "\\n")
    tail   = the answer-format instruction ("Reply ...")
    answer = canvas head ids | piece(q) for q in order, where a lines piece is
             "[\\n]{id}:" + " {label}" and an indexed piece is "[ ]{id}{label}"

``chat suffix`` (state turn and generation prompt) never changes between draws:
a prepared record takes it from its cached canvas, a reserve record stores it.
Label pieces ("  {label}" in the prompt, " {label}" in a lines answer) come from a
per-tokenizer table, so label shuffling needs no tokenizer either; an indexed
answer piece is stored per (question, label) for the question's own labels.

Why concatenation equals tokenizing the joined text
---------------------------------------------------
A fast tokenizer matches added tokens, then splits the rest into pre-tokens with
one regex and runs BPE inside each pre-token. Token spans concatenate exactly when
no pre-token and no added token crosses a junction. Every junction above is one of

  (a) "...\\n" | "Question" / "Reply"   newline, then a letter
  (b) "...\\n" | "  X"                  newline, then spaces and a non-space label
  (c) "X" | ":" / "\\n" / " "           a label letter or digit, then a colon,
                                        newline or space
  (d) "{id}:" | " X"                    a colon, then a space and a label

For the Nemotron-Labs-Diffusion (Tekken-style) split in ``_SAFE_PATTERNS``, with
alternatives A1/A2 (one optional non-letter/digit/newline char, then letters and
marks), A3 (one digit), A4 (optional space, punctuation run, then newlines or
slashes), A5 (whitespace ending in a newline), A6 (whitespace not followed by a
non-space) and A7 (whitespace):

- (c): no alternative continues past a letter or digit into a colon, newline or
  space (A1/A2 take their one non-letter char only at the start; A3 is one digit;
  A4-A7 hold no letters or digits).
- (a), (b): a pre-token holding a newline continues only with newlines/slashes
  (A4) or whitespace up to a later newline (A5); A6/A7 never start inside a
  whitespace run with a later newline because A5 matches there first. The right
  side starts with a letter, or with spaces then a non-space and no newline, so
  nothing crosses. A1/A2 cannot take a newline as their leading char.
- (d): A4 cannot take a space after its punctuation run and A1/A2 take a colon
  only as the leading char of letters, so the colon's pre-token ends at the space.
- The split has no lookbehind and A6's one-character lookahead never ends at a
  junction (its left side ends in a newline, matched by A4/A5, or in a non-space),
  so the pre-tokens on each side are the ones that side gets alone.

An added token straddling a junction would contain the junction's newline, space
or colon: tokenizers with such an added token, or one that strips neighbours
(lstrip/rstrip) or matches whole words only, get no segments. So does any other
pre-tokenizer, normalizer or model (they fall back to full tokenization), and the
prefix | head and tail | suffix junctions, which are the same in every draw, are
checked against the full prompt when segments are built. Labels must be letters
or digits and answer ids are checked as the schema requires.
"""

from __future__ import annotations

import json
import unicodedata
import zlib
from array import array
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .adapters.jsonl import normalize_jsonl
from .permutation import permute_record_with_orders, shuffle_labels
from .preprocessing import (
    _reserves_image_markers,
    _slot_plan,
    canvas_head,
    check_canvas_arguments,
    decision_schema,
    finish_decision_canvas,
)
from .prompts import serialize_state
from .records import DecisionCanvas
from .slots import SlotPlan
from .template import FORMATS, answer_text, codebook_labels, system_text

# Nemotron-Labs-Diffusion 3B/8B (identical tokenizer.json): Tekken-style split.
_NEMOTRON_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_SAFE_PATTERNS = frozenset({_NEMOTRON_PATTERN})
_ITEMSIZE = array("I").itemsize
assert _ITEMSIZE == 4

# Process-local counters: assembled draws, full-tokenization fallbacks by reason,
# and indexed answer pieces encoded at draw time.
SEGMENT_STATS: Counter = Counter()


class SegmentFallback(Exception):
    """Segments cannot reproduce this canvas; tokenize it in full instead."""


# ------------------------------------------------------------------ tokenizer
_SAFETY: dict[int, tuple[Any, str | None]] = {}


def segment_safety(tokenizer: Any) -> str | None:
    """None when this tokenizer's encodings of the segment pieces concatenate, else why not.

    A tokenizer that encodes every character on its own (test tokenizers) may set
    ``decision_segments_safe = True``.
    """
    cached = _SAFETY.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    reason = _segment_safety(tokenizer)
    _SAFETY[id(tokenizer)] = (tokenizer, reason)
    return reason


def _segment_safety(tokenizer: Any) -> str | None:
    if getattr(tokenizer, "decision_segments_safe", False) is True:
        return None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    to_str = getattr(backend, "to_str", None)
    if not callable(to_str):
        return "no fast tokenizer backend"
    try:
        spec = json.loads(to_str())
    except (TypeError, ValueError):
        return "unreadable tokenizer backend"
    if spec.get("normalizer") is not None:
        return "normalizer"
    if spec.get("truncation") is not None or spec.get("padding") is not None:
        return "truncation or padding"
    pre = spec.get("pre_tokenizer") or {}
    parts = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if (
        not isinstance(parts, list)
        or len(parts) != 2
        or parts[0].get("type") != "Split"
        or (parts[0].get("pattern") or {}).get("Regex") not in _SAFE_PATTERNS
        or parts[0].get("behavior") != "Isolated"
        or parts[0].get("invert")
        or parts[1].get("type") != "ByteLevel"
        or parts[1].get("use_regex")
        or parts[1].get("add_prefix_space")
    ):
        return "pre-tokenizer not proven safe"
    model = spec.get("model") or {}
    if model.get("type") != "BPE" or model.get("dropout") not in (None, 0, 0.0):
        return "model is not deterministic BPE"
    for token in spec.get("added_tokens") or ():
        content = str(token.get("content", ""))
        if (
            token.get("lstrip")
            or token.get("rstrip")
            or token.get("single_word")
            or any(char.isspace() or char == ":" for char in content)
        ):
            return f"added token {content!r} could straddle a segment junction"
    return None


def _encode_many(tokenizer: Any, texts: Sequence[str]) -> list[list[int]]:
    """``tokenizer.encode(text, add_special_tokens=False)`` for each text."""
    if not texts:
        return []
    if getattr(tokenizer, "backend_tokenizer", None) is not None:
        encoded = tokenizer(
            list(texts),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
        return [list(ids) for ids in encoded]
    return [list(tokenizer.encode(text, add_special_tokens=False)) for text in texts]


def _is_label(label: str) -> bool:
    return bool(label) and all(
        unicodedata.category(char)[0] in {"L", "N"} for char in label
    )


class LabelTables:
    """Token runs of every label a codebook can render, per tokenizer.

    ``prompt[label]`` encodes "  {label}" (an option line's start) and
    ``answer[label]`` " {label}" (a lines-format answer); ``slot(labels)`` caches
    the per-question template slot of a label tuple.
    """

    def __init__(self, tokenizer: Any, codebook: str) -> None:
        labels = list(
            dict.fromkeys(
                [
                    *codebook_labels(codebook),
                    *(chr(ord("A") + index) for index in range(26)),
                    *(str(index) for index in range(1, 10)),
                    "yes",
                    "no",
                ]
            )
        )
        labels = [label for label in labels if _is_label(label)]
        encoded = _encode_many(
            tokenizer,
            [f"  {label}" for label in labels] + [f" {label}" for label in labels],
        )
        self.prompt = {
            label: tuple(ids)
            for label, ids in zip(labels, encoded[: len(labels)], strict=True)
        }
        self.answer = {
            label: tuple(ids)
            for label, ids in zip(labels, encoded[len(labels) :], strict=True)
        }
        self._slots: dict[tuple[Any, ...], tuple[int, int, tuple[int, ...]] | None] = {}

    def slot(
        self, runs: Sequence[Sequence[int]], key: tuple[Any, ...]
    ) -> tuple[int, int, tuple[int, ...]]:
        """(length, slot offset, label ids) of a question's label alternatives.

        Mirrors ``resolve_template``: every alternative must be as long as the first
        and differ from it in exactly one shared position, with distinct ids.
        """
        if key in self._slots:
            cached = self._slots[key]
        else:
            cached = self._slots[key] = _slot(runs)
        if cached is None:
            raise SegmentFallback("labels do not share one template slot")
        return cached


def _slot(runs: Sequence[Sequence[int]]) -> tuple[int, int, tuple[int, ...]] | None:
    first = runs[0]
    position = None
    for run in runs[1:]:
        if len(run) != len(first):
            return None
        diffs = [index for index in range(len(run)) if run[index] != first[index]]
        if len(diffs) != 1 or (position is not None and diffs[0] != position):
            return None
        position = diffs[0]
    if position is None:
        return None
    ids = tuple(run[position] for run in runs)
    if len(set(ids)) != len(ids):
        return None
    return len(first), position, ids


_TABLES: dict[tuple[int, str], tuple[Any, LabelTables]] = {}


def label_tables(tokenizer: Any, codebook: str) -> LabelTables:
    key = (id(tokenizer), codebook)
    cached = _TABLES.get(key)
    if cached is None or cached[0] is not tokenizer:
        cached = (tokenizer, LabelTables(tokenizer, codebook))
        _TABLES[key] = cached
    return cached[1]


# ------------------------------------------------------------------- segments
@dataclass(frozen=True)
class RecordSegments:
    """One record's token spans, keyed by its normalized (source) order.

    ``tokens`` and ``bounds`` are native uint32 arrays as bytes (``tokens``
    zlib-compressed when ``compressed``, as for reserve records, whose stored chat
    suffix is most of the tokens); span ``i`` is ``tokens[bounds[i]:bounds[i + 1]]``. Spans: chat prefix, head, tail, chat suffix
    (empty unless ``suffix_stored``), then per question (``question_ids`` order)
    the header line, the lines answer pieces (first, later) and the option rests
    without and with the block separator; an indexed record stores, after the
    rests, the answer piece (first, later) of each label in ``indexed_labels``.
    """

    fmt: str
    question_ids: tuple[str, ...]
    option_counts: tuple[int, ...]
    indexed_labels: tuple[tuple[str, ...], ...]
    tokens: bytes
    bounds: bytes
    suffix_len: int
    suffix_stored: bool
    compressed: bool = False

    def arrays(self) -> tuple[array, array]:
        tokens, bounds = array("I"), array("I")
        tokens.frombytes(
            zlib.decompress(self.tokens) if self.compressed else self.tokens
        )
        bounds.frombytes(self.bounds)
        return tokens, bounds

    def without_suffix(self) -> "RecordSegments":
        """A copy for a prepared record, whose cached canvas holds the suffix."""
        if not self.suffix_stored:
            return self
        tokens, bounds = self.arrays()
        start, end = bounds[3], bounds[4]
        cut = end - start
        tokens = tokens[:start] + tokens[end:]
        bounds = array("I", [*bounds[:4], *(value - cut for value in bounds[4:])])
        return RecordSegments(
            fmt=self.fmt,
            question_ids=self.question_ids,
            option_counts=self.option_counts,
            indexed_labels=self.indexed_labels,
            tokens=tokens.tobytes(),
            bounds=bounds.tobytes(),
            suffix_len=self.suffix_len,
            suffix_stored=False,
        )


def _question_spans(fmt: str, count: int, labels: int) -> int:
    return 3 + 2 * count + (2 * labels if fmt == "indexed" else 0)


def _rest(question: Mapping[str, Any], name: str, description: Any) -> str:
    # Matches system_text: an option line is "  {label}" + this.
    if question["type"] == "noul":
        return f": {str(description).strip()}\n" if description else "\n"
    if description:
        return f": {name} ({str(description).strip()})\n"
    return f": {name}\n"


def _system_pieces(schema: Mapping[str, Any]) -> tuple[str, str]:
    head = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        head += "\n" + str(schema["instructions"]).strip() + "\n"
    return head + "\n", FORMATS[schema.get("format", "lines")][2]


def build_segments(
    tokenizer: Any,
    record: Mapping[str, Any],
    *,
    codebook: str,
    answer_format: str,
    permuted: Mapping[str, Any],
    orders: Mapping[str, Sequence[int]],
    prompt_ids: Sequence[int],
) -> RecordSegments:
    """Tokenize ``record``'s pieces; ``prompt_ids`` is the full prompt of ``permuted``.

    ``record`` is the record as given to ``_canvas_row``; ``permuted`` and
    ``orders`` are one draw of it (``permute_record_with_orders``, then any label
    shuffle), whose full prompt locates the chat prefix and suffix. Raises
    SegmentFallback when the record cannot be segmented.
    """
    reason = segment_safety(tokenizer)
    if reason is not None:
        raise SegmentFallback(reason)
    if _aliased(record):
        raise SegmentFallback("questions or targets share objects")
    if record.get("images"):
        raise SegmentFallback("image prompts are not segmented")
    normalized = normalize_jsonl(record, codebook=codebook)
    schema = decision_schema(normalized, codebook=codebook, answer_format=answer_format)
    fmt = schema["format"]
    head_text, tail_text = _system_pieces(schema)
    if not tail_text[:1].isalpha():
        raise SegmentFallback("format instruction does not start with a letter")
    texts: list[str] = [head_text, tail_text]
    question_ids, counts, indexed = [], [], []
    for question in schema["questions"]:
        choices = question["choices"]
        if len(choices) < 2 or not all(
            _is_label(label) for label in question["labels"]
        ):
            raise SegmentFallback("question without letter or digit labels")
        qid = question["id"]
        question_ids.append(qid)
        counts.append(len(choices))
        texts.append(f"Question {qid}: {question['instructions'].strip()}\n")
        if fmt == "lines":
            texts.extend((f"{qid}:", f"\n{qid}:"))
        else:
            texts.extend(("", ""))
        rests = [_rest(question, name, description) for name, description in choices]
        texts.extend(rests)
        texts.extend(rest + "\n" for rest in rests)
        labels = tuple(question["labels"]) if fmt == "indexed" else ()
        indexed.append(labels)
        for label in labels:
            texts.extend((f"{qid}{label}", f" {qid}{label}"))
    # The pieces must spell the system text (guards against template drift).
    probe = decision_schema(permuted, codebook=codebook, answer_format=answer_format)
    system = system_text(probe)
    if system != _spell(probe, head_text, tail_text):
        raise SegmentFallback("pieces do not spell the system text")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": serialize_state(permuted["state"])},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or rendered.count(system) != 1:
        raise SegmentFallback("chat template does not hold the system text once")
    texts.append(rendered[: rendered.index(system)])
    encoded = _encode_many(tokenizer, texts)
    prefix = encoded.pop()
    spans: list[Sequence[int]] = [prefix, encoded[0], encoded[1], ()]
    spans.extend(encoded[2:])
    tokens = array("I")
    bounds = array("I", [0])
    for span in spans:
        tokens.extend(span)
        bounds.append(len(tokens))
    segments = RecordSegments(
        fmt=fmt,
        question_ids=tuple(question_ids),
        option_counts=tuple(counts),
        indexed_labels=tuple(indexed),
        tokens=tokens.tobytes(),
        bounds=bounds.tobytes(),
        suffix_len=0,
        suffix_stored=False,
    )
    # Locate the suffix in the full prompt of ``permuted``.
    prompt_ids = tuple(prompt_ids)
    tables = label_tables(tokenizer, codebook)
    middle = _system_ids(
        segments, _question_starts(segments), probe, orders, tables, *segments.arrays()
    )
    if tuple(prompt_ids[: len(prefix)]) != tuple(prefix) or tuple(
        prompt_ids[len(prefix) : len(prefix) + len(middle)]
    ) != tuple(middle):
        raise SegmentFallback("prompt does not split into prefix, segments and suffix")
    suffix = prompt_ids[len(prefix) + len(middle) :]
    tokens = tokens[: bounds[3]] + array("I", suffix) + tokens[bounds[3] :]
    bounds = array("I", [*bounds[:4], *(value + len(suffix) for value in bounds[4:])])
    return RecordSegments(
        fmt=fmt,
        question_ids=tuple(question_ids),
        option_counts=tuple(counts),
        indexed_labels=tuple(indexed),
        tokens=zlib.compress(tokens.tobytes(), 1),
        bounds=bounds.tobytes(),
        suffix_len=len(suffix),
        suffix_stored=True,
        compressed=True,
    )


def _spell(schema: Mapping[str, Any], head: str, tail: str) -> str:
    text = head
    for question in schema["questions"]:
        text += f"Question {question['id']}: {question['instructions'].strip()}\n"
        lines = [
            f"  {label}" + _rest(question, name, description)
            for (name, description), label in zip(
                question["choices"], question["labels"], strict=True
            )
        ]
        text += "".join(lines) + "\n"
    return text + tail


def _system_ids(
    segments: RecordSegments,
    starts: Mapping[str, tuple[int, int]],
    schema: Mapping[str, Any],
    orders: Mapping[str, Sequence[int]],
    tables: LabelTables,
    tokens: array,
    bounds: array,
) -> array:
    """head | block per question in ``schema`` order | tail."""
    out = tokens[bounds[1] : bounds[2]]
    prompt = tables.prompt
    for question in schema["questions"]:
        found = starts.get(question["id"])
        if found is None:
            raise SegmentFallback("question not in segments")
        start, count = found
        labels = question["labels"]
        if len(labels) != count:
            raise SegmentFallback("option count changed")
        out += tokens[bounds[start] : bounds[start + 1]]
        order = orders.get(question["id"])
        if order is None:
            if question["type"] == "choice":  # e.g. a key the schema strips
                raise SegmentFallback("choice question without an option order")
            order = range(count)
        elif len(order) != count:
            raise SegmentFallback("option order does not match the options")
        rests = start + 3
        last = count - 1
        for position, option in enumerate(order):
            run = prompt.get(labels[position])
            if run is None:
                raise SegmentFallback("label not in the label table")
            out.extend(run)
            span = rests + option + (count if position == last else 0)
            out += tokens[bounds[span] : bounds[span + 1]]
    out += tokens[bounds[2] : bounds[3]]
    return out


def _question_starts(segments: RecordSegments) -> dict[str, tuple[int, int]]:
    """First span and option count of each question."""
    starts = {}
    span = 4
    for qid, count, labels in zip(
        segments.question_ids,
        segments.option_counts,
        segments.indexed_labels,
        strict=True,
    ):
        starts[qid] = (span, count)
        span += _question_spans(segments.fmt, count, len(labels))
    return starts


def _answer(
    tokenizer: Any,
    segments: RecordSegments,
    starts: Mapping[str, tuple[int, int]],
    schema: Mapping[str, Any],
    tables: LabelTables,
    tokens: array,
    bounds: array,
    head: Sequence[int],
    canvas_length: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """``resolve_template``'s (base, slots) from segments."""
    base = list(head)
    slots: list[dict[str, Any]] = []
    answer = tables.answer
    for qi, question in enumerate(schema["questions"]):
        start, count = starts[question["id"]]
        labels = tuple(question["labels"])
        if segments.fmt == "lines":
            span = start + (1 if qi == 0 else 2)
            ids = tokens[bounds[span] : bounds[span + 1]]
            try:
                runs = [answer[label] for label in labels]
            except KeyError as error:
                raise SegmentFallback("label not in the label table") from error
            length, offset, label_ids = tables.slot(runs, labels)
            slots.append(
                {"pos": len(base) + len(ids) + offset, "label_ids": list(label_ids)}
            )
            base.extend(ids)
            base.extend(runs[0])
        else:
            stored = segments.indexed_labels[
                segments.question_ids.index(question["id"])
            ]
            pieces_at = start + 3 + 2 * count
            runs = []
            for label in labels:
                if label in stored:
                    span = pieces_at + 2 * stored.index(label) + (0 if qi == 0 else 1)
                    runs.append(tuple(tokens[bounds[span] : bounds[span + 1]]))
                else:
                    SEGMENT_STATS["indexed_piece_encoded"] += 1
                    runs.append(
                        tuple(
                            tokenizer.encode(
                                ("" if qi == 0 else " ") + question["id"] + label,
                                add_special_tokens=False,
                            )
                        )
                    )
            found = _slot(runs)
            if found is None:
                raise SegmentFallback("labels do not share one template slot")
            length, offset, label_ids = found
            slots.append({"pos": len(base) + offset, "label_ids": list(label_ids)})
            base.extend(runs[0])
    if len(base) + 1 > canvas_length:
        raise SegmentFallback("answer template overflows the canvas")
    return base, slots


def assemble_canvas(
    tokenizer: Any,
    record: Mapping[str, Any],
    segments: RecordSegments,
    suffix: Sequence[int] | None,
    *,
    seed: int,
    codebook: str,
    answer_format: str,
    shuffle_prob: float,
    draw: tuple[int, int] | None,
    scaffold_ids: Sequence[int],
    turn_close_id: int,
    pad_id: int,
    vocab_size: int,
    width: int,
    steps: int,
    noise_kind: str,
    mask_token_id: int | None,
    slot_plan: SlotPlan | None,
    thought_open_ids: Sequence[int],
    thought_close_ids: Sequence[int],
) -> tuple[dict[str, Any], DecisionCanvas]:
    """(permuted record, canvas) of one training draw, as ``_canvas_row`` builds it.

    ``suffix``: the chat suffix (``segment_suffix``), or None for the stored one.
    Raises SegmentFallback where the segments cannot reproduce full tokenization.
    """
    check_canvas_arguments(
        answer_format=answer_format,
        steps=steps,
        width=width,
        vocab_size=vocab_size,
        noise_kind=noise_kind,
        mask_token_id=mask_token_id,
    )
    if _aliased(record):
        raise SegmentFallback("questions or targets share objects")
    # The permuted record (the row's "record") shares unchanged nested values
    # with ``record``; nothing downstream mutates them.
    permuted, orders = permute_record_with_orders(
        record, seed=seed, codebook=codebook, draw=draw, share=True
    )
    if shuffle_prob > 0.0:
        permuted = shuffle_labels(
            permuted, seed=seed, probability=shuffle_prob, codebook=codebook, draw=draw
        )
    normalized = dict(permuted)
    schema = decision_schema(normalized, codebook=codebook, answer_format=answer_format)
    if schema["format"] != segments.fmt:
        raise SegmentFallback("answer format changed")
    tables = label_tables(tokenizer, codebook)
    tokens, bounds = segments.arrays()
    starts = _question_starts(segments)
    prompt = tokens[bounds[0] : bounds[1]]
    prompt += _system_ids(segments, starts, schema, orders, tables, tokens, bounds)
    if suffix is None:
        if not segments.suffix_stored:
            raise SegmentFallback("prepared segments need the cached prompt")
        prompt += tokens[bounds[3] : bounds[4]]
    else:
        prompt.extend(suffix)
    plan = _slot_plan(slot_plan)
    head, pad_slots = canvas_head(
        scaffold_ids, plan, thought_open_ids, thought_close_ids
    )
    base, slots = _answer(
        tokenizer,
        segments,
        starts,
        schema,
        tables,
        tokens,
        bounds,
        head,
        width - len(pad_slots),
    )
    canvas = finish_decision_canvas(
        normalized,
        schema,
        tuple(prompt),
        base,
        slots,
        plan=plan,
        scaffold_ids=scaffold_ids,
        turn_close_id=turn_close_id,
        pad_id=pad_id,
        vocab_size=vocab_size,
        width=width,
        seed=seed,
        steps=steps,
        noise_kind=noise_kind,
        mask_token_id=mask_token_id,
        thought_open_ids=thought_open_ids,
        trusted_prompt=True,
        image_markers_reserved=_reserves_image_markers(tokenizer),
    )
    return permuted, canvas


def _aliased(record: Mapping[str, Any]) -> bool:
    """Whether two questions (or two targets) are one object.

    ``normalize_jsonl``'s deepcopy keeps such aliasing, so the full path permutes
    (and normalizes) a shared dict once per key; the shared-value permutation used
    here copies each one, so such a record keeps full tokenization. JSON-loaded
    records never alias.
    """
    for field in ("questions", "labels"):
        values = record.get(field)
        if isinstance(values, Mapping):
            ids = {id(value) for value in values.values() if isinstance(value, dict)}
            if len(ids) != sum(isinstance(value, dict) for value in values.values()):
                return True
    return False


def segment_suffix(
    segments: RecordSegments, prompt_ids: Sequence[int] | None = None
) -> Sequence[int]:
    """The chat suffix: stored, or the tail of a prepared record's cached prompt."""
    if segments.suffix_stored:
        tokens, bounds = segments.arrays()
        return tokens[bounds[3] : bounds[4]]
    if prompt_ids is None:
        raise SegmentFallback("prepared segments need the cached prompt")
    if segments.suffix_len == 0:
        return ()
    return prompt_ids[len(prompt_ids) - segments.suffix_len :]


def answer_base_ids(
    tokenizer: Any, schema: Mapping[str, Any], head: Sequence[int]
) -> list[int]:
    """``resolve_template``'s base (every question at its first label), tokenized in full."""
    questions = schema["questions"]
    return list(head) + list(
        tokenizer.encode(
            answer_text(questions, [0] * len(questions), schema["format"]),
            add_special_tokens=False,
        )
    )


__all__ = [
    "LabelTables",
    "RecordSegments",
    "SEGMENT_STATS",
    "SegmentFallback",
    "assemble_canvas",
    "answer_base_ids",
    "build_segments",
    "label_tables",
    "segment_safety",
    "segment_suffix",
]
