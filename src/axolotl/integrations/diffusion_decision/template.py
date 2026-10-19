"""Tokenizer-local access to the pinned djev schema and template functions."""

from .vendored.djev_template import (
    FORMATS,
    SchemaError,
    answer_text,
    parse_schema,
    resolve_template as _resolve_template,
    schedule,
    system_text,
)

EXPANDED52 = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
# 128 one-character labels that are single tokens with distinct ids, in the answer-template context, for the
# Nemotron-Labs-Diffusion, Qwen3.5 and Gemma 3 tokenizers: ASCII letters, then Greek and Cyrillic letters, skipping
# capitals that look like Latin ones (A/Α/А, O/Ο/О, ...) and the letters whose tokens merge with the template
# (digits, ρ Ξ Ψ, ъ ы ь Ъ Ы Ь).
EXPANDED128 = tuple(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "αβγδεζηθικλμνξοπστυφχψω"
    "ΓΔΘΛΠΣΦΩ"
    "абвгдежзийклмнопрстуфхцчшщэюя"
    "БГДЖЗИЙЛПФЦЧШЩЭЮ"
)
assert len(EXPANDED128) == 128 and len(set(EXPANDED128)) == 128
# 255 labels (Jev's per-question maximum): EXPANDED128 unchanged and in order, so existing adapters and data read the
# same way, then 127 letters from non-Latin scripts. Every added label is one token with a distinct id in the answer
# template (lines format with several ids and neighbouring questions, and indexed format with digit-ending ids) for
# the Nemotron-Labs-Diffusion, Qwen3.5, Gemma 3 and DiffusionGemma (Gemma 4) tokenizers; see
# tests/integrations/diffusion_decision/test_codebook255.py. CJK ideographs fail in Qwen3.5 (" " + ideograph is two
# tokens). Excluded as look-alikes of existing labels or of each other (Unicode confusables.txt plus by eye): Hebrew
# ו ט י ס ה ח כ ר ש and final forms; voiced kana (が, グ, ...); katakana that copy hiragana or Latin shapes
# (カ キ モ エ ニ ロ ト ハ メ ナ レ ...); Thai and Devanagari letters that differ from a kept one by a stroke
# (ด ต ซ ข บ ป ผ ฟ ส ร; ध भ व ठ ढ ष ई ऊ ऐ आ ओ औ); Georgian თ ი ყ უ პ ც ვ.
EXPANDED255_ADDED = (
    "あいうえかきこしすそたつてとなにのはまみもやゆを"  # Hiragana (24)
    "アイオクコサシスチテホミラル"  # Katakana (14)
    "가나다라마바사아자차카타파하고노도로모보소오조초코토포호"  # Hangul syllables (28)
    "אבגדזלמנעפצקת"  # Hebrew (13)
    "अइउएकगघचजटडतदनपबमयरलशसह"  # Devanagari (23)
    "คจชทพมยลวหอ"  # Thai (11)
    "აბგდეკმნრსტფქშ"  # Georgian (14)
)
EXPANDED255 = EXPANDED128 + tuple(EXPANDED255_ADDED)
assert len(EXPANDED255) == 255 and len(set(EXPANDED255)) == 255
CODEBOOK_SIZES = {
    "vendored26": 26,
    "expanded52": 52,
    "expanded128": 128,
    "expanded255": 255,
}


def codebook_alphabet(codebook):
    if codebook == "vendored26":
        return None
    if codebook == "expanded52":
        return EXPANDED52
    if codebook == "expanded128":
        return EXPANDED128
    if codebook == "expanded255":
        return EXPANDED255
    raise ValueError(f"unknown decision label codebook {codebook!r}")


def codebook_labels(codebook):
    """The codebook's labels in order (A-Z for vendored26)."""
    alphabet = codebook_alphabet(codebook)
    return tuple(chr(ord("A") + i) for i in range(26)) if alphabet is None else alphabet


def parse_decision_schema(value, *, codebook="vendored26"):
    """Parse a schema; a choice question may carry ``label_indices`` (codebook positions, one per option, distinct)
    to use those labels instead of the first N."""
    questions = value.get("questions") if isinstance(value, dict) else None
    overrides = {}
    if isinstance(questions, list):
        stripped = []
        for question in questions:
            if isinstance(question, dict) and "label_indices" in question:
                question = dict(question)
                overrides[str(question.get("id", "")).strip()] = question.pop(
                    "label_indices"
                )
            stripped.append(question)
        if overrides:
            value = {**value, "questions": stripped}
    schema = parse_schema(value, alphabet=codebook_alphabet(codebook))
    if overrides:
        labels = codebook_labels(codebook)
        for question in schema["questions"]:
            if question["id"] not in overrides:
                continue
            indices = overrides[question["id"]]
            if question["type"] != "choice":
                raise SchemaError(
                    f"question {question['id']!r}: label_indices needs a choice question"
                )
            if (
                not isinstance(indices, (list, tuple))
                or len(indices) != len(question["choices"])
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, int)
                    or not 0 <= i < len(labels)
                    for i in indices
                )
                or len(set(indices)) != len(indices)
            ):
                raise SchemaError(
                    f"question {question['id']!r}: label_indices must be {len(question['choices'])} distinct "
                    f"positions in the {codebook} codebook (0 to {len(labels) - 1})"
                )
            question["labels"] = [labels[i] for i in indices]
    return schema


def resolve_template(tokenizer, qs, head=(), lead="", fmt="lines", width=128):
    return _resolve_template(
        qs,
        list(head),
        lead,
        fmt,
        enc=lambda text: tokenizer.encode(text, add_special_tokens=False),
        canvas_length=width,
    )


__all__ = [
    "FORMATS",
    "SchemaError",
    "answer_text",
    "parse_schema",
    "resolve_template",
    "schedule",
    "system_text",
    "codebook_alphabet",
    "codebook_labels",
    "CODEBOOK_SIZES",
    "EXPANDED128",
    "EXPANDED255",
    "parse_decision_schema",
]
