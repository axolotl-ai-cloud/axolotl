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


def codebook_alphabet(codebook):
    if codebook == "vendored26":
        return None
    if codebook == "expanded52":
        return EXPANDED52
    if codebook == "expanded128":
        return EXPANDED128
    raise ValueError(f"unknown decision label codebook {codebook!r}")


def parse_decision_schema(value, *, codebook="vendored26"):
    return parse_schema(value, alphabet=codebook_alphabet(codebook))


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
    "parse_decision_schema",
]
