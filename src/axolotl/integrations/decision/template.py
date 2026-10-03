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
SPREADSHEET151 = tuple(
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
    "AA AB AC AD AE AF AG AH AI AK AL AM AN AO AP AR AS AT AU AV AX AZ "
    "BA BB BC BD BE BF BG BI BL BM BN BO BP BR BS BT BU BV BW BY "
    "CA CB CC CD CE CF CG CH CI CK CL CM CN CO CP CR CS CT CU CV CX CY "
    "DA DB DC DD DE DF DG DH DI DJ DK DL DM DN DO DP DR DS DT DU DX "
    "EA EB EC ED EE EF EG EK EL EM EN EP EQ ER ES ET EU EV EX "
    "FA FB FC FD FE FF FG FI FK FL FM FN FO FP FR FS FT FW FX "
    "GA GB".split()
)

# Keep the tokenizer/model control range through ID 100 out of decision labels.
RESERVED151_TOKEN_IDS = tuple(range(200, 351))
RESERVED151 = tuple(f"<SPECIAL_{token_id}>" for token_id in RESERVED151_TOKEN_IDS)


def codebook_alphabet(codebook):
    if codebook == "vendored26":
        return None
    if codebook == "expanded52":
        return EXPANDED52
    if codebook == "spreadsheet151":
        return SPREADSHEET151
    if codebook == "reserved151":
        return RESERVED151
    raise ValueError(f"unknown decision label codebook {codebook!r}")


def codebook_allows_multichar_labels(codebook):
    return codebook in {"spreadsheet151", "reserved151"}


def parse_decision_schema(value, *, codebook="vendored26"):
    """Parse a schema with the configured answer-token codebook.

    DJeV keeps ``yes``/``no`` and short score labels outside its optional
    alphabet.  The reserved-token protocol needs every finite answer position
    to use the same existing-token codebook, so rewrite those rendered labels
    after validation.  Targets remain candidate indices and are never changed.
    """
    schema = parse_schema(
        value,
        alphabet=codebook_alphabet(codebook),
        allow_multichar_labels=codebook_allows_multichar_labels(codebook),
    )
    if codebook == "reserved151":
        for question in schema["questions"]:
            if question["type"] not in {"noul", "choice", "score"}:
                continue
            question["labels"] = list(RESERVED151[: len(question["labels"])])
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
    "codebook_allows_multichar_labels",
    "parse_decision_schema",
    "SPREADSHEET151",
    "RESERVED151",
    "RESERVED151_TOKEN_IDS",
]
