"""Template helpers from mmastrac/djev, Apache-2.0 (see DJEV_LICENSE).

Revision e5841cf41e9211608e698492658685c36e24e77a; source SHA-256
9348332d5c33391df50c321bc1f7721b1c57b587e333f7e80058aa6499378a24.
The resolver receives its tokenizer and width explicitly instead of globals.
"""

MAX_QUESTIONS = 64


MAX_SAMPLES = 32


SPAN_REGION = 24


SPAN_LIST_REGION = 64


class SchemaError(ValueError):
    pass


def parse_schema(value, *, alphabet=None, allow_multichar_labels=False):
    """Parse a DJeV schema, with an optional local label alphabet adaptation."""
    if alphabet is not None and (
        not isinstance(alphabet, (list, tuple))
        or len(alphabet) < 2
        or any(
            not isinstance(label, str)
            or not label
            or label != label.strip()
            or "\n" in label
            or "\r" in label
            or (not allow_multichar_labels and len(label) != 1)
            for label in alphabet
        )
        or len(set(alphabet)) != len(alphabet)
    ):
        detail = (
            "distinct nonempty labels without whitespace"
            if allow_multichar_labels
            else "distinct one-character labels"
        )
        raise SchemaError(f"schema: label alphabet must contain {detail}")
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("questions"), list)
        or not value["questions"]
    ):
        raise SchemaError("schema: needs a non-empty questions array")
    if len(value["questions"]) > MAX_QUESTIONS:
        raise SchemaError(f"schema: at most {MAX_QUESTIONS} questions")
    qs = []
    seen = set()
    for q in value["questions"]:
        qid = str(q.get("id", "")).strip()
        if not qid or ":" in qid or "\n" in qid:
            raise SchemaError(
                f"question id {qid!r} must be non-empty, no ':' or newline"
            )
        if qid in seen:
            raise SchemaError(f"duplicate question id {qid!r}")
        seen.add(qid)
        kind = q.get("type")
        if kind in ("noul", "bool", "boolean"):
            kind = "noul"
            crit = q.get("criteria") or {}
            choices = [("yes", crit.get("true")), ("no", crit.get("false"))]
            labels = ["yes", "no"]
        elif kind == "choice":
            opts = q.get("options") or []
            choices = [
                (o["name"], o.get("description"))
                if isinstance(o, dict)
                else (str(o), None)
                for o in opts
            ]
            labels = (
                list(alphabet[: len(choices)])
                if alphabet is not None
                else [chr(ord("A") + i) for i in range(len(choices))]
            )
        elif kind == "score":
            choices = [(str(level), None) for level in (q.get("levels") or [])]
            labels = (
                [str(i + 1) for i in range(len(choices))]
                if len(choices) <= 9
                else (
                    list(alphabet[: len(choices)])
                    if alphabet is not None
                    else [chr(ord("A") + i) for i in range(len(choices))]
                )
            )
        elif kind in ("span", "spans"):
            default = SPAN_REGION if kind == "span" else SPAN_LIST_REGION
            span_cfg = {
                "max_tokens": q.get("max_tokens", default),
                "max_items": q.get("max_items", 16),
            }
            for key, hi in (("max_tokens", 96), ("max_items", 64)):
                v = span_cfg[key]
                if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= hi:
                    raise SchemaError(f"question {qid!r}: {key} must be 1 to {hi}")
            choices, labels = [], []
        else:
            raise SchemaError(f"question {qid!r}: unknown type {kind!r}")
        if kind not in ("span", "spans") and len(choices) < 2:
            raise SchemaError(f"question {qid!r}: needs at least two alternatives")
        maximum = len(alphabet) if alphabet is not None else 26
        if len(choices) > maximum:
            raise SchemaError(f"question {qid!r}: at most {maximum} alternatives")
        deps = q.get("depends_on") or []
        ask_if = q.get("ask_if") or {}
        if not isinstance(deps, list) or not all(isinstance(d, str) for d in deps):
            raise SchemaError(
                f"question {qid!r}: depends_on must be a list of question ids"
            )
        if not isinstance(ask_if, dict) or not all(
            isinstance(v, list) and v for v in ask_if.values()
        ):
            raise SchemaError(
                f"question {qid!r}: ask_if must map a question id to a "
                "non-empty list of its answers"
            )
        if kind in ("span", "spans") and (deps or ask_if):
            raise SchemaError(
                f"question {qid!r}: a span question cannot depend on another question"
            )
        qs.append(
            {
                "id": qid,
                "type": kind,
                "instructions": str(q.get("instructions", "")),
                "choices": choices,
                "labels": labels,
                "depends_on": list(dict.fromkeys(list(deps) + list(ask_if))),
                "ask_if": ask_if,
                "alone": bool(q.get("alone", False)),
                "span": span_cfg if kind in ("span", "spans") else None,
            }
        )
    by_id = {q["id"]: q for q in qs}
    for q in qs:
        for dep in q["depends_on"]:
            if dep not in by_id or dep == q["id"]:
                raise SchemaError(
                    f"question {q['id']!r}: depends on unknown question {dep!r}"
                )
            if by_id[dep]["span"]:
                raise SchemaError(
                    f"question {q['id']!r}: cannot depend on the span question {dep!r}"
                )
        for dep, vals in q["ask_if"].items():
            names = [c[0] for c in by_id[dep]["choices"]]
            if any(v not in names for v in vals):
                raise SchemaError(
                    f"question {q['id']!r}: ask_if values for {dep!r} "
                    f"must be among {names}"
                )
    schedule(qs)  # refuses a cycle
    samples = value.get("samples", "auto")
    if samples == "auto":
        policy = {
            "mode": "auto",
            "max": max(1, min(int(value.get("auto_max", 4)), MAX_SAMPLES)),
            "threshold": float(value.get("auto_threshold", 0.1)),
        }
    elif isinstance(samples, int) and samples >= 1:
        policy = {"mode": "fixed", "n": min(samples, MAX_SAMPLES)}
    else:
        raise SchemaError('schema: samples must be a positive count or "auto"')
    ask = value.get("ask")
    if ask is not None:
        if not isinstance(ask, list) or not ask or any(a not in seen for a in ask):
            raise SchemaError("schema: ask must list question ids from this schema")
        for q in qs:
            if q["id"] in ask and any(d not in ask for d in q["depends_on"]):
                raise SchemaError(
                    f"schema: ask names {q['id']!r} but not everything it depends on"
                )
    chunk_rows = value.get("chunk_rows")
    if chunk_rows is not None and (not isinstance(chunk_rows, int) or chunk_rows < 8):
        raise SchemaError("schema: chunk_rows must be an integer of at least 8")
    chunk_prompt = value.get("chunk_prompt", "own")
    if chunk_prompt not in ("shared", "own"):
        raise SchemaError('schema: chunk_prompt must be "shared" or "own"')
    sequential = bool(value.get("sequential", False))
    think = value.get("think", 0)
    if isinstance(think, bool) or not isinstance(think, int) or not 0 <= think <= 4096:
        raise SchemaError("schema: think must be a thought budget in tokens, 0 to 4096")
    return {
        "questions": qs,
        "instructions": value.get("instructions"),
        "policy": policy,
        "steps": max(1, min(int(value.get("steps", 1)), 8)),
        "think": think,
        "ask": ask,
        "chunk_rows": chunk_rows,
        "chunk_prompt": chunk_prompt,
        "sequential": sequential,
        "format": "lines" if len([q for q in qs if not q["span"]]) <= 10 else "indexed",
    }


FORMATS = {
    "lines": (
        "\n",
        "{id}: ",
        'Reply with one line per question, in this order, formatted as "id: label".',
    ),
    "indexed": (
        " ",
        "{id}",
        "Reply on one line with each question's id immediately followed by its "
        "label, separated by single spaces.",
    ),
}


def system_text(schema, chunked=False):
    s = (
        "Answer a fixed set of questions about the state the user provides. "
        "Each question lists its allowed answers; reply with exactly one label "
        "per question.\n"
    )
    if schema.get("instructions"):
        s += "\n" + str(schema["instructions"]).strip() + "\n"
    for q in schema["questions"]:
        s += f"\nQuestion {q['id']}: {q['instructions'].strip()}\n"
        for (name, desc), label in zip(q["choices"], q["labels"], strict=False):
            if q["type"] == "noul":
                s += f"  {label}: {str(desc).strip()}\n" if desc else f"  {label}\n"
            elif desc:
                s += f"  {label}: {name} ({str(desc).strip()})\n"
            else:
                s += f"  {label}: {name}\n"
    s += "\n" + FORMATS[schema.get("format", "lines")][2]
    if chunked:
        s += (
            " A reply may cover only some of the questions; answer every line "
            "that is present."
        )
    return s


def answer_text(qs, labels, fmt="lines"):
    join, lead, _ = FORMATS[fmt]
    return join.join(
        lead.format(id=q["id"]) + q["labels"][i]
        for q, i in zip(qs, labels, strict=False)
    )


def resolve_template(qs, head, lead, fmt, *, enc, canvas_length):
    """Tokenize the answer template and find each question's slot. Every label
    must change exactly one token, at the same position for all of a question's
    labels, or this raises SchemaError. ``head`` is the token run the canvas
    starts with: the empty thought block for a plain read, and empty when the
    prompt already ends the thought channel. ``lead`` is the text before the
    first answer: the join when earlier answers are in the prompt, so the
    tokens match one joint template."""
    base_labels = [0] * len(qs)
    base = head + enc(lead + answer_text(qs, base_labels, fmt))
    if len(base) + 1 > canvas_length:
        raise SchemaError(
            f"answer template is {len(base)} tokens; the canvas holds {canvas_length - 1}"
        )
    if len(qs) == 1 and len(base) + 1 > canvas_length:
        raise SchemaError(
            f"question {qs[0]['id']!r} alone needs {len(base) + 1} canvas rows"
        )
    slots = []
    for qi, q in enumerate(qs):
        pos = None
        ids = [0] * len(q["labels"])
        for li in range(1, len(q["labels"])):
            labels = list(base_labels)
            labels[qi] = li
            e = head + enc(lead + answer_text(qs, labels, fmt))
            if len(e) != len(base):
                raise SchemaError(
                    f"question {q['id']!r}: label {q['labels'][li]!r} is not a "
                    "single token"
                )
            diffs = [i for i in range(len(e)) if e[i] != base[i]]
            if len(diffs) != 1 or (pos is not None and diffs[0] != pos):
                raise SchemaError(
                    f"question {q['id']!r}: labels do not share one template slot"
                )
            pos = diffs[0]
            ids[li] = e[pos]
        ids[0] = base[pos]
        if len(set(ids)) != len(ids):
            raise SchemaError(
                f"question {q['id']!r}: two labels tokenize to the same id"
            )
        slots.append({"pos": pos, "label_ids": ids})
    return base, slots


def schedule(qs):
    """Questions in stages: a question's stage comes after the stages of
    everything it depends on. Declaration order is kept within a stage."""
    ids = {q["id"] for q in qs}
    pending = list(qs)
    done: set = set()
    levels = []
    while pending:
        level = [
            q
            for q in pending
            if all(d in done or d not in ids for d in q["depends_on"])
        ]
        if not level:
            raise SchemaError(
                "schema: dependency cycle among " + ", ".join(q["id"] for q in pending)
            )
        levels.append(level)
        done |= {q["id"] for q in level}
        pending = [q for q in pending if q["id"] not in done]
    return levels
