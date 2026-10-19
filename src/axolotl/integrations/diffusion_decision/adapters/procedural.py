from __future__ import annotations

from .common import decode_json, finite_question, label_target


def normalize(row):
    questions, answers = decode_json(row["questions"]), decode_json(row["answers"])
    out = {}
    labels = {}
    for key, q in questions.items():
        if key not in answers:
            raise ValueError(f"missing target for {key}")
        a = answers[key]
        kind = q["type"]
        criteria = q.get("criteria", {})
        if kind == "choice":
            options = list(criteria)
            descriptions = [criteria[x] for x in options]
        elif kind == "score":
            options = [str(x) for x in criteria]
            descriptions = options
        elif kind == "noul":
            options = ["yes", "no"]
            descriptions = (
                [criteria.get("true"), criteria.get("false")]
                if isinstance(criteria, dict)
                else [None, None]
            )
        else:
            raise ValueError(f"unsupported procedural type {kind}")
        if kind == "noul":
            if "noul" not in a:
                raise ValueError(f"missing noul target for {key}")
            values = [float(a["noul"]), 1 - float(a["noul"])]
        else:
            probs = a.get("probabilities")
            if not isinstance(probs, dict):
                raise ValueError(f"incomplete probabilities for {key}")
            identities = options
            if kind == "score" and a.get("legend") is not None:
                legend = a["legend"]
                if not isinstance(legend, dict) or set(legend) != set(probs):
                    raise ValueError(
                        f"score legend must identify every probability for {key}"
                    )
                inverse = {str(level): identity for identity, level in legend.items()}
                if len(inverse) != len(legend) or set(inverse) != set(options):
                    raise ValueError(
                        f"score legend must identify each level exactly once for {key}"
                    )
                identities = [inverse[level] for level in options]
            elif kind == "score" and set(probs) != set(options):
                identities = [str(index) for index in range(len(options))]
            if set(probs) != set(identities):
                raise ValueError(f"incomplete probabilities for {key}")
            values = [float(probs[identity]) for identity in identities]
        question, values = finite_question(
            kind, q.get("instructions", ""), options, values
        )
        if kind == "choice":
            question["options"] = [
                {"name": name, "description": desc}
                for name, desc in zip(options, descriptions, strict=True)
            ]
        elif kind == "noul":
            question["criteria"] = {"true": descriptions[0], "false": descriptions[1]}
        labels[key] = label_target(values, "hard" if max(values) == 1 else "dist")
        out[key] = question
    record_id = str(row["id"])
    task, separator, official_split = record_id.partition(":")
    metadata = {}
    if separator and official_split:
        metadata = {"task": task, "official_split": official_split.split(":", 1)[0]}
    return {
        "source": "procedural",
        "id": record_id,
        "group": record_id,
        "source_metadata": metadata,
        "state": decode_json(row["state"]),
        "questions": out,
        "labels": labels,
    }
