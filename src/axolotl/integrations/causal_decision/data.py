"""Causal-LM decision examples built from normalized ``diffusion_decision`` records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from datasets import Dataset

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.loaders import load_tokenizer

from ..diffusion_decision._util import _value
from ..diffusion_decision.adapters import normalize_record
from ..diffusion_decision.datasets import _is_eval, _rows
from ..diffusion_decision.permutation import permute_record
from ..diffusion_decision.prompts import decision_prompt_ids
from ..diffusion_decision.template import (
    parse_decision_schema,
    resolve_template,
    system_text,
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# resolve_template bounds diffusion canvases; causal prompts have no canvas.
_UNBOUNDED = 1 << 30


def letter_token_ids(tokenizer: Any, count: int) -> list[int]:
    """Token ids of the answer letters as they appear after ``"id: "``."""
    question = {"id": "answer", "labels": LETTERS[:count]}
    _, slots = resolve_template(tokenizer, [question], width=_UNBOUNDED)
    return slots[0]["label_ids"]


def soft_target(target: Mapping[str, Any], count: int) -> list[float]:
    probs = [0.0] * count
    if target["kind"] == "hard":
        probs[target["gold_idx"]] = 1.0
    elif target["kind"] == "dist":
        if target.get("other_probability"):
            raise ValueError("causal_decision targets must put all mass on options")
        indices = target.get("candidate_indices") or range(len(target["probs"]))
        for index, probability in zip(indices, target["probs"], strict=True):
            probs[index] = probability
    else:
        for index in target["allowed_set"]:
            probs[index] = 1.0 / len(target["allowed_set"])
    return probs


def decision_examples(
    tokenizer: Any, record: Mapping[str, Any], letter_ids: Sequence[int]
) -> list[dict[str, Any]]:
    """One example per question: the prompt up to its answer letter and a soft target."""
    if record.get("images"):
        raise ValueError("causal_decision does not support image records")
    examples = []
    for key, question in record["questions"].items():
        (parsed,) = parse_decision_schema({"questions": [{**question, "id": key}]})[
            "questions"
        ]
        count = len(parsed["choices"])
        if count > len(letter_ids):
            raise ValueError(f"question {key!r} has more options than num_options")
        # Yes/no and score questions are read through the same letter head.
        parsed = {**parsed, "type": "choice", "labels": list(LETTERS[:count])}
        schema = {"questions": [parsed], "instructions": record.get("instructions")}
        prompt = decision_prompt_ids(tokenizer, system_text(schema), record["state"])
        answer, slots = resolve_template(tokenizer, [parsed], width=_UNBOUNDED)
        if slots[0]["label_ids"] != list(letter_ids[:count]):
            raise ValueError(f"question {key!r} answer letters tokenize differently")
        examples.append(
            {
                "input_ids": [*prompt, *answer[: slots[0]["pos"]]],
                "decision_targets": soft_target(record["labels"][key], len(letter_ids)),
                "num_options": count,
            }
        )
    return examples


def load_causal_decision_datasets(cfg: Any) -> TrainDatasetMeta:
    tokenizer = load_tokenizer(cfg)
    decision = cfg.causal_decision
    letter_ids = letter_token_ids(tokenizer, decision.num_options)

    def build(entries: Sequence[Any] | None, training: bool) -> Dataset | None:
        examples = []
        for entry in entries or ():
            split = str(_value(entry, "split", "train")).lower()
            for row in _rows(entry):
                record = normalize_record(
                    str(_value(entry, "type")),
                    row,
                    training=training and not _is_eval(entry),
                    source_split=split,
                )
                if decision.shuffle:
                    record = permute_record(record, seed=int(cfg.seed or 0))
                examples.extend(decision_examples(tokenizer, record, letter_ids))
        return Dataset.from_list(examples) if examples else None

    return TrainDatasetMeta(
        train_dataset=build(cfg.datasets, True),
        eval_dataset=build(cfg.test_datasets, False),
    )


class CausalDecisionCollator:
    """Left-pad so every row's answer slot is the last position."""

    def __init__(self, tokenizer: Any, **_: Any):
        self.pad_token_id = tokenizer.pad_token_id or 0

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(row["input_ids"]) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_token_id)
        attention_mask = torch.zeros_like(input_ids)
        for i, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[i, width - length :] = torch.tensor(row["input_ids"])
            attention_mask[i, width - length :] = 1
        targets = torch.tensor(
            [row["decision_targets"] for row in rows], dtype=torch.float32
        )
        counts = torch.tensor([row["num_options"] for row in rows])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decision_targets": targets,
            "decision_mask": torch.arange(targets.shape[1]) < counts[:, None],
        }
