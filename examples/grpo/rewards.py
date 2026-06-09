"""Reward + dataset transform for examples/grpo/advantage_estimator.yaml.

Must be importable from the directory you launch `axolotl train` in
(it is, when run from the repo root).
"""

import re


def prompt_transform(cfg, *args, **kwargs):
    """gsm8k row -> chat prompt, keeping the ground-truth answer for rewards."""

    def map_fn(example, tokenizer=None):
        return {
            "prompt": [
                {
                    "role": "system",
                    "content": "Answer the question. Put the final answer in "
                    "<answer></answer> tags.",
                },
                {"role": "user", "content": example["question"]},
            ],
            "answer": example["answer"],
        }

    return map_fn, {"remove_columns": ["question"]}


def format_reward(completions, **kwargs) -> list[float]:
    """1.0 if the completion contains a well-formed <answer>...</answer> block."""
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    return [
        1.0 if pattern.search(c[0]["content"]) else 0.0 for c in completions
    ]
