"""The ``on_rollouts_scored`` plugin hook for async GRPO.

Rank 0 collates every micro-batch, so it records the samples of each one and, once
an optimizer step has fetched its micro-batches (and scored them, with trainer-side
scorers), hands that step's prompts, completions, rewards and advantages to plugins.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from axolotl.core.trainers.grpo.async_progress import STOP_KEY, ProgressCollator

TRACE_ID = "trace_id"


class TraceRecorder:
    """Rank 0's samples of each collated micro-batch, until its step pops them."""

    def __init__(self):
        self._next_id = 0
        self._samples: dict[int, list[dict[str, Any]]] = {}

    def record(
        self, batch: dict[str, Any], samples: list[dict[str, Any]], num_processes: int
    ) -> None:
        trace_id = self._next_id
        self._next_id += 1
        self._samples[trace_id] = samples
        batch[TRACE_ID] = torch.full((num_processes,), trace_id, dtype=torch.long)

    def pop(self, batches: list[dict[str, Any]]) -> list[dict[str, Any]]:
        samples = []
        for batch in batches:
            samples.extend(self._samples.pop(int(batch[TRACE_ID][0]), []))
        return samples


@dataclass
class TraceCollator(ProgressCollator):
    """TRL's collator that records each micro-batch's samples for the trace hook."""

    recorder: TraceRecorder | None = None

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        if self.recorder is not None and STOP_KEY not in batch:
            (rows,) = examples
            self.recorder.record(
                batch,
                [sample for row in rows for sample in row.samples],
                self.num_processes,
            )
        return batch


def rollout_traces(
    tokenizer,
    samples: list[dict[str, Any]],
    scored: tuple[list[str], dict[int, tuple[np.ndarray, float]]] | None = None,
) -> tuple[list[str], list[str], dict[str, list[float]], list[float]]:
    """The hook's ``(prompts, completions, rewards, advantages)`` for ``samples``.

    Prompt and completion are the decoded tokens before and from the first trained
    one. ``scored`` holds the reward names and, per sample uid, the reward columns
    and advantage computed by trainer-side scoring; without it they come from what
    the rollout process stamped on each sample.
    """
    prompts, completions, advantages = [], [], []
    rewards: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        uid = sample.get("score_uid")
        if uid is not None and uid < 0:
            continue  # a filler row, trained on nothing
        ids = sample["input_ids"]
        mask = np.asarray(sample["completion_mask"])
        start = int(mask.argmax()) if mask.any() else len(mask)
        prompts.append(tokenizer.decode(ids[:start], skip_special_tokens=True))
        completions.append(tokenizer.decode(ids[start:], skip_special_tokens=True))
        if scored is not None and uid is not None:
            names, by_uid = scored
            columns, advantage = by_uid[uid]
            for name, value in zip(names, columns, strict=True):
                rewards[name].append(float(value))
            advantages.append(float(advantage))
        else:
            for key, value in sample["metrics"].items():
                if key.startswith("rewards/"):
                    rewards[key.removeprefix("rewards/")].append(float(value))
            advantages.append(float(sample["advantage"]))
    return prompts, completions, dict(rewards), advantages
