# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""TRL rows and batchers that never share tokens between samples."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from trl.experimental.async_grpo.async_grpo_trainer import (
    FixedCountBatcher,
    TokenBudgetBatcher,
)
from trl.experimental.async_grpo.packing import TreeRow

LOG = logging.getLogger(__name__)


class ChainRow(TreeRow):
    """A ``TreeRow`` whose samples never share tokens: each one is its own chain."""

    def fits(self, sample: dict[str, Any], budget: int) -> bool:
        return (
            len(self.tokens) + len(sample["input_ids"]) <= budget
            and self.trained + int(sample["completion_mask"].sum()) <= budget
        )

    def add(self, sample: dict[str, Any]) -> None:
        input_ids = sample["input_ids"].tolist()
        start = len(self.tokens)
        nodes = list(range(start, start + len(input_ids)))
        # A key no token pair can equal, so no later sample walks into this chain.
        self.roots[("sample", len(self.samples))] = start  # type: ignore[index]
        for depth, token in enumerate(input_ids):
            self.tokens.append(token)
            self.children.append({})
            self.load += depth + 1
            if depth:
                self.children[-2][(token, None)] = nodes[depth]
        self.samples.append(sample)
        self.sample_nodes.append(nodes)
        self.groups.add(sample["group_id"])
        self.trained += int(sample["completion_mask"].sum())


class ChainTokenBudgetBatcher(TokenBudgetBatcher):
    """TRL's ``TokenBudgetBatcher`` placing samples in ``ChainRow`` rows."""

    def __iter__(self) -> Iterator[list[ChainRow]]:
        rows = [ChainRow() for _ in range(self.num_processes)]
        for sample in self.dataset:
            n = len(sample["input_ids"])
            if n > self.token_budget:
                LOG.warning(
                    f"Dropping a rollout sample of {n} tokens that exceeds "
                    f"token_budget={self.token_budget}. Raise token_budget to avoid "
                    "dropping samples."
                )
                self.metrics["batch/dropped_oversize_total"].append(1.0)
                continue
            while True:
                placed_row = self._place(rows, sample)
                if placed_row is not None:
                    placed_row.add(sample)
                    break
                yield rows
                rows = [ChainRow() for _ in range(self.num_processes)]


class ChainFixedCountBatcher(FixedCountBatcher):
    """TRL's ``FixedCountBatcher`` placing samples in ``ChainRow`` rows."""

    def __iter__(self) -> Iterator[list[ChainRow]]:
        batch: list[dict[str, Any]] = []
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) < self.microbatch_size:
                continue
            groups: dict[Any, list] = {}
            for queued in batch:
                groups.setdefault(queued["group_id"], []).append(queued)
            group_list = list(groups.values())
            while len(group_list) < self.num_processes:
                largest = group_list.pop(
                    max(range(len(group_list)), key=lambda i: len(group_list[i]))
                )
                half = len(largest) // 2
                group_list += [largest[:half], largest[half:]]
            priced: list[ChainRow] = []
            for group in group_list:
                priced.append(ChainRow())
                for queued in group:
                    priced[-1].add(queued)
            rows = [ChainRow() for _ in range(self.num_processes)]
            for group in sorted(priced, key=lambda row: row.load, reverse=True):
                row = min(rows, key=lambda row: row.load)
                for queued in group.samples:
                    row.add(queued)
            yield rows
            batch = []
