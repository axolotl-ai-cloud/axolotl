"""Scoring async GRPO rollouts on the trainer.

Some rewards need the GPU: a reward model, or EBFT's feature network. TRL's rollout
process is CUDA-free, so with trainer scorers it scores only its own reward columns and
leaves each sample's advantage pending (see ``AxolotlRolloutLoop``). The trainer then
scores whole groups collectively, on every rank, before training on them:

1. ``GroupedQueue`` releases a group's samples together, once all of them arrived.
2. ``StepPlanner`` replaces TRL's batchers: it plans one optimizer step at a time
   (``gradient_accumulation_steps`` micro-batches of one row per rank) so that no
   group straddles two steps, and no row is empty.
3. ``ScoringCollator`` adds the samples' ids, their rollout-side rewards and each
   scorer's inputs to TRL's packed micro-batch.
4. ``score_step``, called from ``get_batch_samples`` on every rank, runs each scorer's
   forward on the rank's micro-batches, all-gathers the features, computes group
   rewards and advantages, and writes them into the micro-batches. Samples that
   carry no gradient (zero advantage, or truncated under
   ``mask_truncated_completions``) are taken out of the loss and its normalization,
   which is what dropping them before packing would have done.
"""

from __future__ import annotations

import itertools
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from accelerate.utils import gather_object
from tree_recurrence.integrations.trl import ChainRow

from axolotl.core.trainers.grpo.async_progress import (
    STOP_KEY,
    RolloutProgress,
)
from axolotl.core.trainers.grpo.async_rollout import (
    SCORE_PAYLOAD,
    compute_group_advantages,
)
from axolotl.core.trainers.grpo.async_traces import TraceCollator
from axolotl.core.trainers.grpo.trl_compat import TreeRow, pad
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class TrainerScorer:
    """One reward column computed on the trainer from a group's samples.

    ``group_extras`` and ``collate`` run on rank 0 while steps are planned and packed;
    ``features`` runs on every rank for every micro-batch, so it may run collective
    forwards (FSDP); ``group_rewards`` runs on every rank, on the gathered features.
    """

    name = "scorer"
    # False for models with recurrent layers: pack each sequence as its own chain.
    share_prefixes = True
    # Tokens per scoring forward (see ``chunk_sequences``); 0 runs one per row.
    token_budget = 0

    def group_extras(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sequences scored with a group but not trained, e.g. a reference answer."""
        return []

    def collate(
        self, rows: list[ScoredRow], index: int
    ) -> dict[str, tuple[list[torch.Tensor], float]]:
        """Per-row input tensors for ``features`` and their padding values.

        ``index`` is this scorer's position, which marks its entries in
        ``row.extras``. Every row must yield input for ``features``, also when it
        holds only filler samples (uid -1), so that collective forwards line up
        across ranks. Keys prefixed ``c{j}_`` split a row into chunks, each scored
        by its own ``features`` call; every row of a micro-batch must then have the
        same number of chunks, for the same reason.
        """
        raise NotImplementedError

    def features(
        self, trainer, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(uids, features)`` of the sequences in one rank's chunk of a micro-batch."""
        raise NotImplementedError

    def group_rewards(
        self, sample_features: torch.Tensor, extra_features: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Rewards of a group's samples, and metrics to average over groups."""
        raise NotImplementedError


class ScoredRow(TreeRow):
    """A ``TreeRow`` that also carries the group extras assigned to it."""

    def __init__(self):
        super().__init__()
        self.extras: list[dict[str, Any]] = []


class ScoredChainRow(ChainRow, ScoredRow):
    """A ``ScoredRow`` that packs each sample as its own chain."""


def chunk_sequences(
    sequences: list[Any], token_budget: int, share_prefixes: bool
) -> list[list[int]]:
    """Indices of ``sequences`` split into chunks that pack to ``token_budget`` tokens.

    Bounds the memory of a scoring forward, whose sequences (references, whole
    conversations) the planner's budget for trained tokens does not cover. A
    sequence longer than the budget is scored alone.
    """
    if token_budget <= 0 or not sequences:
        return [list(range(len(sequences)))]
    chunks: list[list[int]] = []
    current: list[int] = []
    for index in range(len(sequences)):
        candidate = current + [index]
        packed = pack_sequences([sequences[i] for i in candidate], share_prefixes)
        if not current or int(packed["sizes"][0]) <= token_budget:
            current = candidate
            continue
        chunks.append(current)
        current = [index]
    chunks.append(current)
    for chunk in chunks:
        if len(chunk) == 1 and len(sequences[chunk[0]]) > token_budget:
            LOG.warning_once(
                f"A scored sequence of {len(sequences[chunk[0]])} tokens exceeds the "
                f"scoring token budget of {token_budget}; it is scored in a forward "
                "of its own. Raise scoring_token_budget to silence this."
            )
    return chunks


def chunk_inputs(inputs: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """One scorer's micro-batch inputs, split into its ``c{j}_`` chunks in order."""
    chunks: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in inputs.items():
        head, _, name = key.partition("_")
        if head[:1] != "c" or not head[1:].isdigit():
            return [inputs]
        chunks.setdefault(int(head[1:]), {})[name] = value
    return [chunks[j] for j in sorted(chunks)]


def pack_sequences(
    sequences: list[np.ndarray], share_prefixes: bool = True
) -> dict[str, torch.Tensor]:
    """Tree-pack token sequences for a no-grad forward, as ``TreeRow.pack`` would.

    Besides the model inputs (``input_ids``, ``position_ids``, ``tree_*``), returns
    ``positions``: the packed position of each sequence's tokens, laid end to end, with
    ``offsets`` marking where each sequence starts, and ``sizes``: the lengths of
    ``input_ids``, ``tree_kv_index``, ``tree_cu_seqlens_q`` and ``offsets``, so a padded
    copy can be cut back.
    """
    row = TreeRow() if share_prefixes else ChainRow()
    for ids in sequences:
        row.add(
            {
                "input_ids": np.asarray(ids),
                "completion_mask": np.zeros(len(ids), dtype=np.int64),
                "group_id": None,
            }
        )
    # The layout of `TreeRow.pack`: depth-first, unary chains as one segment.
    input_ids: list[int] = []
    position_ids: list[int] = []
    offsets = [0]
    paths: list[list[int]] = []
    kv_index: list[int] = []
    cu_seqlens_k = [0]
    packed = [0] * len(row.tokens)
    stack: list[tuple[int, int | None, int]] = [
        (node, None, 0) for node in reversed(row.roots.values())
    ]
    while stack:
        node, parent, depth = stack.pop()
        segment = len(paths)
        start = len(input_ids)
        while True:
            packed[node] = len(input_ids)
            input_ids.append(row.tokens[node])
            position_ids.append(depth)
            depth += 1
            if len(row.children[node]) != 1:
                break
            node = next(iter(row.children[node].values()))
        offsets.append(len(input_ids))
        paths.append(
            (paths[parent] if parent is not None else [])
            + list(range(start, len(input_ids)))
        )
        kv_index.extend(paths[-1])
        cu_seqlens_k.append(len(kv_index))
        stack.extend(
            (child, segment, depth) for child in reversed(row.children[node].values())
        )
    positions = [packed[node] for nodes in row.sample_nodes for node in nodes]
    sequence_offsets = np.cumsum([0, *(len(nodes) for nodes in row.sample_nodes)])
    return {
        "input_ids": torch.tensor(input_ids),
        "position_ids": torch.tensor(position_ids),
        "tree_kv_index": torch.tensor(kv_index, dtype=torch.long),
        "tree_cu_seqlens_q": torch.tensor(offsets, dtype=torch.int32),
        "tree_cu_seqlens_k": torch.tensor(cu_seqlens_k, dtype=torch.int32),
        "tree_max_seqlens": torch.tensor(
            [
                max(b - a for a, b in zip(offsets[:-1], offsets[1:], strict=True)),
                max(map(len, paths)),
            ]
        ),
        "positions": torch.tensor(positions, dtype=torch.long),
        "offsets": torch.tensor(sequence_offsets, dtype=torch.long),
        "sizes": torch.tensor(
            [len(input_ids), len(kv_index), len(offsets), len(sequence_offsets)]
        ),
    }


def unpack_sequences(inputs: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Cut one rank's padded ``pack_sequences`` output back to model inputs."""
    n_tokens, n_kv, n_segments, n_offsets = inputs["sizes"][0].tolist()
    max_q, max_k = inputs["tree_max_seqlens"][0].tolist()
    offsets = inputs["offsets"][0, :n_offsets]
    return {
        "model_inputs": {
            "input_ids": inputs["input_ids"][:, :n_tokens],
            "position_ids": inputs["position_ids"][:, :n_tokens],
            "tree_kv_index": inputs["tree_kv_index"][0, :n_kv],
            "tree_cu_seqlens_q": inputs["tree_cu_seqlens_q"][0, :n_segments],
            "tree_cu_seqlens_k": inputs["tree_cu_seqlens_k"][0, :n_segments],
            "tree_max_seqlen_q": max_q,
            "tree_max_seqlen_k": max_k,
        },
        "positions": inputs["positions"][0, : int(offsets[-1])],
        "offsets": offsets,
    }


class GroupedQueue:
    """A view of the rollout queue that releases each group's samples together."""

    def __init__(self, inner):
        self._inner = inner
        self._pending: dict[int, list] = {}
        self._ready: deque = deque()

    def qsize(self) -> int:
        return self._inner.qsize() + len(self._ready)

    def get(self, timeout: float | None = None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._ready:
            remaining = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            sample = self._inner.get(timeout=remaining)
            rows = self._pending.setdefault(sample.group_id, [])
            rows.append(sample)
            if len(rows) == sample.metrics[SCORE_PAYLOAD]["group_rows"]:
                self._ready.extend(self._pending.pop(sample.group_id))
        return self._ready.popleft()


@dataclass
class _Group:
    samples: list[dict[str, Any]]
    extras: list[dict[str, Any]]


class StepPlanner(torch.utils.data.IterableDataset):
    """Plans one optimizer step at a time, keeping every group inside one step.

    A step is ``steps`` micro-batches of ``num_processes`` rows. Samples are placed
    like TRL's ``TokenBudgetBatcher`` (rows holding the group first, then the least
    loaded row they fit), within ``token_budget`` unique and trained tokens per row,
    or ``samples_per_row`` samples when ``token_budget <= 0``. When a group does not
    fit, the step is rebuilt from the groups accepted so far and emitted, and the group
    opens the next step. Empty rows, which would desync collectives, are filled by
    spreading the samples, or failing that with a copy of a sample that trains no
    token and is not scored.
    """

    def __init__(
        self,
        dataset,
        *,
        num_processes: int,
        steps: int,
        token_budget: int,
        samples_per_row: int,
        scorers: list[TrainerScorer],
        metrics: dict,
        share_prefixes: bool = True,
        stop_token_ids: frozenset[int] | None = None,
        progress: RolloutProgress | None = None,
    ):
        self.dataset = dataset
        self.num_processes = num_processes
        self.steps = steps
        self.token_budget = token_budget
        self.samples_per_row = samples_per_row
        self.scorers = scorers
        self.metrics = metrics
        self.row_cls = ScoredRow if share_prefixes else ScoredChainRow
        self.stop_token_ids = stop_token_ids
        self.progress = progress
        self.reward_names: list[str] | None = None
        self._uids = itertools.count()

    def __iter__(self) -> Iterator[list[ScoredRow]]:
        source = iter(self.dataset)
        accepted: list[_Group] = []
        rows = self._new_rows()
        pending = None
        while True:
            group = pending if pending is not None else self._read_group(source)
            pending = None
            if self._place(rows, group):
                accepted.append(group)
                continue
            if not accepted:
                LOG.warning(
                    f"Dropping a group of {len(group.samples)} samples that does not "
                    "fit in one optimizer step; raise token_budget or "
                    "gradient_accumulation_steps."
                )
                self.metrics["batch/dropped_oversize_group_total"].append(1.0)
                rows = self._new_rows()
                if self.progress is not None:
                    self.progress.drop(group.samples[0]["group_id"], "oversize")
                continue
            yield from self._emit(accepted)
            accepted, rows, pending = [], self._new_rows(), group

    def _new_rows(self) -> list[ScoredRow]:
        return [self.row_cls() for _ in range(self.steps * self.num_processes)]

    def _read_group(self, source) -> _Group:
        # A group is read whole before anything is yielded, so the staleness check
        # upstream sees one policy version for all of its samples.
        first = next(source)
        samples = [first]
        group_rows = first["metrics"][SCORE_PAYLOAD]["group_rows"]
        while len(samples) < group_rows:
            samples.append(next(source))
        for sample in samples:
            if sample["group_id"] != first["group_id"]:
                raise RuntimeError("Rollout samples arrived out of their groups")
            payload = sample["metrics"].pop(SCORE_PAYLOAD)
            sample["score_payload"] = payload
            sample["score_uid"] = next(self._uids)
            sample["score_truncated"] = self.stop_token_ids is not None and (
                len(sample["input_ids"]) == 0
                or int(sample["input_ids"][-1]) not in self.stop_token_ids
            )
        if self.reward_names is None:
            self.reward_names = list(samples[0]["score_payload"]["reward_names"])
        extras = []
        for index, scorer in enumerate(self.scorers):
            for extra in scorer.group_extras(samples):
                extras.append(
                    {
                        **extra,
                        "scorer": index,
                        "group_id": first["group_id"],
                        "uid": next(self._uids),
                    }
                )
        return _Group(samples, extras)

    def _fits(self, row: ScoredRow, sample: dict[str, Any]) -> bool:
        if self.token_budget > 0:
            return row.fits(sample, self.token_budget)
        return len(row.samples) < self.samples_per_row

    def _choose(
        self, rows: list[ScoredRow], sample: dict[str, Any], spread: bool
    ) -> ScoredRow | None:
        candidates = [row for row in rows if self._fits(row, sample)]
        if spread:
            empty = [row for row in candidates if not row.samples]
            if empty:
                return empty[0]
        home = [row for row in candidates if sample["group_id"] in row.groups]
        return min(home or candidates, key=lambda row: row.load, default=None)

    def _place(self, rows: list[ScoredRow], group: _Group, spread=False) -> bool:
        placed = []
        for sample in group.samples:
            row = self._choose(rows, sample, spread)
            if row is None:
                return False
            row.add(sample)
            placed.append(row)
        placed[0].extras.extend(group.extras)
        return True

    def _build(self, groups: list[_Group], spread: bool) -> list[ScoredRow] | None:
        rows = self._new_rows()
        for group in groups:
            if not self._place(rows, group, spread):
                return None
        return rows

    def _emit(self, groups: list[_Group]) -> Iterator[list[ScoredRow]]:
        rows = self._build(groups, spread=False)
        assert rows is not None, "a step that held its groups must rebuild"
        if any(not row.samples for row in rows):
            spread = self._build(groups, spread=True)
            if spread is not None and all(row.samples for row in spread):
                rows = spread
        smallest = min(
            (s for row in rows for s in row.samples), key=lambda s: len(s["input_ids"])
        )
        for row in rows:
            if not row.samples:
                row.add(
                    {
                        **smallest,
                        "completion_mask": np.zeros_like(smallest["completion_mask"]),
                        "metrics": {},
                        "score_uid": -1,
                        "score_truncated": False,
                    }
                )
                self.metrics["batch/filler_rows_total"].append(1.0)
        # Rows of similar cost share a micro-batch, so ranks wait less on each other.
        rows.sort(key=lambda row: row.load, reverse=True)
        for m in range(self.steps):
            yield rows[m * self.num_processes : (m + 1) * self.num_processes]


@dataclass
class ScoringCollator(TraceCollator):
    """TRL's collator plus what ``score_step`` needs on every rank."""

    scorers: list[TrainerScorer] = field(default_factory=list)

    def torch_call(self, examples: list[Any]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        if STOP_KEY in batch:
            return batch
        (rows,) = examples
        per_row: dict[str, tuple[list[torch.Tensor], float]] = {
            "score_uid": ([], -1),
            "score_group": ([], -1),
            "score_truncated": ([], 0),
            "score_rewards": ([], float("nan")),
            "score_extra_uid": ([], -1),
            "score_extra_group": ([], -1),
        }
        for row in rows:
            per_row["score_uid"][0].append(
                torch.tensor([s["score_uid"] for s in row.samples], dtype=torch.long)
            )
            per_row["score_group"][0].append(
                torch.tensor([s["group_id"] for s in row.samples], dtype=torch.long)
            )
            per_row["score_truncated"][0].append(
                torch.tensor(
                    [s["score_truncated"] for s in row.samples], dtype=torch.long
                )
            )
            # Fillers keep their source's payload; their uid of -1 leaves them unscored.
            rewards = [s["score_payload"]["rewards"] for s in row.samples]
            per_row["score_rewards"][0].append(
                torch.tensor(rewards, dtype=torch.float32).reshape(
                    len(rewards), len(rewards[0])
                )
            )
            # A row without extras still needs one entry to pad against.
            per_row["score_extra_uid"][0].append(
                torch.tensor([e["uid"] for e in row.extras] or [-1], dtype=torch.long)
            )
            per_row["score_extra_group"][0].append(
                torch.tensor(
                    [e["group_id"] for e in row.extras] or [-1], dtype=torch.long
                )
            )
        for index, scorer in enumerate(self.scorers):
            for key, value in scorer.collate(rows, index).items():
                per_row[f"score{index}_{key}"] = value
        for key, (tensors, padding_value) in per_row.items():
            batch[key] = pad(tensors, padding_value=padding_value)
        return batch


def _gather(record):
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return gather_object([record])
    return [record]


def score_step(
    trainer, batches: list[dict[str, torch.Tensor]]
) -> tuple[list[str], dict[int, tuple[np.ndarray, float]]] | None:
    """Score one optimizer step's micro-batches in place (see module doc).

    On rank 0, returns the reward names and each sample's reward columns and
    advantage, for the ``on_rollouts_scored`` hook.
    """
    start = time.time()
    scorers: list[TrainerScorer] = trainer._trainer_scorers
    args = trainer.args
    local_features: list[dict[int, torch.Tensor]] = []
    for index, scorer in enumerate(scorers):
        prefix = f"score{index}_"
        local: dict[int, torch.Tensor] = {}
        for batch in batches:
            inputs = {
                key[len(prefix) :]: value
                for key, value in batch.items()
                if key.startswith(prefix)
            }
            for chunk in chunk_inputs(inputs):
                uids, feats = scorer.features(trainer, chunk)
                for uid, feat in zip(uids.tolist(), feats.float().cpu(), strict=True):
                    if uid >= 0:
                        local[uid] = feat
        local_features.append(local)

    local_records: dict[int, tuple[int, list[float], bool]] = {}
    local_extras: dict[int, int] = {}
    for batch in batches:
        uids = batch["score_uid"][0].tolist()
        groups = batch["score_group"][0].tolist()
        truncated = batch["score_truncated"][0].tolist()
        rewards = batch["score_rewards"][0].cpu()
        for i, uid in enumerate(uids):
            if uid >= 0:
                local_records[uid] = (
                    groups[i],
                    rewards[i].tolist(),
                    bool(truncated[i]),
                )
        for uid, group in zip(
            batch["score_extra_uid"][0].tolist(),
            batch["score_extra_group"][0].tolist(),
            strict=True,
        ):
            if uid >= 0:
                local_extras[uid] = group

    records: dict[int, tuple[int, list[float], bool]] = {}
    extras: dict[int, int] = {}
    features: list[dict[int, torch.Tensor]] = [{} for _ in scorers]
    for rank_features, rank_records, rank_extras in _gather(
        (local_features, local_records, local_extras)
    ):
        for merged, feats in zip(features, rank_features, strict=True):
            merged.update(feats)
        records.update(rank_records)
        extras.update(rank_extras)

    members: dict[int, list[int]] = {}
    for uid in sorted(records):
        members.setdefault(records[uid][0], []).append(uid)
    extra_members: dict[int, list[int]] = {}
    for uid in sorted(extras):
        extra_members.setdefault(extras[uid], []).append(uid)

    if not records:
        return None
    num_child_columns = len(next(iter(records.values()))[1])
    weights = np.asarray(
        args.reward_weights or [1.0] * (num_child_columns + len(scorers)),
        dtype=float,
    )
    advantage: dict[int, float] = {}
    reward: dict[int, float] = {}
    columns: dict[int, np.ndarray] = {}
    group_stds: list[float] = []
    scorer_metrics: list[dict[str, list[float]]] = [{} for _ in scorers]
    for group_id, uids in members.items():
        child = np.array([records[uid][1] for uid in uids], dtype=float).T
        scored = []
        for index, scorer in enumerate(scorers):
            feats = features[index]
            extra = [
                feats[uid] for uid in extra_members.get(group_id, []) if uid in feats
            ]
            rewards, metrics = scorer.group_rewards(
                torch.stack([feats[uid] for uid in uids]),
                torch.stack(extra) if extra else None,
            )
            scored.append(np.asarray(rewards, dtype=float))
            for key, value in metrics.items():
                scorer_metrics[index].setdefault(key, []).append(value)
        per_func = np.concatenate([child.reshape(-1, len(uids)), np.stack(scored)])
        group_rewards, group_advantages = compute_group_advantages(
            per_func,
            weights,
            args.multi_objective_aggregation,
            args.scale_rewards,
        )
        finite = group_rewards[~np.isnan(group_rewards)]
        if finite.size:
            group_stds.append(float(finite.std()))
        for i, uid in enumerate(uids):
            advantage[uid] = float(group_advantages[i])
            reward[uid] = float(group_rewards[i])
            columns[uid] = per_func[:, i]

    skip_zero = args.skip_zero_advantage_batches
    mask_truncated = args.mask_truncated_completions
    excluded = {
        uid
        for uid in records
        if (skip_zero and advantage[uid] == 0.0) or (mask_truncated and records[uid][2])
    }
    progress = getattr(trainer, "_progress", None)
    if progress is not None and trainer.accelerator.is_main_process:
        progress.train(
            group_id
            for group_id, uids in members.items()
            if any(uid not in excluded for uid in uids)
        )
        for group_id, uids in members.items():
            if all(uid in excluded for uid in uids):
                zero = skip_zero and any(advantage[uid] == 0.0 for uid in uids)
                progress.exclude(group_id, "zero_adv" if zero else "truncated")
    counts = []
    for batch in batches:
        uids = batch["score_uid"][0]
        token_index = batch["token_index"][0]
        segment = batch["segment_id"][0].clamp(min=0, max=max(len(uids) - 1, 0))
        row_advantage = torch.tensor(
            [advantage.get(uid, 0.0) for uid in uids.tolist()],
            dtype=batch["advantages"].dtype,
            device=uids.device,
        )
        row_excluded = torch.tensor(
            [uid < 0 or uid in excluded for uid in uids.tolist()],
            dtype=torch.bool,
            device=uids.device,
        )
        selected = token_index >= 0
        drop = selected & row_excluded[segment]
        batch["advantages"][0] = torch.where(
            selected, row_advantage[segment], torch.zeros_like(row_advantage[segment])
        )
        batch["token_index"][0] = torch.where(drop, -1, token_index)
        counts.append((selected & ~drop).sum())
    counts = torch.stack(counts).to(torch.float32)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(counts)
    for batch, count in zip(batches, counts, strict=True):
        batch["global_n_tokens"].fill_(float(count))

    if trainer.accelerator.is_main_process and records:
        metrics = trainer._metrics["train"]
        values = [reward[uid] for uid in records if not np.isnan(reward[uid])]
        metrics["reward"].append(float(np.mean(values)) if values else float("nan"))
        if group_stds:
            metrics["reward_std"].append(float(np.mean(group_stds)))
        names = list(trainer._step_planner.reward_names or []) + [
            scorer.name for scorer in scorers
        ]
        stacked = np.stack([columns[uid] for uid in records])
        for index, name in enumerate(names):
            column = stacked[:, index]
            column = column[~np.isnan(column)]
            if column.size:
                metrics[f"rewards/{name}"].append(float(column.mean()))
        for scorer_metric in scorer_metrics:
            for key, per_group in scorer_metric.items():
                metrics[key].append(float(np.mean(per_group)))
        metrics["sample/excluded_frac"].append(
            (float(len(excluded)), float(len(records)))
        )
        metrics["perf/score_s"].append(time.time() - start)
        return names, {uid: (columns[uid], advantage[uid]) for uid in records}
    return None


def check_step_capacity(
    *,
    group_rows: int,
    steps: int,
    num_processes: int,
    token_budget: int,
    samples_per_row: int,
) -> None:
    """Fail early when a group can never fit in one optimizer step."""
    if token_budget <= 0 and group_rows > steps * num_processes * samples_per_row:
        raise ValueError(
            f"Trainer-side scoring keeps each group of {group_rows} samples inside one "
            f"optimizer step, which holds only {steps * num_processes * samples_per_row} "
            "samples (gradient_accumulation_steps x processes x "
            "per_device_train_batch_size). Raise one of them, or set token_budget."
        )


__all__ = [
    "GroupedQueue",
    "ScoredRow",
    "ScoringCollator",
    "StepPlanner",
    "TrainerScorer",
    "check_step_capacity",
    "chunk_inputs",
    "chunk_sequences",
    "pack_sequences",
    "score_step",
    "unpack_sequences",
]
