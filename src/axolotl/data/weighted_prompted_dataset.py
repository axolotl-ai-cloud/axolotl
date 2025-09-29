"""Weighted iterable dataset for prompted multitask JSONL.

Provides on-the-fly weighted sampling with replacement using per-record
`sample_weight`. Designed for small/medium JSONL loaded into memory.

Features:
- Deterministic set_epoch(seed) handling (like torch DistributedSampler)
- Optional temperature scaling of weights
- Supports distributed training: each replica independently samples its share
  of the global epoch_size so overall expected distribution matches weights.
- Exposes per-epoch empirical task distribution stats.

Usage example:

```python
from axolotl.data.weighted_prompted_dataset import WeightedPromptedIterableDataset

ds = WeightedPromptedIterableDataset(
    path="data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl",
    epoch_size=1000,
    temperature=1.0,
    seed=42,
)
for rec in ds:  # iter over one epoch
    ...
# advance epoch
ds.set_epoch(1)
```
"""
from __future__ import annotations
import json
import math
import random
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional

try:
    import torch
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover
    class IterableDataset:  # type: ignore
        pass
    torch = None  # type: ignore


def _load_records(path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def _compute_probs(records: List[Dict[str, Any]], temperature: float) -> List[float]:
    weights = [float(r.get("sample_weight", 1.0)) for r in records]
    if temperature != 1.0:
        scaled = [w ** (1.0/temperature) for w in weights]
    else:
        scaled = weights
    total = sum(scaled)
    if total <= 0:
        raise ValueError("Total weight must be positive")
    return [w/total for w in scaled]


class WeightedPromptedIterableDataset(IterableDataset):
    def __init__(
            self,
            path: str,
            epoch_size: int,
            temperature: float = 1.0,
            seed: int = 42,
            world_size: int = 1,
            rank: int = 0,
            enforce_quota: bool = False,
            stratify_window_size: int | None = None,
            stratify_shuffle: bool = True,
    ) -> None:
        self.path = Path(path)
        self.records = _load_records(self.path)
        self.epoch_size = epoch_size
        self.temperature = temperature
        self.base_seed = seed
        self.world_size = world_size
        self.rank = rank
        self.enforce_quota = enforce_quota
        self.stratify_window_size = stratify_window_size
        self.stratify_shuffle = stratify_shuffle
        self._epoch = 0
        self._probs = _compute_probs(self.records, self.temperature)
        # precompute cumulative for binary search
        self._cumulative: List[float] = []
        c = 0.0
        for p in self._probs:
            c += p
            self._cumulative.append(c)
        # build task index mapping
        self._task_to_indices: Dict[str, List[int]] = {}
        for i, rec in enumerate(self.records):
            self._task_to_indices.setdefault(rec.get("task"), []).append(i)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _rng(self) -> random.Random:
        # combine base seed, epoch, rank for determinism
        composite = (self.base_seed * 31 + self._epoch * 131 + self.rank * 997) & 0xFFFFFFFF
        return random.Random(composite)

    def _sample_index(self, r: float) -> int:
        # binary search cumulative
        lo, hi = 0, len(self._cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= self._cumulative[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        # each rank samples epoch_size/world_size (ceil last rank adjustment)
        per_rank = self.epoch_size // self.world_size
        remainder = self.epoch_size % self.world_size
        share = per_rank + (1 if self.rank < remainder else 0)
        rng = self._rng()
        appearances = [0] * len(self.records)
        if not self.enforce_quota:
            if self.stratify_window_size and self.stratify_window_size > 0:
                # Build a window with approximate balanced task representation
                tasks = sorted(self._task_to_indices.keys())  # deterministic order
                # Precompute cumulative per-task local probabilities for sampling within task
                per_task_local_probs: Dict[str, List[float]] = {}
                per_task_cumul: Dict[str, List[float]] = {}
                for t, idxs in self._task_to_indices.items():
                    local_weights = [self._probs[i] for i in idxs]
                    s = sum(local_weights)
                    if s == 0:
                        local_probs = [1/len(local_weights)] * len(local_weights)
                    else:
                        local_probs = [w/s for w in local_weights]
                    per_task_local_probs[t] = local_probs
                    cumul = []
                    csum = 0.0
                    for lp in local_probs:
                        csum += lp
                        cumul.append(csum)
                    per_task_cumul[t] = cumul
                window: List[Dict[str, Any]] = []
                produced = 0
                task_index = 0
                while produced < share:
                    # ensure each window contains at most one sample per task (round-robin)
                    t = tasks[task_index]
                    task_index = (task_index + 1) % len(tasks)
                    idxs = self._task_to_indices[t]
                    cumul = per_task_cumul[t]
                    r_local = rng.random()
                    # binary search within task
                    lo, hi = 0, len(cumul) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if r_local <= cumul[mid]:
                            hi = mid
                        else:
                            lo = mid + 1
                    real_idx = idxs[lo]
                    appearances[real_idx] += 1
                    rec = self.records[real_idx]
                    out = json.loads(json.dumps(rec))
                    out["epoch_instance"] = appearances[real_idx]
                    out["epoch"] = self._epoch
                    window.append(out)
                    produced += 1
                    if len(window) >= self.stratify_window_size:
                        if self.stratify_shuffle:
                            rng.shuffle(window)
                        for w_rec in window:
                            yield w_rec
                        window = []
                if window:
                    if self.stratify_shuffle:
                        rng.shuffle(window)
                    for w_rec in window:
                        yield w_rec
            else:
                for _ in range(share):
                    r = rng.random()
                    idx = self._sample_index(r)
                    appearances[idx] += 1
                    rec = self.records[idx]
                    out = json.loads(json.dumps(rec))
                    out["epoch_instance"] = appearances[idx]
                    out["epoch"] = self._epoch
                    yield out
        else:
            # Quota mode: deterministic per-task allocation using largest remainder method.
            # Compute global expected counts then slice this rank's share.
            # Step 1: expected global counts per task
            task_probs: Dict[str, float] = {}
            for rec, p in zip(self.records, self._probs):
                task_probs.setdefault(rec.get("task"), 0.0)
                task_probs[rec.get("task")] += p
            # target counts (global)
            raw_targets = {t: task_probs[t] * self.epoch_size for t in task_probs}
            base_targets = {t: int(raw_targets[t]) for t in raw_targets}
            remainder_pairs = [
                (raw_targets[t] - base_targets[t], t) for t in raw_targets
            ]
            remaining = self.epoch_size - sum(base_targets.values())
            # allocate remaining by largest remainder
            for _, t in sorted(remainder_pairs, key=lambda x: x[0], reverse=True)[:remaining]:
                base_targets[t] += 1
            global_task_counts = base_targets
            # derive per-rank task counts (simple proportional split + remainder by task name order)
            per_rank_task: Dict[str, int] = {}
            for t, gcount in global_task_counts.items():
                q = gcount // self.world_size
                rmd = gcount % self.world_size
                per_rank_task[t] = q + (1 if self.rank < rmd else 0)
            if self.stratify_window_size and self.stratify_window_size > 0:
                # Build deterministic windows honoring per-rank task quotas.
                tasks = sorted(per_rank_task.keys())
                # Prepare per-task cumulative distributions
                task_cumul: Dict[str, List[float]] = {}
                task_indices: Dict[str, List[int]] = {}
                remaining_task_need = {t: per_rank_task[t] for t in tasks}
                for t in tasks:
                    need = remaining_task_need[t]
                    if need <= 0:
                        continue
                    idxs = self._task_to_indices[t]
                    task_indices[t] = idxs
                    local_weights = [self._probs[i] for i in idxs]
                    s = sum(local_weights)
                    local_probs = [w / s for w in local_weights]
                    cumul = []
                    csum = 0.0
                    for lp in local_probs:
                        csum += lp
                        cumul.append(csum)
                    task_cumul[t] = cumul
                window: List[Dict[str, Any]] = []
                t_index = 0
                produced_total = 0
                # Round-robin fill until all per-rank quotas consumed
                while any(remaining_task_need[t] > 0 for t in tasks):
                    t = tasks[t_index]
                    t_index = (t_index + 1) % len(tasks)
                    if remaining_task_need[t] <= 0:
                        continue
                    cumul = task_cumul[t]
                    idxs = task_indices[t]
                    rloc = rng.random()
                    lo, hi = 0, len(cumul) - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if rloc <= cumul[mid]:
                            hi = mid
                        else:
                            lo = mid + 1
                    real_idx = idxs[lo]
                    appearances[real_idx] += 1
                    rec = self.records[real_idx]
                    out = json.loads(json.dumps(rec))
                    out["epoch_instance"] = appearances[real_idx]
                    out["epoch"] = self._epoch
                    window.append(out)
                    remaining_task_need[t] -= 1
                    produced_total += 1
                    if len(window) >= self.stratify_window_size:
                        if self.stratify_shuffle:
                            rng.shuffle(window)
                        for w_rec in window:
                            yield w_rec
                        window = []
                if window:
                    if self.stratify_shuffle:
                        rng.shuffle(window)
                    for w_rec in window:
                        yield w_rec
            else:
                # For each task, sample indices with weighted probabilities restricted to that task.
                for t in sorted(per_rank_task.keys()):
                    need = per_rank_task[t]
                    if need <= 0:
                        continue
                    idxs = self._task_to_indices[t]
                    local_weights = [self._probs[i] for i in idxs]
                    lw_sum = sum(local_weights)
                    local_probs = [w / lw_sum for w in local_weights]
                    cumulative = []
                    csum = 0.0
                    for lp in local_probs:
                        csum += lp
                        cumulative.append(csum)
                    for _ in range(need):
                        r = rng.random()
                        lo, hi = 0, len(cumulative) - 1
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if r <= cumulative[mid]:
                                hi = mid
                            else:
                                lo = mid + 1
                        real_idx = idxs[lo]
                        appearances[real_idx] += 1
                        rec = self.records[real_idx]
                        out = json.loads(json.dumps(rec))
                        out["epoch_instance"] = appearances[real_idx]
                        out["epoch"] = self._epoch
                        yield out

    def expected_task_counts(self) -> Dict[str, float]:
        # expected values for this rank only
        per_rank = self.epoch_size / self.world_size
        counts: Dict[str, float] = {}
        for rec, p in zip(self.records, self._probs):
            counts.setdefault(rec.get("task"), 0.0)
            counts[rec.get("task")] += p * per_rank
        return counts

    def tasks(self) -> List[str]:
        return sorted({rec.get("task") for rec in self.records})

    def record_count(self) -> int:
        return len(self.records)


__all__ = ["WeightedPromptedIterableDataset"]


def verify_prompted_manifest(data_path: str | Path, manifest_path: str | Path, strict_weights: bool = True) -> dict:
    """Verify a previously frozen manifest matches the current dataset file.

    Checks:
      - sha256 hash matches
      - line_count matches
      - per-task record counts match
      - per-task and total sample_weight sums match (optionally tolerant to fp rounding)

    Args:
        data_path: path to JSONL dataset
        manifest_path: path to manifest JSON produced by freeze_prompted_dataset.py
        strict_weights: if False, allow small relative diff (1e-6) in weight sums

    Returns: loaded manifest dict (on success)
    Raises: ValueError on any mismatch
    """
    import hashlib, json as _json
    from collections import Counter as _Counter
    data_p = Path(data_path)
    man_p = Path(manifest_path)
    if not data_p.exists():
        raise ValueError(f"Dataset file missing: {data_p}")
    if not man_p.exists():
        raise ValueError(f"Manifest file missing: {man_p}")
    manifest = _json.loads(man_p.read_text(encoding="utf-8"))
    # recompute hash
    h = hashlib.sha256()
    with data_p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    current_sha = h.hexdigest()
    if current_sha != manifest.get('sha256'):
        raise ValueError(f"SHA256 mismatch: manifest={manifest.get('sha256')} current={current_sha}")
    # recompute counts & weights
    counts = _Counter()
    weights = _Counter()
    line_count = 0
    with data_p.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            rec = _json.loads(line)
            t = rec.get('task')
            counts[t] += 1
            weights[t] += float(rec.get('sample_weight', 1.0))
    if line_count != manifest.get('line_count'):
        raise ValueError(f"line_count mismatch: manifest={manifest.get('line_count')} current={line_count}")
    man_counts = manifest.get('task_counts', {})
    if dict(counts) != man_counts:
        raise ValueError(f"task_counts mismatch: manifest={man_counts} current={dict(counts)}")
    man_weights = manifest.get('per_task_sample_weight', {})
    # compare weights
    for t, w in weights.items():
        mw = float(man_weights.get(t, 'nan'))
        if strict_weights:
            if abs(mw - w) > 1e-9:
                raise ValueError(f"weight mismatch for task {t}: manifest={mw} current={w}")
        else:
            if mw == 0 and w == 0:
                continue
            rel = abs(mw - w) / max(abs(mw), abs(w), 1e-12)
            if rel > 1e-6:
                raise ValueError(f"weight mismatch for task {t}: manifest={mw} current={w} rel_diff={rel}")
    total_w = sum(weights.values())
    if strict_weights:
        if abs(total_w - manifest.get('total_sample_weight')) > 1e-9:
            raise ValueError("total_sample_weight mismatch")
    else:
        man_total = float(manifest.get('total_sample_weight', 'nan'))
        rel = abs(total_w - man_total) / max(abs(total_w), abs(man_total), 1e-12)
        if rel > 1e-6:
            raise ValueError(f"total_sample_weight mismatch: manifest={man_total} current={total_w} rel_diff={rel}")
    return manifest

__all__.append('verify_prompted_manifest')
