"""Count host-device synchronizations per micro-step with torch's sync debug mode."""

import json
import os
import traceback
import warnings
from collections import Counter

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_SYNC_MARKER = "synchroniz"
_LIBRARY_ROOTS = ("/axolotl/", "/transformers/", "/trl/", "/accelerate/", "/peft/")
_SKIP_ROOTS = ("/torch/", "/warnings.py", "sync_profiler.py")


def _site_of(stack: list[traceback.FrameSummary]) -> str:
    """Innermost frame outside torch; prefer a frame in one of the training libraries."""
    fallback = None
    for frame in reversed(stack):
        filename = frame.filename
        if any(root in filename for root in _SKIP_ROOTS):
            continue
        for root in _LIBRARY_ROOTS:
            if root in filename:
                short = filename[filename.index(root) + 1 :]
                return f"{short}:{frame.lineno} {frame.name}"
        if fallback is None:
            fallback = f"{os.path.basename(filename)}:{frame.lineno} {frame.name}"
    return fallback or "<torch-internal>"


class SyncProfilerCallback(TrainerCallback):
    """Count syncing CUDA ops per micro-step for a window of optimizer steps.

    Uses ``torch.cuda.set_sync_debug_mode("warn")`` and buckets the resulting
    warnings by the trainer callback that follows them: one bucket per
    gradient-accumulation micro-step, plus separate buckets for logging, saving
    and evaluation. Warnings raised on autograd worker threads are delivered
    when ``backward()`` returns, so they land in the right micro-step but are
    attributed to the backward call rather than the op inside it.
    """

    def __init__(self, start_step: int = 2, num_steps: int = 3):
        self.start_step = start_step
        self.num_steps = num_steps
        self._active = False
        self._prev_showwarning = None
        self._prev_filters: list | None = None
        self._current: Counter = Counter()
        self._samples: dict[str, list[str]] = {}
        self._buckets: list[dict] = []
        self._substep = 0
        self._window_done = False
        self._prev_sync_mode: str | None = None
        self.result: dict | None = None

    # ---- lifecycle -------------------------------------------------------

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "between_steps")
            if self._window_done:
                # the last measured step's log/save/evaluate events have now run
                self._finish(args)
        elif state.global_step == self.start_step:
            self._enable()
        self._substep = 0

    def on_substep_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "microstep")
            self._substep += 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if not self._active:
            return
        self._flush(state.global_step, "microstep")
        # HF increments global_step before on_step_end, so it already counts this step.
        if state.global_step >= self.start_step + self.num_steps:
            self._window_done = True

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "log")

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "save")

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "evaluate")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._active:
            self._flush(state.global_step, "between_steps")
            self._finish(args)

    # ---- capture ---------------------------------------------------------

    def _enable(self):
        if not torch.cuda.is_available():
            LOG.warning("sync profiler needs CUDA; skipping")
            return
        self._prev_sync_mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode("warn")
        self._prev_showwarning = warnings.showwarning
        self._prev_filters = warnings.filters[:]
        warnings.simplefilter("always")
        warnings.showwarning = self._showwarning
        self._active = True
        LOG.info(
            f"sync profiler: counting syncing CUDA ops for {self.num_steps} step(s)"
        )

    def _disable(self):
        if not self._active:
            return
        torch.cuda.set_sync_debug_mode(self._prev_sync_mode or "default")
        warnings.showwarning = self._prev_showwarning
        if self._prev_filters is not None:
            warnings.filters[:] = self._prev_filters
        self._active = False

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        if _SYNC_MARKER not in str(message):
            self._prev_showwarning(message, category, filename, lineno, file, line)
            return
        stack = traceback.extract_stack()[:-1]
        site = _site_of(stack)
        self._current[site] += 1
        if site not in self._samples:
            self._samples[site] = [
                s.rstrip() for s in traceback.format_list(stack[-10:])
            ]

    def _flush(self, step: int, kind: str):
        if kind == "between_steps" and not self._current:
            return
        self._buckets.append(
            {
                "step": step,
                "substep": self._substep if kind == "microstep" else None,
                "kind": kind,
                "total": sum(self._current.values()),
                "sites": dict(self._current),
            }
        )
        self._current = Counter()

    # ---- report ----------------------------------------------------------

    def _finish(self, args: TrainingArguments):
        self._disable()
        microsteps = [b for b in self._buckets if b["kind"] == "microstep"]
        totals = sorted(b["total"] for b in microsteps)
        per_site: Counter = Counter()
        for bucket in microsteps:
            per_site.update(bucket["sites"])
        other = {
            kind: sum(b["total"] for b in self._buckets if b["kind"] == kind)
            for kind in ("log", "save", "evaluate", "between_steps")
        }
        self.result = {
            "microsteps": len(microsteps),
            "syncs_per_microstep": {
                "min": totals[0] if totals else 0,
                "median": totals[len(totals) // 2] if totals else 0,
                "max": totals[-1] if totals else 0,
            },
            "sites_per_microstep": {
                site: count / max(len(microsteps), 1)
                for site, count in per_site.most_common()
            },
            "other_buckets": other,
            "buckets": self._buckets,
            "samples": self._samples,
        }
        lines = [
            f"sync profiler: {len(microsteps)} micro-steps, syncs per micro-step "
            f"min/median/max = {self.result['syncs_per_microstep']['min']}/"
            f"{self.result['syncs_per_microstep']['median']}/"
            f"{self.result['syncs_per_microstep']['max']}; "
            + ", ".join(f"{k}={v}" for k, v in other.items())
        ]
        for site, rate in list(self.result["sites_per_microstep"].items())[:25]:
            lines.append(f"  {rate:8.2f}/microstep  {site}")
        LOG.info("\n".join(lines))
        if getattr(args, "output_dir", None) and getattr(args, "process_index", 0) == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            path = os.path.join(args.output_dir, "sync_profile.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.result, handle, indent=2)
            LOG.info(f"sync profiler written to {path}")
