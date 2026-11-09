"""Publish a merged checkpoint per optimizer step, built by every rank in parallel.

Each rank keeps its own chain (baseline of the tensors it owns), builds and
encodes its share, and uploads its own shard files. Rank 0 gathers the per-rank
indexes, uploads the combined index last and hot-loads it through the
destination; only then do all ranks promote their staged bytes. Any failure on
any rank discards the attempt everywhere, so the next publish is built against
the last commit again.

A small record per rank under ``state_dir`` (durable storage) names the last
committed identity, its step and checksums. After a restart, ``restore``
rebuilds the baseline from the resumed adapter and adopts it only if every
tensor matches the published checksums; otherwise the next publish is full.
"""

import json
import shutil
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.checkpoint_publish import MAX_CHAIN_DELTAS

from .chain import (
    INDEX_NAME,
    ChainWriter,
    StagedCheckpoint,
    _dtype_name,
    _write_json,
    combine_indexes,
)
from .comm import LocalComm
from .destinations import Destination
from .exporters import _scale_mode
from .fingerprint import fingerprint_digest, merge_fingerprint
from .lora_shards import Piece, assemble, global_shapes, plan_transfers
from .pinned_base import PinnedBase
from .rank_builder import MergeContext, RankBuilder, plan_work, work_digest

LOG = get_logger(__name__)
MERGE_PATH = "rank_builder"
TENSOR_CLASSES = (
    "expert_weight",
    "block_scales",
    "global_scales",
    "attention",
    "other",
)


def tensor_class(name: str) -> str:
    if ".experts." in name:
        if name.endswith(".weight_scale_2"):
            return "global_scales"
        if name.endswith(".weight_scale"):
            return "block_scales"
        if name.endswith(".weight"):
            return "expert_weight"
    if ".self_attn." in name or ".attention." in name:
        return "attention"
    return "other"


def _staged_stats(staged: StagedCheckpoint, dtypes: dict) -> dict[str, Any]:
    changed = dict.fromkeys(TENSOR_CLASSES, 0)
    compressed = dict.fromkeys(TENSOR_CLASSES, 0)
    tensors = dict.fromkeys(TENSOR_CLASSES, 0)
    for name in staged.weight_map:
        kind = tensor_class(name)
        dtype, shape = dtypes[name]
        changed[kind] += torch.empty((), dtype=getattr(torch, dtype)).element_size() * (
            int(torch.Size(shape).numel())
        )
        compressed[kind] += (staged.encoded_sizes or {}).get(name, 0)
        tensors[kind] += 1
    return {"changed": changed, "compressed": compressed, "tensors": tensors}


class StepPublisher:
    """One rank's side of the distributed per-step publish."""

    def __init__(
        self,
        *,
        base: PinnedBase,
        adapter_config: dict,
        pieces: Sequence[Piece],
        destination: Destination,
        workdir: Path | str,
        config: dict | None = None,
        state_dir: Path | str | None = None,
        comm=None,
        device: str = "cpu",
        max_deltas: int = MAX_CHAIN_DELTAS,
        delta: bool = True,
        max_shard_size: int = 1 << 30,
        identity_prefix: str = "step",
        retain_artifacts: bool = False,
        expert_chunk: int = 8,
    ):
        self.comm = comm or LocalComm()
        self.rank, self.world = self.comm.rank, self.comm.world
        self.base, self.destination = base, destination
        self.adapter_config, self.device = adapter_config, device
        self.model_config = dict(config if config is not None else base.config)
        self.identity_prefix = identity_prefix
        self.pieces = list(pieces)
        self.all_pieces = self.comm.all_gather_object(self.pieces)
        self.context = MergeContext(
            base, adapter_config, global_shapes(self.all_pieces), device
        )
        self.work = plan_work(self.context, self.all_pieces, expert_chunk=expert_chunk)
        self.transfers = plan_transfers(
            self.all_pieces, [rank.needs for rank in self.work]
        )
        self.builder = RankBuilder(self.context, self.work[self.rank])
        self.split = work_digest(self.work)
        if destination.max_deltas is not None:
            max_deltas = min(max_deltas, destination.max_deltas)
        self.workdir = Path(workdir) / f"rank-{self.rank:05d}"
        self.chain = ChainWriter(
            self.workdir,
            max_deltas=max_deltas,
            delta=delta,
            max_shard_size=max_shard_size,
            rank=self.rank,
            retain_artifacts=retain_artifacts,
            resume=True,
        )
        self.state_dir = Path(state_dir) if state_dir is not None else None
        self._stream = None
        idle = [
            r
            for r, work in enumerate(self.work)
            if not work.owned(self.context.inventory)
        ]
        if idle:
            raise ValueError(
                f"Ranks {idle} of {self.world} would publish no tensors; the base has "
                "fewer files and LoRA targets than ranks"
            )

    @property
    def identity(self) -> str | None:
        return self.chain.identity

    def fingerprint(self) -> dict:
        """Computed per publish so the live TF32 state is the one recorded."""
        return merge_fingerprint(
            export="merged",
            base_identity=self.base.identity,
            merge_path=MERGE_PATH,
            merge_compile=None,
            scale_mode=_scale_mode(self.adapter_config),
            device=self.device,
            extra={
                "merged_format": "preserve",
                "world": self.world,
                "split": self.split,
            },
        )

    def new_identity(self, step: int) -> str:
        return f"{self.identity_prefix}-{step:08d}-{uuid.uuid4().hex}"

    def close(self):
        self.chain.close()

    def _build(self, identity, data, *, fingerprint, force_full, timings):
        device = torch.device(self.device)
        if device.type != "cuda":
            return self._build_on(identity, data, fingerprint, force_full, timings)
        # the publisher thread starts on GPU 0 and must not queue behind training kernels
        with torch.cuda.device(device):
            if self._stream is None:
                self._stream = torch.cuda.Stream(device)
            with torch.cuda.stream(self._stream):
                return self._build_on(identity, data, fingerprint, force_full, timings)

    def _build_on(self, identity, data, fingerprint, force_full, timings):
        start = time.perf_counter()
        assembled = assemble(
            self.comm,
            self.rank,
            data,
            self.work[self.rank].needs,
            self.transfers,
            self.all_pieces,
        )
        timings["gather"] = time.perf_counter() - start
        # after the collective gather, so a changed base fails this rank without hanging the others
        self.base.verify()
        start = time.perf_counter()
        layout = self.base.layout
        with self.chain.begin(
            identity, fingerprint=fingerprint, force_full=force_full
        ) as session:

            def sink(name, tensor):
                entry = (_dtype_name(tensor.dtype), list(tensor.shape))
                if entry != layout.get(name):
                    raise ValueError(
                        f"Merged tensor {name} does not match the pinned base layout: "
                        f"{entry} vs {layout.get(name)}"
                    )
                session.add(name, tensor, copy=False)

            self.builder.build(assembled, sink, full=session.full)
            if not session.full:
                for name in self.builder.untargeted:
                    session.keep(name)
            timings["build"] = time.perf_counter() - start
            start = time.perf_counter()
            staged = session.finish(self.model_config)
        timings["encode"] = time.perf_counter() - start
        return staged

    def _agree(self, fingerprint) -> tuple[bool, str]:
        digest = fingerprint_digest(fingerprint)
        reason = self.chain.full_reason(fingerprint)
        states = self.comm.all_gather_object(
            (self.chain.identity, self.chain.delta_count, digest, reason)
        )
        if len({state[:3] for state in states}) > 1:
            return True, "ranks disagree on the committed baseline or merge numerics"
        reasons = sorted({state[3] for state in states if state[3] is not None})
        return bool(reasons), "; ".join(reasons) or "delta"

    def publish(self, data: Sequence[torch.Tensor], step: int) -> dict[str, Any]:
        """Collective: every rank calls this with its own staged LoRA slices."""
        started = time.perf_counter()
        identity = self.comm.broadcast_object(
            self.new_identity(step) if self.rank == 0 else None
        )
        fingerprint = self.fingerprint()
        force_full, reason = self._agree(fingerprint)
        timings: dict[str, float] = {}
        staged: StagedCheckpoint | None = None
        error = None
        try:
            staged = self._build(
                identity,
                data,
                fingerprint=fingerprint,
                force_full=force_full,
                timings=timings,
            )
            start = time.perf_counter()
            names = sorted(set(staged.weight_map.values()))
            if names:
                self.destination.publish_files(staged.path, identity, names)
            timings["upload"] = time.perf_counter() - start
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Rank %d failed to build checkpoint %s", self.rank, identity)
            error = f"rank {self.rank}: {exc!r}"
        report = None
        if staged is not None and error is None:
            report = {
                "index": json.loads((staged.path / INDEX_NAME).read_text()),
                "full": staged.full,
                "empty": staged.empty,
                "base_identity": staged.base_identity,
                "stats": _staged_stats(staged, self.base.layout),
                "artifact_size": staged.artifact_size,
                "timings": timings,
            }
        reports = self.comm.all_gather_object((error, report))
        decision = None
        if self.rank == 0:
            decision = self._decide(identity, reports, step)
        decision = self.comm.broadcast_object(decision)
        committed = False
        try:
            if decision["action"] == "commit":
                assert staged is not None
                self.chain.commit(staged)
                self._record(step)
                committed = True
            elif staged is not None:
                self.chain.discard(staged)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Rank %d failed to commit %s", self.rank, identity)
            error = f"rank {self.rank} commit: {exc!r}"
        outcomes = self.comm.all_gather_object(
            error if decision["action"] == "commit" else None
        )
        if decision["action"] == "commit" and any(outcomes):
            LOG.error(
                "Checkpoint %s was published but not committed on every rank (%s); "
                "the next publish will be a full checkpoint",
                identity,
                "; ".join(o for o in outcomes if o),
            )
        metrics = dict(decision.get("metrics") or {})
        metrics.update(
            identity=identity if committed else self.chain.identity,
            published=decision["action"] == "commit",
            action=decision["action"],
            reason=reason,
            seconds_total=time.perf_counter() - started,
        )
        if decision["action"] == "discard":
            metrics["errors"] = decision["errors"]
        return metrics

    def _decide(self, identity, reports, step) -> dict[str, Any]:
        errors = [error for error, _ in reports if error is not None]
        metrics = self._aggregate([report for _, report in reports if report])
        if errors:
            self.destination.discard_files(identity)
            LOG.warning("Discarding checkpoint %s: %s", identity, "; ".join(errors))
            return {"action": "discard", "errors": errors, "metrics": metrics}
        if all(report["empty"] for _, report in reports):
            return {"action": "empty", "metrics": metrics}
        directory = self.workdir / f".index-{identity}"
        start = time.perf_counter()
        try:
            directory.mkdir(parents=True)
            full = reports[0][1]["full"]
            index = combine_indexes(report["index"] for _, report in reports)
            (directory / INDEX_NAME).write_text(json.dumps(index), encoding="utf-8")
            if full:
                (directory / "config.json").write_text(
                    json.dumps(self.model_config), encoding="utf-8"
                )
            self.destination.publish_index(
                directory,
                identity,
                base_identity=reports[0][1]["base_identity"],
                full=full,
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOG.exception("Publishing the index of %s failed", identity)
            self.destination.discard_files(identity)
            return {"action": "discard", "errors": [repr(exc)], "metrics": metrics}
        finally:
            shutil.rmtree(directory, ignore_errors=True)
        metrics["seconds_publish_index"] = time.perf_counter() - start
        if self.state_dir is not None:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                self.state_dir / "committed.json",
                {"identity": identity, "step": step, "world": self.world, "full": full},
            )
        return {"action": "commit", "metrics": metrics}

    @staticmethod
    def _aggregate(reports: list[dict]) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        if not reports:
            return metrics
        metrics["full"] = any(report["full"] for report in reports)
        metrics["artifact_bytes"] = sum(report["artifact_size"] for report in reports)
        for measure in ("changed", "compressed", "tensors"):
            for kind in TENSOR_CLASSES:
                metrics[f"{measure}_{kind}"] = sum(
                    report["stats"][measure][kind] for report in reports
                )
        stages = {stage for report in reports for stage in report["timings"]}
        for stage in sorted(stages):
            metrics[f"seconds_{stage}"] = max(
                report["timings"].get(stage, 0.0) for report in reports
            )
        return metrics

    def _record_path(self) -> Path:
        assert self.state_dir is not None
        return self.state_dir / f"rank-{self.rank:05d}.json"

    def _record(self, step: int):
        if self.state_dir is None:
            return
        baseline = self.chain.baseline
        assert baseline is not None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            self._record_path(),
            {
                "identity": baseline.identity,
                "step": step,
                "delta_count": baseline.delta_count,
                "fingerprint": baseline.fingerprint_digest,
                "checksums": baseline.checksums,
            },
        )

    def _read_record(self) -> dict | None:
        if self.state_dir is None:
            return None
        path = self._record_path()
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def restore(self, data: Sequence[torch.Tensor], step: int) -> bool:
        """Collective: recover the committed baseline after a restart at ``step``.

        Returns whether every rank holds the last committed snapshot again; if
        not, every rank drops its baseline and the next publish is full.
        """
        record = self._read_record()
        committed = None
        if self.state_dir is not None and (self.state_dir / "committed.json").is_file():
            committed = json.loads((self.state_dir / "committed.json").read_text())
        status = "missing"
        if record is not None and committed is not None:
            if (
                record["identity"] != committed["identity"]
                or committed["world"] != self.world
            ):
                status = "stale"
            elif self.chain.identity == record["identity"]:
                status = "kept"
            elif record["step"] == step:
                status = "rebuild"
            else:
                status = "other step"
        statuses = self.comm.all_gather_object(status)
        ok = all(s in ("kept", "rebuild") for s in statuses)
        if ok and "rebuild" in statuses:
            rebuilt = False
            error = None
            if status == "rebuild":
                try:
                    rebuilt = self._rebuild(data, record)
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    LOG.exception("Rank %d could not rebuild its baseline", self.rank)
                    error = repr(exc)
            else:
                assemble(
                    self.comm,
                    self.rank,
                    data,
                    self.work[self.rank].needs,
                    self.transfers,
                    self.all_pieces,
                )
                rebuilt = True
            results = self.comm.all_gather_object((rebuilt, error))
            ok = all(result for result, _ in results)
        if not ok:
            if any(s != "missing" for s in statuses):
                LOG.warning(
                    "Cannot resume the published chain (%s); the next publish will be a "
                    "full checkpoint",
                    ", ".join(sorted(set(statuses))),
                )
            self._drop_baseline()
        return ok

    def _rebuild(self, data, record) -> bool:
        identity = record["identity"]
        stale = self.workdir / identity
        if stale.exists():
            shutil.rmtree(stale)
        self._drop_baseline()
        staged = self._build(
            identity,
            data,
            fingerprint=self.fingerprint(),
            force_full=True,
            timings={},
        )
        if (
            fingerprint_digest(self.fingerprint()) != record["fingerprint"]
            or staged.checksums != record["checksums"]
        ):
            self.chain.discard(staged)
            LOG.warning(
                "Rank %d: the rebuilt baseline does not match the published checksums "
                "of %s",
                self.rank,
                identity,
            )
            return False
        self.chain.adopt(staged, delta_count=record["delta_count"])
        return True

    def _drop_baseline(self):
        if self.chain.baseline is not None:
            shutil.rmtree(self.chain.baseline.path, ignore_errors=True)
            self.chain.baseline = None
