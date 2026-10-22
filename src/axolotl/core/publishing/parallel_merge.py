"""Split the sharded LoRA merge across several workers (one per GPU).

Each worker runs ``merge_lora_sharded_efficient`` on its own base shards, with its
own read, merge and write pipeline, and all workers share one read-only copy of
the adapter. The caller's ``shard_sink`` still sees every merged shard in base
order, exactly as the single-worker merge emits them, so a chain written from
the parallel merge is byte-identical to one written from a single worker.
"""

import json
import re
import shutil
import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import safetensors
import safetensors.torch
import torch

ShardSink = Callable[[dict[str, torch.Tensor], Path, dict], None]
_LAYER = re.compile(r"^(.*?\.\d+)\.")


def visible_devices(device: str, workers: int | None = None) -> list[str]:
    """One device per worker: ``cuda`` means every visible GPU, ``cpu`` repeats."""
    parsed = torch.device(device)
    if parsed.type == "cuda" and parsed.index is None:
        available = torch.cuda.device_count()
        count = available if workers is None else workers
        if count > available:
            raise ValueError(
                f"{count} merge workers requested, {available} GPUs visible"
            )
        return [f"cuda:{index}" for index in range(max(count, 1))]
    if workers is not None and workers > 1 and parsed.type != "cpu":
        raise ValueError(f"Cannot run {workers} merge workers on one device {device}")
    return [device] * (workers or 1)


def _shard_keys(shard: Path) -> list[str]:
    if shard.suffix == ".safetensors":
        with safetensors.safe_open(shard, framework="pt") as handle:
            return list(handle.keys())
    return list(torch.load(shard, map_location="meta", weights_only=True))  # nosec B614


def shard_groups(shards: Sequence[Path]) -> list[list[Path]]:
    """Contiguous runs of shards that must be merged by the same worker.

    Two shards belong together when they share a layer (expert lists are buffered
    across shard boundaries) or a module (a weight and its scales).
    """
    owner: dict[str, int] = {}
    spans = [[index, index] for index in range(len(shards))]
    for index, shard in enumerate(shards):
        for key in _shard_keys(shard):
            layer = _LAYER.match(key)
            for group_key in (key.rpartition(".")[0], layer and layer.group(1)):
                if group_key:
                    first = owner.setdefault(group_key, index)
                    spans[first][1] = max(spans[first][1], index)
    groups: list[list[Path]] = []
    end = -1
    for start, stop in sorted(spans):
        if start > end:
            groups.append([])
        groups[-1].extend(shards[max(start, end + 1) : stop + 1])
        end = max(end, stop)
    return groups


class _Ordered:
    """Buffer worker output per base shard and release it in base order.

    A worker blocks once it is ``window`` shards ahead of the shard being
    released, which bounds the merged tensors held in host memory.
    """

    def __init__(self, shards: Sequence[Path], window: int):
        self.position = {shard.name: index for index, shard in enumerate(shards)}
        self.items: list[deque] = [deque() for _ in shards]
        self.done = [False] * len(shards)
        self.next = 0
        self.window = max(window, 1)
        self.error: BaseException | None = None
        self.cond = threading.Condition()

    def put(self, shard_name: str, item):
        index = self.position[shard_name]
        with self.cond:
            self.cond.wait_for(
                lambda: self.error is not None or index < self.next + self.window
            )
            if self.error is not None:
                raise RuntimeError("Merge aborted") from self.error
            self.items[index].append(item)
            self.cond.notify_all()

    def finish(self, shard_name: str):
        with self.cond:
            self.done[self.position[shard_name]] = True
            self.cond.notify_all()

    def fail(self, error: BaseException):
        with self.cond:
            if self.error is None:
                self.error = error
            self.cond.notify_all()

    def drain(self, sink: ShardSink, alive: Callable[[], bool]):
        while self.next < len(self.items):
            with self.cond:
                self.cond.wait_for(
                    lambda: (
                        self.error is not None
                        or bool(self.items[self.next])
                        or self.done[self.next]
                        or not alive()
                    ),
                    timeout=1.0,
                )
                if self.error is not None:
                    raise self.error
                if self.items[self.next]:
                    item = self.items[self.next].popleft()
                elif self.done[self.next]:
                    self.next += 1
                    self.cond.notify_all()
                    continue
                elif not alive():
                    raise RuntimeError("Merge workers exited without finishing")
                else:
                    continue
            sink(*item)


def _combine_reports(reports: Sequence[dict]) -> dict:
    groups = [r["expert_groups"] for r in reports if r.get("expert_groups") is not None]
    return {
        "merged": sum(r.get("merged", 0) for r in reports),
        "expert_groups": sum(groups) if groups else None,
        "expert_lora_dropped": any(r.get("expert_lora_dropped") for r in reports),
    }


def _publish_config(worker_dirs: Sequence[Path], output_path: Path, dequant: bool):
    """Copy the non-weight files the first worker wrote into ``output_path``."""
    output_path.mkdir(parents=True, exist_ok=True)
    for path in worker_dirs[0].iterdir():
        if (
            path.is_file()
            and path.suffix not in (".safetensors", ".bin")
            and not path.name.endswith(".index.json")
        ):
            shutil.copy2(path, output_path / path.name)
    if not dequant:
        return
    # only workers that dequantized something strip quantization_config
    for directory in worker_dirs:
        config_path = directory / "config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text())
        if "quantization_config" not in config:
            shutil.copy2(config_path, output_path / "config.json")
            return


def merge_sharded_parallel(
    base_model_path: Path | str,
    lora_adapter_path: Path | str,
    output_path: Path | str,
    *,
    devices: Sequence[str],
    shard_sink: ShardSink,
    merge_report: dict | None = None,
    window: int | None = None,
    **merge_kwargs,
) -> None:
    """Merge with one worker per entry of ``devices``; ``shard_sink`` runs here."""
    from axolotl.cli.utils import lora_merge

    base_model_path, output_path = Path(base_model_path), Path(output_path)
    lora_adapter_path = Path(lora_adapter_path)
    shards = lora_merge.get_model_shards(base_model_path)
    if not shards:
        raise FileNotFoundError(f"No model shards found in {base_model_path}")
    groups = shard_groups(shards)
    devices = list(devices)[: len(groups)]
    if not devices:
        raise ValueError("Parallel merge needs at least one device")
    assigned: list[set[str]] = [set() for _ in devices]
    for number, group in enumerate(groups):
        assigned[number % len(devices)].update(shard.name for shard in group)
    weights_file = lora_adapter_path / "adapter_model.safetensors"
    lora_weights = (
        safetensors.torch.load_file(weights_file) if weights_file.exists() else None
    )
    ordered = _Ordered(shards, window or 2 * len(devices))
    reports: list[dict] = [{} for _ in devices]
    worker_dirs = [output_path / f".worker-{rank}" for rank in range(len(devices))]

    def run(rank: int):
        device = devices[rank]
        try:
            if torch.device(device).type == "cuda":
                torch.cuda.set_device(torch.device(device))
            lora_merge.merge_lora_sharded_efficient(
                base_model_path,
                lora_adapter_path,
                worker_dirs[rank],
                device=device,
                shard_sink=lambda tensors, destination, metadata: ordered.put(
                    destination.name,
                    (tensors, output_path / destination.name, metadata),
                ),
                merge_report=reports[rank],
                shard_filter=lambda shard: shard.name in assigned[rank],
                on_shard_merged=lambda shard: ordered.finish(shard.name),
                lora_weights=lora_weights,
                **merge_kwargs,
            )
        except BaseException as exc:
            ordered.fail(exc)

    threads = [
        threading.Thread(target=run, args=(rank,), name=f"merge-worker-{rank}")
        for rank in range(len(devices))
    ]
    for thread in threads:
        thread.start()
    try:
        ordered.drain(shard_sink, lambda: any(t.is_alive() for t in threads))
    except BaseException as exc:
        ordered.fail(exc)
        raise
    finally:
        for thread in threads:
            thread.join()
    if ordered.error is not None:
        raise ordered.error
    _publish_config(worker_dirs, output_path, bool(merge_kwargs.get("dequant")))
    for directory in worker_dirs:
        shutil.rmtree(directory, ignore_errors=True)
    report = _combine_reports(reports)
    if merge_report is not None:
        merge_report.update(report)
    if report["merged"] == 0:
        raise ValueError("No LoRA weights matched the base model")
