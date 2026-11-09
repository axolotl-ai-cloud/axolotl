"""Time each stage of an offline checkpoint publish.

Synthetic mode (default, CPU only) builds BF16 tensors, perturbs a fraction of
them like a LoRA step would, and times the full write, the reference
single-threaded delta encoder, the chained delta write with the threaded
encoder, and a multipart upload into a local directory that stands in for S3.

With ``--base`` and ``--adapter`` it also times the sharded merge through
``MergedExporter``, with one worker per visible GPU (``--device cuda``) or
``--merge-workers`` simulated workers on CPU.

    python scripts/bench_publish.py --tensors 64 --tensor-mb 64
    python scripts/bench_publish.py --base /models/beam --adapter ckpt/ --device cuda
"""

import argparse
import json
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch


class DirectoryS3:
    """The subset of the S3 client API that ``S3Destination`` uses, on local disk."""

    def __init__(self, root: Path):
        self.root = root
        self.parts: dict[str, dict[int, bytes]] = {}
        self.lock = threading.Lock()

    def _path(self, bucket, key):
        path = self.root / bucket / key
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def upload_file(self, path, bucket, key):
        shutil.copyfile(path, self._path(bucket, key))

    def create_multipart_upload(self, Bucket, Key):  # noqa: N803
        with self.lock:
            upload_id = f"{Key}-{len(self.parts)}"
            self.parts[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ContentLength):  # noqa: N803
        with self.lock:
            self.parts[UploadId][PartNumber] = Body
        return {"ETag": f"{PartNumber}"}

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):  # noqa: N803
        parts = self.parts.pop(UploadId)
        with open(self._path(Bucket, Key), "wb") as handle:
            for part in MultipartUpload["Parts"]:
                handle.write(parts[part["PartNumber"]])

    def abort_multipart_upload(self, Bucket, Key, UploadId):  # noqa: N803
        self.parts.pop(UploadId, None)


class Timer:
    def __init__(self):
        self.stages: dict[str, dict] = {}

    @contextmanager
    def stage(self, name: str, nbytes: int = 0):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        self.stages[name] = {
            "seconds": round(seconds, 3),
            "GiB": round(nbytes / 2**30, 3),
            "GiB/s": round(nbytes / 2**30 / seconds, 3) if nbytes and seconds else None,
        }
        print(f"{name:<28} {seconds:9.3f} s  {self.stages[name]['GiB/s'] or '':>8}")


def synthetic(args, timer: Timer, workdir: Path):
    from axolotl.core.publishing import ChainWriter, S3Destination, destinations
    from axolotl.utils.delta_checkpoint import encode_xor_delta

    numel = (args.tensor_mb << 20) // 2
    generator = torch.Generator().manual_seed(0)
    with timer.stage("generate"):
        weights = {
            f"layers.{i}.weight": torch.randn(numel, generator=generator).to(
                torch.bfloat16
            )
            for i in range(args.tensors)
        }
    total = sum(t.numel() * 2 for t in weights.values())
    changed = max(1, int(args.tensors * args.changed_fraction))
    updated = dict(weights)
    for name in list(weights)[:changed]:
        updated[name] = (
            weights[name].float() + 1e-3 * torch.randn(numel, generator=generator)
        ).to(torch.bfloat16)
    changed_bytes = changed * numel * 2

    writer = ChainWriter(
        workdir / "chain",
        workers=args.encode_workers,
        zstd_threads=args.zstd_threads,
        large_tensor_bytes=args.large_tensor_mb << 20,
        max_shard_size=args.shard_mb << 20,
    )
    with timer.stage("full write", total):
        full = writer.write(weights, "full", config={"model_type": "bench"}, copy=False)
    writer.commit(full)

    def flat(tensor):
        return tensor.reshape(-1).view(torch.uint8).numpy()

    with timer.stage("reference encode (1 thread)", changed_bytes):
        sizes = [
            encoded[0].nbytes
            for name in list(weights)[:changed]
            if (encoded := encode_xor_delta(flat(updated[name]), flat(weights[name])))
        ]
    with timer.stage("delta write (threaded)", total):
        delta = writer.write(updated, "delta", copy=False)
    writer.close()
    print(
        f"{'delta size':<28} {delta.artifact_size / 2**20:9.1f} MiB "
        f"(reference {sum(sizes) / 2**20:.1f} MiB, {changed} changed tensors)"
    )

    destinations.MIN_PART_SIZE = min(destinations.MIN_PART_SIZE, args.part_mb << 20)
    destination = S3Destination(
        "bench",
        "run",
        client=DirectoryS3(workdir / "s3"),
        max_workers=args.upload_workers,
        multipart_threshold=args.part_mb << 20,
        multipart_chunksize=args.part_mb << 20,
    )
    with timer.stage("upload full (local S3)", full.artifact_size):
        destination.publish(full)
    with timer.stage("upload delta (local S3)", delta.artifact_size):
        destination.publish(delta)


def merge(args, timer: Timer, workdir: Path):
    from axolotl.core.publishing import MergedExporter, pin_base

    base = pin_base(Path(args.base), workdir)
    exporter = MergedExporter(
        base,
        args.merged_format,
        device=args.device,
        workdir=workdir,
        workers=args.merge_workers,
    )
    devices = exporter.devices
    print(f"merge workers: {len(devices)} on {', '.join(sorted(set(devices)))}")
    nbytes = 0

    def sink(name, tensor):
        nonlocal nbytes
        nbytes += tensor.numel() * tensor.element_size()

    with timer.stage("validate adapter"):
        exporter.validate(Path(args.adapter))
    with timer.stage("merge"):
        exporter.export(Path(args.adapter), sink)
    stage = timer.stages["merge"]
    stage["GiB"] = round(nbytes / 2**30, 3)
    stage["GiB/s"] = round(nbytes / 2**30 / stage["seconds"], 3)
    print(f"{'merged output':<28} {stage['GiB']:9.3f} GiB  {stage['GiB/s']:>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tensors", type=int, default=16)
    parser.add_argument("--tensor-mb", type=int, default=32)
    parser.add_argument("--changed-fraction", type=float, default=0.5)
    parser.add_argument("--encode-workers", type=int, default=None)
    parser.add_argument("--zstd-threads", type=int, default=None)
    parser.add_argument("--large-tensor-mb", type=int, default=32)
    parser.add_argument("--shard-mb", type=int, default=1024)
    parser.add_argument("--upload-workers", type=int, default=16)
    parser.add_argument("--part-mb", type=int, default=64)
    parser.add_argument("--base", help="base model directory, enables the merge stage")
    parser.add_argument("--adapter", help="adapter directory for the merge stage")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--merge-workers", type=int, default=None)
    parser.add_argument("--merged-format", default="preserve")
    parser.add_argument("--workdir", help="scratch directory (default: a temp dir)")
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--json", help="write the stage timings here")
    args = parser.parse_args()
    if bool(args.base) != bool(args.adapter):
        parser.error("--base and --adapter go together")

    workdir = Path(args.workdir or tempfile.mkdtemp(prefix="bench-publish-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(
        f"cuda: {torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"GPUs, workdir {workdir}"
    )
    timer = Timer()
    try:
        if not args.skip_synthetic:
            synthetic(args, timer, workdir / "synthetic")
        if args.base:
            merge(args, timer, workdir / "merge")
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    if args.json:
        Path(args.json).write_text(json.dumps(timer.stages, indent=2))


if __name__ == "__main__":
    main()
