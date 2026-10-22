"""Offline throughput paths: parallel sharded merge, chunked encoder, multipart S3."""

import hashlib
import json
import threading
import time
import zlib
from pathlib import Path

import numpy as np
import pytest
import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import save_file

from axolotl.cli.utils import lora_merge
from axolotl.core.publishing import (
    ChainPublisher,
    ChainWriter,
    LocalDestination,
    MergedExporter,
    S3Destination,
    destinations,
    pin_base,
)
from axolotl.core.publishing.encoder import encode_xor_delta_chunked
from axolotl.core.publishing.parallel_merge import shard_groups, visible_devices
from axolotl.utils.delta_checkpoint import encode_xor_delta

from tests.core.publishing.nvfp4_moe import (
    INDEX,
    apply_delta,
    assert_bytes_equal,
    build_base,
    load_checkpoint,
    save_adapter,
)


def reshard(source: Path, target: Path) -> Path:
    """Five shards in three groups: layers 0 and 2 split in two, layer 1 alone."""
    weights = load_checkpoint(source)

    def layer(key):
        return int(key.split(".")[2]) if key.startswith("model.layers.") else None

    def halves(number):
        keys = sorted(k for k in weights if layer(k) == number)
        return keys[: len(keys) // 2], keys[len(keys) // 2 :]

    first, second = halves(0)
    third, fourth = halves(2)
    shards = [
        ["lm_head.weight", "model.embed_tokens.weight", *first],
        second,
        sorted(k for k in weights if layer(k) == 1),
        third,
        [*fourth, "model.norm.weight"],
    ]
    assert sorted(k for shard in shards for k in shard) == sorted(weights)
    target.mkdir(parents=True)
    weight_map = {}
    for number, keys in enumerate(shards):
        name = f"model-{number + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({k: weights[k] for k in keys}, target / name, {"format": "pt"})
        weight_map.update(dict.fromkeys(keys, name))
    (target / INDEX).write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (target / "config.json").write_text((source / "config.json").read_text())
    return target


@pytest.fixture
def sharded_base(tmp_path):
    pytest.importorskip("torchao")
    return reshard(build_base(tmp_path / "raw"), tmp_path / "base")


def artifact_contents(path: Path) -> dict:
    """Every file's tensors as raw bytes plus its metadata.

    safetensors writes ``__metadata__`` in hash-map order, so delta shard headers
    are not byte-stable across writes; everything else is compared exactly.
    """
    contents: dict[str, object] = {}
    for file in sorted(path.iterdir()):
        if file.suffix != ".safetensors":
            contents[file.name] = file.read_bytes()
            continue
        with safe_open(file, framework="pt") as shard:
            contents[file.name] = (
                shard.metadata(),
                {
                    key: (
                        str(tensor.dtype),
                        tuple(tensor.shape),
                        tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
                    )
                    for key in shard.keys()
                    for tensor in [shard.get_tensor(key)]
                },
                list(shard.keys()),
            )
    return contents


def test_shard_groups_keep_layers_together(sharded_base):
    shards = lora_merge.get_model_shards(sharded_base)
    groups = shard_groups(shards)
    assert [s for group in groups for s in group] == shards
    assert [len(group) for group in groups] == [2, 1, 2]
    owner = {s.name: number for number, group in enumerate(groups) for s in group}
    index = json.loads((sharded_base / INDEX).read_text())["weight_map"]
    for key, name in index.items():
        if ".layers." in key:
            layer = key.split(".layers.")[1].split(".")[0]
            first = next(
                n for k, n in index.items() if k.startswith(f"model.layers.{layer}.")
            )
            assert owner[name] == owner[first], key


def test_visible_devices():
    assert visible_devices("cpu", 3) == ["cpu"] * 3
    assert visible_devices("cpu") == ["cpu"]
    with pytest.raises(ValueError, match="one device"):
        visible_devices("cuda:0", 2)


def publish_chain(tmp_path, base_dir, workers, steps=(1, 2, 3), merge_aware=False):
    work = tmp_path / f"work-{workers}"
    publisher = ChainPublisher(
        ChainWriter(work, max_shard_size=48 << 10),
        LocalDestination(tmp_path / f"published-{workers}"),
    )
    exporter = MergedExporter(pin_base(base_dir, work), workdir=work, workers=workers)
    identities = []
    for step in steps:
        adapter = save_adapter(
            tmp_path / f"adapter-{step}", step, merge_aware=merge_aware
        )
        identities.append(publisher.publish_export(exporter, adapter, f"step-{step}"))
    return tmp_path / f"published-{workers}", identities, work


@pytest.mark.parametrize("merge_aware", [False, True], ids=["reuse", "fresh"])
def test_parallel_merge_matches_single_worker(tmp_path, sharded_base, merge_aware):
    single, identities, _ = publish_chain(
        tmp_path, sharded_base, 1, merge_aware=merge_aware
    )
    for workers in (2, 3):
        parallel, again, work = publish_chain(
            tmp_path, sharded_base, workers, merge_aware=merge_aware
        )
        assert again == identities
        for identity in identities:
            assert artifact_contents(parallel / identity) == (
                artifact_contents(single / identity)
            )
        assert not list(work.glob(".merge-*"))
    previous = load_checkpoint(single / identities[0])
    expected = tmp_path / "expected-1"
    lora_merge.merge_lora_sharded_efficient(
        sharded_base, tmp_path / "adapter-1", expected
    )
    assert_bytes_equal(previous, load_checkpoint(expected))
    for step, identity in zip((2, 3), identities[1:], strict=True):
        previous = apply_delta(single / identity, previous)
        expected = tmp_path / f"expected-{step}"
        lora_merge.merge_lora_sharded_efficient(
            sharded_base, tmp_path / f"adapter-{step}", expected
        )
        assert_bytes_equal(previous, load_checkpoint(expected))


def test_parallel_merge_worker_failure_publishes_nothing(
    tmp_path, sharded_base, monkeypatch
):
    real = lora_merge.merge_lora_sharded_efficient
    calls = []

    def failing(*args, shard_filter, **kwargs):
        calls.append(shard_filter)
        if len(calls) == 2:
            raise RuntimeError("worker died")
        return real(*args, shard_filter=shard_filter, **kwargs)

    monkeypatch.setattr(lora_merge, "merge_lora_sharded_efficient", failing)
    work = tmp_path / "work"
    publisher = ChainPublisher(
        ChainWriter(work), LocalDestination(tmp_path / "published")
    )
    exporter = MergedExporter(pin_base(sharded_base, work), workdir=work, workers=3)
    with pytest.raises(RuntimeError, match="worker died"):
        publisher.publish_export(exporter, save_adapter(tmp_path / "a", 1), "step-1")
    assert publisher.baseline is None and publisher.pending is None
    assert not list(work.glob(".staging-*")) and not list(work.glob(".merge-*"))
    assert not (tmp_path / "published").exists()


def large_pair(size=40 << 20, *, changed=((30 << 20, 1 << 20),)):
    rng = np.random.default_rng(0)
    previous = rng.integers(0, 255, size=size, dtype=np.uint8)
    current = previous.copy()
    for start, length in changed:
        current[start : start + length] ^= rng.integers(
            1, 255, size=length, dtype=np.uint8
        )
    return current, previous


def decode(frame: np.ndarray, previous: np.ndarray) -> np.ndarray:
    delta = zstandard.ZstdDecompressor().decompress(frame.tobytes())
    return np.bitwise_xor(np.frombuffer(delta, dtype=np.uint8), previous)


def test_chunked_encoder_on_large_tensor():
    current, previous = large_pair()
    frames = {}
    for threads, chunk in ((1, 4 << 20), (4, 16 << 20), (8, 3 << 20)):
        frame, checksum = encode_xor_delta_chunked(
            current, previous, threads=threads, chunk_bytes=chunk
        )
        assert np.array_equal(decode(frame, previous), current)
        assert checksum == f"{zlib.adler32(current):08x}"
        assert zstandard.get_frame_parameters(frame.tobytes()).content_size == (
            current.nbytes
        )
        frames[threads, chunk] = frame.tobytes()
    assert len(set(frames.values())) == 1
    reference, reference_checksum = encode_xor_delta(current, previous)
    assert checksum == reference_checksum
    assert np.array_equal(decode(reference, previous), current)
    assert len(frame) < current.nbytes // 10
    assert encode_xor_delta_chunked(previous.copy(), previous) is None


def test_chunked_encoder_change_in_first_and_last_byte():
    current, previous = large_pair(size=(5 << 20) + 3, changed=())
    current[0] ^= 1
    current[-1] ^= 1
    frame, _ = encode_xor_delta_chunked(current, previous, chunk_bytes=1 << 20)
    assert np.array_equal(decode(frame, previous), current)
    with pytest.raises(ValueError, match="same size"):
        encode_xor_delta_chunked(current, previous[:-1])


def test_chain_uses_chunked_encoder_for_large_tensors(tmp_path):
    current, previous = large_pair(size=6 << 20, changed=((5 << 20, 4096),))
    as_tensor = lambda a: torch.from_numpy(a.copy()).view(torch.bfloat16)  # noqa: E731
    small = torch.arange(64, dtype=torch.float32)
    writer = ChainWriter(tmp_path / "work", large_tensor_bytes=1 << 20, zstd_threads=4)
    full = writer.write(
        {"big": as_tensor(previous), "small": small}, "full", config={"m": 1}
    )
    writer.commit(full)
    delta = writer.write({"big": as_tensor(current), "small": small + 1}, "delta")
    rebuilt = apply_delta(delta.path, load_checkpoint(full.path))
    assert_bytes_equal(rebuilt, {"big": as_tensor(current), "small": small + 1})
    writer.close()


class FakeMultipartS3:
    """Fake S3 with the multipart API; records order and part concurrency."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}
        self.uploads: dict[str, dict[int, bytes]] = {}
        self.order: list[str] = []
        self.aborted: list[str] = []
        self.fail_part: int | None = None
        self.lock = threading.Lock()
        self.active = self.peak = 0

    def upload_file(self, path, bucket, key):
        with self.lock:
            self.objects[key] = Path(path).read_bytes()
            self.order.append(key)

    def create_multipart_upload(self, Bucket, Key):  # noqa: N803
        upload_id = f"{Key}#{len(self.uploads)}"
        self.uploads[upload_id] = {}
        return {"UploadId": upload_id}

    def upload_part(self, Bucket, Key, UploadId, PartNumber, Body, ContentLength):  # noqa: N803
        assert len(Body) == ContentLength
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.01)
            if PartNumber == self.fail_part:
                raise OSError("part upload failed")
            with self.lock:
                self.uploads[UploadId][PartNumber] = Body
            return {"ETag": hashlib.md5(Body, usedforsecurity=False).hexdigest()}
        finally:
            with self.lock:
                self.active -= 1

    def complete_multipart_upload(self, Bucket, Key, UploadId, MultipartUpload):  # noqa: N803
        parts = MultipartUpload["Parts"]
        assert [p["PartNumber"] for p in parts] == list(range(1, len(parts) + 1))
        stored = self.uploads.pop(UploadId)
        for part in parts:
            expected = hashlib.md5(stored[part["PartNumber"]], usedforsecurity=False)
            assert part["ETag"] == expected.hexdigest()
        with self.lock:
            self.objects[Key] = b"".join(stored[p["PartNumber"]] for p in parts)
            self.order.append(Key)

    def abort_multipart_upload(self, Bucket, Key, UploadId):  # noqa: N803
        self.uploads.pop(UploadId, None)
        self.aborted.append(UploadId)


@pytest.fixture
def staged_large(tmp_path):
    rng = np.random.default_rng(1)
    weights = {
        f"w{i}": torch.from_numpy(rng.integers(0, 255, size=(300_000,), dtype=np.uint8))
        for i in range(3)
    }
    weights["tiny"] = torch.zeros(4)
    writer = ChainWriter(tmp_path / "work", max_shard_size=350_000)
    return writer.write(weights, "full", config={"model_type": "test"})


def s3_destination(client, monkeypatch, **kwargs):
    monkeypatch.setattr(destinations, "MIN_PART_SIZE", 1)
    return S3Destination(
        "bucket",
        "runs",
        client=client,
        multipart_threshold=100_000,
        multipart_chunksize=64_000,
        **kwargs,
    )


def test_parallel_multipart_upload(staged_large, monkeypatch):
    s3 = FakeMultipartS3()
    s3_destination(s3, monkeypatch, max_workers=8).publish(staged_large)
    names = sorted(p.name for p in staged_large.path.iterdir())
    assert sorted(k.rsplit("/", 1)[1] for k in s3.objects) == names
    for name in names:
        assert (
            s3.objects[f"runs/full/{name}"] == (staged_large.path / name).read_bytes()
        )
    assert s3.order[-1] == f"runs/full/{INDEX}"
    assert s3.peak > 1 and not s3.uploads and not s3.aborted


def test_multipart_failure_aborts_and_skips_index(staged_large, monkeypatch):
    s3 = FakeMultipartS3()
    s3.fail_part = 3
    with pytest.raises(OSError, match="part upload failed"):
        s3_destination(s3, monkeypatch, max_workers=4).publish(staged_large)
    assert f"runs/full/{INDEX}" not in s3.objects
    assert s3.aborted and not s3.uploads


def test_part_size_respects_part_limit(monkeypatch):
    destination = s3_destination(FakeMultipartS3(), monkeypatch)
    assert destination.part_size(10) == 64_000
    huge = 64_000 * destinations.MAX_PARTS * 3
    assert -(-huge // destination.part_size(huge)) <= destinations.MAX_PARTS
    monkeypatch.setattr(destinations, "MIN_PART_SIZE", 5 << 20)
    with pytest.raises(ValueError, match="at least"):
        S3Destination("bucket", client=FakeMultipartS3(), multipart_chunksize=1 << 20)
