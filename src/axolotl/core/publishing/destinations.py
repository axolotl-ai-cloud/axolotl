"""Where staged checkpoints go: a local directory, S3, or a hot-load service."""

import json
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from axolotl.utils.delta_checkpoint import validate_identity

from .chain import INDEX_NAME, StagedCheckpoint, _link


def checkpoint_files(directory: Path | str) -> list[str]:
    """Return the flat file list of an artifact, ordered shards, config, index."""
    directory = Path(directory)
    manifest = json.loads((directory / INDEX_NAME).read_text(encoding="utf-8"))
    names = sorted(set(manifest["weight_map"].values()))
    if "delta_encoding" not in manifest["metadata"]:
        names.append("config.json")
    names.append(INDEX_NAME)
    for name in names:
        if Path(name).name != name or not (directory / name).is_file():
            raise ValueError(f"Invalid checkpoint file: {name}")
    return names


class Destination:
    """Publish an artifact; raising means the checkpoint must not be committed."""

    max_deltas: int | None = None
    exports: tuple[str, ...] = ("lora", "merged")

    @classmethod
    def check_export(cls, export: str) -> None:
        if export not in cls.exports:
            raise ValueError(
                f"{cls.__name__} does not accept export: {export}; use one of "
                f"{', '.join(cls.exports)}"
            )

    def publish(self, staged: StagedCheckpoint) -> None:
        raise NotImplementedError

    def recover(self, staged: StagedCheckpoint, operation: Any = None) -> None:
        """Finish a publish that failed or whose outcome is unknown."""
        self.publish(staged)

    def publish_files(self, directory: Path, identity: str, names: list[str]) -> None:
        """Upload one writer's shard files of a checkpoint assembled by several ranks."""
        raise NotImplementedError(f"{type(self).__name__} cannot publish per rank")

    def publish_index(
        self,
        directory: Path,
        identity: str,
        *,
        base_identity: str | None,
        full: bool,
    ) -> None:
        """Publish the combined index (and config) after every rank's files are up."""
        raise NotImplementedError(f"{type(self).__name__} cannot publish per rank")

    def discard_files(self, identity: str) -> None:
        """Forget files of a per-rank publish that will not be committed."""


def index_files(directory: Path, full: bool) -> list[str]:
    """The files rank 0 publishes last: config for a full checkpoint, then the index."""
    names = ["config.json", INDEX_NAME] if full else [INDEX_NAME]
    for name in names:
        if not (Path(directory) / name).is_file():
            raise ValueError(f"Missing checkpoint file: {name}")
    return names


class LocalDestination(Destination):
    """Copy (or hard-link) each artifact to ``root/<identity>/``."""

    def __init__(self, root: Path | str):
        self.root = Path(root)

    def publish(self, staged: StagedCheckpoint) -> None:
        validate_identity(staged.identity)
        target = self.root / staged.identity
        names = checkpoint_files(staged.path)
        if target.exists():
            if sorted(p.name for p in target.iterdir()) == sorted(names):
                return
            raise FileExistsError(target)
        self.root.mkdir(parents=True, exist_ok=True)
        incoming = Path(tempfile.mkdtemp(prefix=".incoming-", dir=self.root))
        try:
            for name in names:
                _link(staged.path / name, incoming / name)
            incoming.rename(target)
        finally:
            if incoming.exists():
                shutil.rmtree(incoming)

    def _incoming(self, identity: str) -> Path:
        return self.root / f".incoming-{validate_identity(identity)}"

    def publish_files(self, directory, identity, names):
        incoming = self._incoming(identity)
        incoming.mkdir(parents=True, exist_ok=True)
        for name in names:
            if Path(name).name != name:
                raise ValueError(f"Invalid checkpoint file: {name}")
            _link(Path(directory) / name, incoming / name)

    def publish_index(self, directory, identity, *, base_identity, full):
        target = self.root / validate_identity(identity)
        if target.exists():
            raise FileExistsError(target)
        incoming = self._incoming(identity)
        incoming.mkdir(parents=True, exist_ok=True)
        for name in index_files(directory, full):
            _link(Path(directory) / name, incoming / name)
        checkpoint_files(incoming)
        incoming.rename(target)

    def discard_files(self, identity):
        shutil.rmtree(self._incoming(identity), ignore_errors=True)


MIN_PART_SIZE = 5 << 20
MAX_PARTS = 10_000


class S3Destination(Destination):
    """Upload flat under ``s3://bucket/prefix/<identity>/``; the index goes last.

    Files at or above ``multipart_threshold`` go up as multipart uploads whose
    parts share one pool of ``max_workers`` threads with the small files, so a
    delta made of a few large shards still uses every connection. A failed
    multipart upload is aborted so no orphaned parts are left in the bucket.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client=None,
        profile: str | None = None,
        endpoint_url: str | None = None,
        max_workers: int = 16,
        multipart_threshold: int = 64 << 20,
        multipart_chunksize: int = 64 << 20,
    ):
        if not bucket:
            raise ValueError("S3 destination needs a bucket")
        if multipart_chunksize < MIN_PART_SIZE:
            raise ValueError(f"multipart_chunksize must be at least {MIN_PART_SIZE}")
        self.bucket, self.prefix = bucket, prefix.strip("/")
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError("Install boto3 to publish checkpoints to S3") from exc
            client = boto3.Session(profile_name=profile).client(
                "s3", endpoint_url=endpoint_url
            )
        self.client = client
        self.max_workers = max(1, max_workers)
        self.multipart_threshold = multipart_threshold
        self.multipart_chunksize = multipart_chunksize

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "S3Destination":
        if not url.startswith("s3://"):
            raise ValueError(f"Not an S3 URL: {url}")
        bucket, _, prefix = url[len("s3://") :].partition("/")
        return cls(bucket, prefix, **kwargs)

    def key(self, identity: str, name: str) -> str:
        validate_identity(identity)
        return "/".join(part for part in (self.prefix, identity, name) if part)

    def _upload(self, path: Path, key: str):
        self.client.upload_file(str(path), self.bucket, key)

    def part_size(self, size: int) -> int:
        return max(self.multipart_chunksize, -(-size // MAX_PARTS))

    def _upload_part(
        self, path: Path, key: str, upload_id: str, number: int, offset, length
    ):
        with open(path, "rb") as handle:
            handle.seek(offset)
            body = handle.read(length)
        if len(body) != length:
            raise OSError(f"{path} changed size during upload")
        response = self.client.upload_part(
            Bucket=self.bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=number,
            Body=body,
            ContentLength=length,
        )
        return {"PartNumber": number, "ETag": response["ETag"]}

    def upload(self, directory: Path | str, identity: str) -> None:
        directory = Path(directory)
        *shards, index = checkpoint_files(directory)
        self.upload_files(directory, identity, shards)
        self._upload(directory / index, self.key(identity, index))

    def publish_files(self, directory, identity, names):
        self.upload_files(Path(directory), identity, names)

    def publish_index(self, directory, identity, *, base_identity, full):
        directory = Path(directory)
        *config, index = index_files(directory, full)
        self.upload_files(directory, identity, config)
        self._upload(directory / index, self.key(identity, index))

    def upload_files(self, directory: Path, identity: str, names: list[str]) -> None:
        """Upload ``names`` in parallel; multipart uploads are aborted on failure."""
        uploads: list[tuple[str, str, list[Future]]] = []
        futures: list[Future] = []
        with ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="s3-upload"
        ) as pool:
            try:
                for name in names:
                    if Path(name).name != name:
                        raise ValueError(f"Invalid checkpoint file: {name}")
                    path, key = directory / name, self.key(identity, name)
                    size = path.stat().st_size
                    if size < self.multipart_threshold:
                        futures.append(pool.submit(self._upload, path, key))
                        continue
                    upload_id = self.client.create_multipart_upload(
                        Bucket=self.bucket, Key=key
                    )["UploadId"]
                    parts: list[Future] = []
                    uploads.append((key, upload_id, parts))
                    step = self.part_size(size)
                    for number, offset in enumerate(range(0, size, step), start=1):
                        parts.append(
                            pool.submit(
                                self._upload_part,
                                path,
                                key,
                                upload_id,
                                number,
                                offset,
                                min(step, size - offset),
                            )
                        )
                for future in futures:
                    future.result()
                for key, upload_id, parts in uploads:
                    self.client.complete_multipart_upload(
                        Bucket=self.bucket,
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": [part.result() for part in parts]},
                    )
            except BaseException:
                pending = futures + [p for *_, parts in uploads for p in parts]
                for future in pending:
                    future.cancel()
                wait(pending)
                for key, upload_id, _ in uploads:
                    try:
                        self.client.abort_multipart_upload(
                            Bucket=self.bucket, Key=key, UploadId=upload_id
                        )
                    except Exception:  # noqa: BLE001  # nosec B110
                        pass
                raise

    def publish(self, staged: StagedCheckpoint) -> None:
        self.upload(staged.path, staged.identity)


def build_destination(spec: str, **s3_options) -> Destination:
    """``s3://bucket/prefix`` or a local directory path."""
    if spec.startswith("s3://"):
        return S3Destination.from_url(spec, **s3_options)
    return LocalDestination(spec)
