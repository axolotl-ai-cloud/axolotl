"""Publish immutable checkpoint artifacts before requesting a hot-load."""

import json
import shutil
import uuid
from pathlib import Path

from axolotl.utils.delta_checkpoint import snapshot_weights, write_checkpoint


class CheckpointPublisher:
    """Advance the delta base only after upload and replica convergence succeed."""

    def __init__(self, config, client, directory, s3=None):
        self.config, self.client = config, client
        self.directory = Path(directory)
        if s3 is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "Install axolotl[coreweave] to upload checkpoints"
                ) from exc

            s3 = boto3.Session(profile_name=config.s3_profile).client(
                "s3", endpoint_url=config.s3_endpoint_url
            )
        self.s3 = s3
        self.base = None
        self.identity = None
        self.delta_count = 0
        self.pending = None

    def publish(self, weights, model_config, step):
        if self.pending is not None:
            raise RuntimeError(
                "A checkpoint publication failed; recover it before publishing again"
            )
        snapshot = snapshot_weights(weights)
        identity = f"step-{step:08d}-{uuid.uuid4().hex}"
        full = self.base is None or self.delta_count >= self.config.max_deltas
        path = write_checkpoint(
            snapshot,
            self.directory / identity,
            identity=identity,
            config=model_config,
            base=None if full else self.base,
            base_identity=None if full else self.identity,
            max_shard_size=self.config.max_shard_size,
        )
        if not full:
            index = json.loads(
                (path / "model.safetensors.index.json").read_text(encoding="utf-8")
            )
            if not index["weight_map"]:
                shutil.rmtree(path)
                return self.identity
        self.pending = (identity, snapshot, full)
        self.client.ready_identity = None
        self.upload(path, identity)
        self.client.hot_load(identity, base_identity=None if full else self.identity)
        self._commit()
        return identity

    def recover(self, operation):
        """Confirm an existing/retried hot-load without issuing a duplicate request."""
        if self.pending is None:
            raise RuntimeError("No pending checkpoint to recover")
        self.client.wait_for_hot_load(operation, self.pending[0])
        self._commit()

    def _commit(self):
        self.identity, self.base, full = self.pending
        self.delta_count = 0 if full else self.delta_count + 1
        self.pending = None
        shutil.rmtree(self.directory / self.identity, ignore_errors=True)

    def upload(self, directory, identity):
        """Upload referenced shards first and the checkpoint index last."""
        from axolotl.utils.delta_checkpoint import validate_identity

        validate_identity(identity)
        directory = Path(directory)
        index = directory / "model.safetensors.index.json"
        manifest = json.loads(index.read_text(encoding="utf-8"))
        names = sorted(set(manifest["weight_map"].values()))
        if "delta_encoding" not in manifest["metadata"]:
            names.append("config.json")
        names.append(index.name)
        for name in names:
            if Path(name).name != name or not (directory / name).is_file():
                raise ValueError(f"Invalid checkpoint file: {name}")
        prefix = self.config.path_prefix.strip("/")
        for name in names:
            self.s3.upload_file(
                str(directory / name), self.config.bucket, f"{prefix}/{identity}/{name}"
            )
