"""Destinations, the commit-after-publish publisher, fingerprints, and config."""

import shutil
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from axolotl.core.publishing import (
    ChainPublisher,
    ChainWriter,
    Destination,
    LocalDestination,
    S3Destination,
    build_destination,
    merge_fingerprint,
)
from axolotl.core.publishing.fingerprint import fingerprint_digest
from axolotl.integrations.coreweave.args import CoreWeaveConfig
from axolotl.integrations.coreweave.destination import CoreWeaveDestination
from axolotl.integrations.coreweave.plugin import publish_config
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.checkpoint_publish import CheckpointPublishConfig
from axolotl.utils.schemas.config import AxolotlInputConfig


class FakeS3:
    """Copies uploads into a local directory tree keyed by bucket and key."""

    def __init__(self, root):
        self.root = Path(root)
        self.keys = []
        self.configs = []
        self.fail = None

    def upload_file(self, path, bucket, key, Config=None):  # noqa: N803
        if self.fail and self.fail in key:
            raise OSError(f"upload of {key} failed")
        target = self.root / bucket / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        self.keys.append(key)
        self.configs.append(Config)


def staged_pair(tmp_path, max_shard_size=16):
    writer = ChainWriter(tmp_path / "work", max_shard_size=max_shard_size)
    weights = {f"w{i}": torch.full((4,), float(i)) for i in range(4)}
    full = writer.write(weights, "full", config={"model_type": "test"})
    writer.commit(full)
    weights["w1"] += 1
    weights["w3"] += 1
    delta = writer.write(weights, "delta")
    return writer, full, delta


def test_s3_layout_is_flat_and_index_last(tmp_path):
    _, full, delta = staged_pair(tmp_path)
    s3 = FakeS3(tmp_path / "s3")
    destination = S3Destination.from_url(
        "s3://bucket/runs/a/", client=s3, max_workers=4
    )
    destination.publish(full)
    destination.publish(delta)
    full_keys = [k for k in s3.keys if "/full/" in k]
    delta_keys = [k for k in s3.keys if "/delta/" in k]
    assert full_keys[-1] == "runs/a/full/model.safetensors.index.json"
    assert delta_keys[-1] == "runs/a/delta/model.safetensors.index.json"
    assert "runs/a/full/config.json" in full_keys
    assert not any(k.endswith("config.json") for k in delta_keys)
    for key in s3.keys:
        prefix, identity, name = key.rsplit("/", 2)
        assert prefix == "runs/a" and identity in {"full", "delta"} and "/" not in name
    assert len(full_keys) == 4 + 2 and len(delta_keys) == 2 + 1
    assert sorted(p.name for p in (tmp_path / "s3/bucket/runs/a/delta").iterdir()) == (
        sorted(p.name for p in delta.path.iterdir())
    )


def test_s3_failure_does_not_upload_index(tmp_path):
    _, full, _ = staged_pair(tmp_path)
    s3 = FakeS3(tmp_path / "s3")
    s3.fail = "model-00002"
    with pytest.raises(OSError):
        S3Destination("bucket", "p", client=s3).publish(full)
    assert not any(k.endswith("index.json") for k in s3.keys)


def test_local_destination_and_builder(tmp_path):
    _, full, delta = staged_pair(tmp_path)
    destination = build_destination(str(tmp_path / "out"))
    assert isinstance(destination, LocalDestination)
    destination.publish(full)
    destination.publish(delta)
    destination.publish(delta)
    assert sorted(p.name for p in (tmp_path / "out" / "delta").iterdir()) == sorted(
        p.name for p in delta.path.iterdir()
    )
    assert not list((tmp_path / "out").glob(".incoming-*"))
    assert isinstance(build_destination("s3://b/p", client=MagicMock()), S3Destination)


class Flaky(Destination):
    def __init__(self):
        self.published, self.fail = [], False

    def publish(self, staged):
        if self.fail:
            raise OSError("destination down")
        self.published.append(staged.identity)


def test_publisher_commits_only_after_destination_success(tmp_path):
    destination = Flaky()
    publisher = ChainPublisher(ChainWriter(tmp_path), destination)
    assert publisher.publish({"w": torch.ones(2)}, "a", config={}) == "a"
    destination.fail = True
    with pytest.raises(OSError):
        publisher.publish({"w": torch.zeros(2)}, "b")
    assert publisher.identity == "a"
    assert torch.equal(publisher.baseline["w"], torch.ones(2))
    with pytest.raises(RuntimeError, match="recover"):
        publisher.publish({"w": torch.zeros(2)}, "c")
    publisher.abandon()
    destination.fail = False
    assert publisher.publish({"w": torch.full((2,), 2.0)}, "c") == "c"
    assert publisher.delta_count == 1
    assert not (tmp_path / "b").exists()
    destination.fail = True
    with pytest.raises(OSError):
        publisher.publish({"w": torch.full((2,), 3.0)}, "d")
    destination.fail = False
    publisher.recover()
    assert publisher.identity == "d" and destination.published == ["a", "c", "d"]
    assert publisher.publish({"w": torch.full((2,), 3.0)}, "e") == "d"


def test_coreweave_destination_caps_chain_and_hot_loads(tmp_path):
    config = CoreWeaveConfig(
        deployment_id="d",
        bucket="bucket",
        path_prefix="weights",
        gateway_url="https://gateway.example",
        model="m",
    )
    client = MagicMock()
    s3 = FakeS3(tmp_path / "s3")
    destination = CoreWeaveDestination(config, client, s3)
    assert destination.max_deltas == 63
    publisher = ChainPublisher(ChainWriter(tmp_path / "work"), destination)
    publisher.publish({"w": torch.ones(2)}, "a", config={})
    publisher.publish({"w": torch.zeros(2)}, "b")
    assert [c.args[0] for c in client.hot_load.call_args_list] == ["a", "b"]
    assert [c.kwargs["base_identity"] for c in client.hot_load.call_args_list] == [
        None,
        "a",
    ]
    assert s3.keys[-1] == "weights/b/model.safetensors.index.json"
    client.hot_load.side_effect = TimeoutError("replicas")
    with pytest.raises(TimeoutError):
        publisher.publish({"w": torch.ones(2)}, "c")
    publisher.recover("operation-1")
    client.wait_for_hot_load.assert_called_once_with("operation-1", "c")
    assert publisher.identity == "c"


def test_fingerprint_records_eager_and_live_tf32():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        eager = merge_fingerprint(export="merged", base_identity="base", device="cpu")
        assert eager["merge_compile"] == "eager"
        assert eager["float32_precision"]["matmul_precision"] == "highest"
        assert eager["merge_code"] and eager["torch"] == torch.__version__
        torch.set_float32_matmul_precision("high")
        tf32 = merge_fingerprint(export="merged", base_identity="base", device="cpu")
    finally:
        torch.set_float32_matmul_precision(previous)
    assert fingerprint_digest(eager) != fingerprint_digest(tf32)
    strict = merge_fingerprint(
        export="merged",
        base_identity="base",
        merge_compile=True,
        merge_compile_precision="strict",
    )
    assert strict["merge_compile"] == "compiled-strict"
    other_base = merge_fingerprint(export="merged", base_identity="other", device="cpu")
    assert fingerprint_digest(other_base) != fingerprint_digest(eager)
    lora = merge_fingerprint(export="lora", base_identity="base")
    assert "merge_code" not in lora


def test_checkpoint_publish_schema():
    settings = CheckpointPublishConfig()
    assert settings.max_deltas == 63 and settings.merged_format == "preserve"
    assert settings.every == 0 and settings.delta
    for bad in (
        {"max_deltas": 64},
        {"export": "lora", "merged_format": "bf16"},
        {"export": "lora", "merge_compile": True},
        {"export": "tensors"},
        {"merged_format": "nvfp4"},
        {"merge_compile_precision": "fast"},
        {"every": -1},
        {"destination": "s3:///prefix"},
        {"unknown": 1},
    ):
        with pytest.raises(ValidationError):
            CheckpointPublishConfig(**bad)
    assert CheckpointPublishConfig(export="lora").merged_format is None
    assert "checkpoint_publish" in AxolotlInputConfig.model_fields


def coreweave(**kwargs):
    return CoreWeaveConfig(
        deployment_id="d",
        bucket="b",
        path_prefix="p",
        gateway_url="https://g",
        model="m",
        **kwargs,
    )


def test_coreweave_max_deltas_deprecation():
    assert coreweave().max_deltas == 63
    with pytest.warns(DeprecationWarning, match="capped to 63"):
        assert coreweave(max_deltas=128).max_deltas == 63
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert coreweave(max_deltas=10).max_deltas == 10
    with pytest.raises(ValidationError):
        coreweave(max_deltas=129)


def test_checkpoint_publish_applies_to_coreweave():
    config = coreweave()
    cfg = DictDefault({"adapter": "lora", "checkpoint_publish": {"max_deltas": 5}})
    capped, settings = publish_config(cfg, config)
    assert capped.max_deltas == 5 and settings.export == "merged"
    cfg = DictDefault({"checkpoint_publish": {"delta": False}})
    assert publish_config(cfg, config)[0].max_deltas == 0
    assert publish_config(DictDefault({}), config) == (config, None)
    for bad, match in (
        ({"destination": "s3://b/p"}, "destination"),
        ({"export": "lora"}, "full model weights"),
        ({"every": 1, "merged_format": "bf16"}, "merged_format: preserve"),
        ({"merge_compile": True}, "merge_compile"),
    ):
        with pytest.raises(ValueError, match=match):
            publish_config(
                DictDefault({"adapter": "lora", "checkpoint_publish": bad}), config
            )
    _, settings = publish_config(
        DictDefault({"adapter": "lora", "checkpoint_publish": {"every": 2}}), config
    )
    assert settings.every == 2
    with pytest.raises(ValueError, match="adapter: lora"):
        publish_config(DictDefault({"checkpoint_publish": {"every": 1}}), config)
    with pytest.raises(ValueError, match="nvfp4_merge_aware_start_step"):
        publish_config(
            DictDefault(
                {
                    "adapter": "lora",
                    "nvfp4_merge_aware": True,
                    "nvfp4_merge_aware_start_step": 10,
                    "checkpoint_publish": {"every": 1},
                }
            ),
            config,
        )
