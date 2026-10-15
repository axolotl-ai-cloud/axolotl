"""Byte-level interoperability and publication failure coverage."""

import json
import zlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import load_file

from axolotl.integrations.coreweave.args import CoreWeaveConfig
from axolotl.integrations.coreweave.checkpoints import CheckpointPublisher
from axolotl.utils.delta_checkpoint import snapshot_weights, write_checkpoint


def config(**kwargs):
    return CoreWeaveConfig(
        deployment_id="deployment",
        bucket="bucket",
        path_prefix="weights",
        gateway_url="https://gateway.example",
        model="policy",
        **kwargs,
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.uint8]
)
def test_delta_reconstructs_exact_bytes(tmp_path, dtype):
    old = {"weight": torch.arange(16).to(dtype), "scalar": torch.tensor(2, dtype=dtype)}
    new = snapshot_weights(old)
    new["weight"][3] = 7
    path = write_checkpoint(
        new, tmp_path / "delta", identity="next", base=old, base_identity="base"
    )
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {"weight"}
    assert index["metadata"] == {
        "version": "next",
        "base_version": "base",
        "delta_encoding": "xor",
        "compression_format": "zstd",
        "checksum_format": "adler32",
    }
    with safe_open(path / index["weight_map"]["weight"], framework="numpy") as shard:
        encoded = shard.get_tensor("weight")
        delta = np.frombuffer(
            zstandard.ZstdDecompressor().decompress(encoded), dtype=np.uint8
        )
        result = np.bitwise_xor(old["weight"].view(torch.uint8).numpy(), delta)
        np.testing.assert_array_equal(result, new["weight"].view(torch.uint8).numpy())
        assert shard.metadata()["weight"] == f"{zlib.adler32(result):08x}"


def test_full_checkpoint_clones_aliases_and_indexes_all_shards(tmp_path):
    weight = torch.arange(4)
    original = {"a": weight, "b": weight, "scalar": torch.tensor(4)}
    snapshot = snapshot_weights(original)
    weight.zero_()
    path = write_checkpoint(
        snapshot,
        tmp_path / "full",
        identity="base",
        config={"model_type": "test"},
        max_shard_size=32,
    )
    index = json.loads((path / "model.safetensors.index.json").read_text())
    assert len(set(index["weight_map"].values())) == 3
    assert index["metadata"]["total_size"] == 72
    for name, filename in index["weight_map"].items():
        assert torch.equal(load_file(path / filename)[name], snapshot[name])
    assert snapshot["a"].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "new",
    [
        {"other": torch.ones(2)},
        {"a": torch.ones(3)},
        {"a": torch.ones(2, dtype=torch.float64)},
    ],
)
def test_layout_change_is_atomic(tmp_path, new):
    with pytest.raises(ValueError):
        write_checkpoint(
            new,
            tmp_path / "delta",
            identity="next",
            base={"a": torch.ones(2)},
            base_identity="base",
        )
    assert list(tmp_path.iterdir()) == []


def test_no_change_delta_and_immutable_destination(tmp_path):
    weights = {"a": torch.ones(2)}
    path = write_checkpoint(
        weights, tmp_path / "delta", identity="next", base=weights, base_identity="base"
    )
    assert (
        json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        == {}
    )
    with pytest.raises(FileExistsError):
        write_checkpoint(weights, path, identity="next", config={})


def test_publisher_chain_rollover_and_upload_order(tmp_path):
    s3, client = MagicMock(), MagicMock()
    publisher = CheckpointPublisher(config(max_deltas=1), client, tmp_path, s3)
    weights = {"weight": torch.ones(2)}
    first = publisher.publish(weights, {}, 0)
    weights["weight"][0] = 2
    second = publisher.publish(weights, {}, 1)
    third = publisher.publish(weights, {}, 2)
    assert [
        call.kwargs["base_identity"] for call in client.hot_load.call_args_list
    ] == [None, first, None]
    assert publisher.identity == third and publisher.delta_count == 0
    assert list(tmp_path.iterdir()) == []
    keys = [call.args[2] for call in s3.upload_file.call_args_list]
    for identity in (first, second, third):
        assert [key for key in keys if identity in key][-1].endswith(
            "model.safetensors.index.json"
        )


def test_unchanged_weights_do_not_submit_empty_delta(tmp_path):
    client, s3 = MagicMock(), MagicMock()
    publisher = CheckpointPublisher(config(), client, tmp_path, s3)
    weights = {"weight": torch.ones(2)}
    identity = publisher.publish(weights, {}, 0)
    s3.upload_file.reset_mock()
    assert publisher.publish(weights, {}, 1) == identity
    s3.upload_file.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    client.hot_load.assert_called_once()
    assert publisher.delta_count == 0


@pytest.mark.parametrize("failure", ["upload", "hot_load"])
def test_failure_does_not_advance_base_or_allow_more_publications(tmp_path, failure):
    s3, client = MagicMock(), MagicMock()
    publisher = CheckpointPublisher(config(), client, tmp_path, s3)
    first = publisher.publish({"a": torch.ones(2)}, {}, 0)
    if failure == "upload":
        s3.upload_file.side_effect = OSError("upload failed")
    else:
        client.hot_load.side_effect = OSError("reply lost")
    with pytest.raises(OSError):
        publisher.publish({"a": torch.zeros(2)}, {}, 1)
    assert publisher.identity == first
    assert torch.equal(publisher.base["a"], torch.ones(2))
    with pytest.raises(RuntimeError, match="recover"):
        publisher.publish({"a": torch.zeros(2)}, {}, 2)
    assert (tmp_path / publisher.pending[0]).is_dir()
    publisher.recover("confirmed-operation")
    assert torch.equal(publisher.base["a"], torch.zeros(2))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("identity", ["../bad", "a/b", "", ".", ".."])
def test_identity_validation(tmp_path, identity):
    with pytest.raises(ValueError, match="path segment"):
        write_checkpoint(
            {"a": torch.ones(1)}, tmp_path / "out", identity=identity, config={}
        )
