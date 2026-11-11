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
    assert publisher.publish({"w": torch.full((2,), 2.0)}, "c", config={}) == "c"
    # readers may hold part of the abandoned publish, so it was not a delta
    assert publisher.delta_count == 0
    assert publisher.publish({"w": torch.full((2,), 4.0)}, "c2") == "c2"
    assert publisher.delta_count == 1
    assert not (tmp_path / "b").exists()
    destination.fail = True
    with pytest.raises(OSError):
        publisher.publish({"w": torch.full((2,), 3.0)}, "d")
    destination.fail = False
    publisher.recover()
    assert publisher.identity == "d"
    assert destination.published == ["a", "c", "c2", "d"]
    assert publisher.publish({"w": torch.full((2,), 3.0)}, "e") == "d"


@pytest.mark.skipif(
    not hasattr(torch.backends, "fp32_precision"), reason="needs fp32_precision"
)
def test_fingerprint_reads_unset_precision_as_its_effective_value():
    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "none"
        unset = fingerprint_digest(
            merge_fingerprint(export="merged", base_identity="b", device="cpu")
        )
        # torch resolves "none" in place once something reads the precision.
        matmul.fp32_precision = "ieee"
        resolved = fingerprint_digest(
            merge_fingerprint(export="merged", base_identity="b", device="cpu")
        )
        matmul.fp32_precision = "tf32"
        tf32 = fingerprint_digest(
            merge_fingerprint(export="merged", base_identity="b", device="cpu")
        )
    finally:
        matmul.fp32_precision = previous
    assert unset == resolved != tf32


@pytest.mark.skipif(
    not hasattr(torch.backends, "fp32_precision"), reason="needs fp32_precision"
)
def test_fingerprint_ignores_another_threads_ieee_override():
    from axolotl.utils.fp32_precision import ieee_fp32_matmul

    matmul = torch.backends.cuda.matmul
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = "tf32"
        configured = merge_fingerprint(export="merged", base_identity="b")
        with ieee_fp32_matmul():
            assert matmul.fp32_precision == "ieee"
            during = merge_fingerprint(export="merged", base_identity="b")
        assert matmul.fp32_precision == "tf32"
    finally:
        matmul.fp32_precision = previous
    assert during["float32_precision"]["cuda_matmul"] == "tf32"
    assert fingerprint_digest(during) == fingerprint_digest(configured)


def test_fingerprint_records_eager_and_live_tf32():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        eager = merge_fingerprint(export="merged", base_identity="base", device="cpu")
        assert eager["merge_compile"] == "eager"
        assert eager["float32_precision"]["cuda_matmul"] == "ieee"
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


def test_a_destination_needs_a_publish_cadence(tmp_path):
    from axolotl.core.publishing.callback import publish_settings

    cfg = {"adapter": "lora", "checkpoint_publish": {"destination": str(tmp_path)}}
    with pytest.raises(ValueError, match="checkpoint_publish.every"):
        publish_settings(DictDefault(cfg))
    cfg["checkpoint_publish"]["every"] = 2
    assert publish_settings(DictDefault(cfg)).every == 2


def test_fan_in_fan_out_adapters_match_the_transposed_base(tmp_path):
    import json

    from safetensors.torch import save_file

    from axolotl.core.publishing.adapters import (
        ADAPTER_CONFIG,
        ADAPTER_WEIGHTS,
        PREFIX,
        validate_adapter,
    )

    # GPT-2's c_attn: a Conv1D stores the weight as [in, out].
    hidden, rank = 4, 2
    module = "transformer.h.0.attn.c_attn"
    save_file(
        {
            f"{PREFIX}{module}.lora_A.weight": torch.zeros(rank, hidden),
            f"{PREFIX}{module}.lora_B.weight": torch.zeros(3 * hidden, rank),
        },
        tmp_path / ADAPTER_WEIGHTS,
    )
    config = {
        "peft_type": "LORA",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": ["c_attn"],
        "fan_in_fan_out": True,
    }
    (tmp_path / ADAPTER_CONFIG).write_text(json.dumps(config))
    layout = {f"{module}.weight": ("float32", [hidden, 3 * hidden])}
    assert validate_adapter(tmp_path, layout)
    config["fan_in_fan_out"] = False
    (tmp_path / ADAPTER_CONFIG).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="LoRA shape"):
        validate_adapter(tmp_path, layout)


@pytest.mark.parametrize(
    ("dtype", "rejected"), [("float16", True), ("bfloat16", False)]
)
def test_bf16_export_of_an_unquantized_base(tmp_path, dtype, rejected):
    from axolotl.core.publishing.exporters import MergedExporter
    from axolotl.core.publishing.pinned_base import PinnedBase

    base = PinnedBase(
        source=tmp_path,
        path=tmp_path,
        identity="base",
        format="unquantized",
        layout={"w.weight": (dtype, [2, 2])},
        files={"w.weight": "model.safetensors"},
        config={},
        linked=False,
    )
    if rejected:
        with pytest.raises(ValueError, match="merged_format: preserve"):
            MergedExporter(base, "bf16")
    else:
        MergedExporter(base, "bf16")
