"""Merged and LoRA export chains over a tiny modelopt NVFP4 MoE base, on CPU."""

import errno
import json
import os
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.cli.utils import lora_merge
from axolotl.core.publishing import (
    ChainPublisher,
    ChainWriter,
    LocalDestination,
    LoRAExporter,
    MergedExporter,
    pin_base,
)
from axolotl.core.publishing.adapters import validate_adapter
from axolotl.core.publishing.destinations import Destination
from axolotl.integrations.coreweave.lora import LoRACheckpointPublisher

from tests.core.publishing.nvfp4_moe import (
    INDEX,
    E,
    apply_delta,
    assert_bytes_equal,
    build_base,
    load_checkpoint,
    save_adapter,
)

pytest.importorskip("torchao")


@pytest.fixture
def base_dir(tmp_path):
    return build_base(tmp_path / "base")


def index_of(path):
    return json.loads((path / INDEX).read_text())


def fresh_merge(base, adapter, out):
    lora_merge.merge_lora_sharded_efficient(base, adapter, out)
    return load_checkpoint(out)


def merged_publisher(tmp_path, base_dir, **chain):
    base = pin_base(base_dir, tmp_path / "work")
    publisher = ChainPublisher(
        ChainWriter(tmp_path / "work", max_shard_size=64 << 10, **chain),
        LocalDestination(tmp_path / "published"),
    )
    return publisher, MergedExporter(base, workdir=tmp_path / "work")


def test_pin_base_links_without_copying(tmp_path, base_dir):
    base = pin_base(base_dir, tmp_path / "work")
    assert base.linked and base.format == "modelopt" and base.num_experts == E
    for name in ("model-00001-of-00002.safetensors", "config.json", INDEX):
        assert (base.path / name).stat().st_ino == (base_dir / name).stat().st_ino
    stem = "model.layers.0.mlp.experts.0.gate_proj"
    assert base.layout[f"{stem}.weight"] == ("uint8", [16, 32])
    assert base.layout[f"{stem}.weight_scale"] == ("float8_e4m3fn", [16, 4])
    assert base.layout[f"{stem}.weight_scale_2"] == ("float32", [])
    assert base.layout[f"{stem}.input_scale"] == ("float32", [])
    assert base.layout["model.layers.0.self_attn.q_proj.weight"][0] == "bfloat16"
    again = pin_base(base_dir, tmp_path / "work")
    assert again.identity == base.identity and again.path == base.path
    os.utime(base_dir / "model-00002-of-00002.safetensors", ns=(1, 1))
    with pytest.raises(RuntimeError, match="changed"):
        base.verify()
    assert pin_base(base_dir, tmp_path / "work").identity != base.identity


def test_pin_base_references_across_filesystems(tmp_path, base_dir, monkeypatch):
    def cross_device(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device)
    base = pin_base(base_dir, tmp_path / "work")
    assert not base.linked and base.path == base_dir
    assert not list((tmp_path / "work").glob(".base-*"))
    base.verify()


@pytest.mark.parametrize(
    "change, match",
    [
        ({"quant_method": "bitsandbytes"}, "quant_method"),
        (None, "no modelopt"),
    ],
)
def test_pin_base_refuses_unsupported_formats(tmp_path, base_dir, change, match):
    path = base_dir / "config.json"
    config = json.loads(path.read_text())
    if change is None:
        del config["quantization_config"]
    else:
        config["quantization_config"] = change
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=match):
        pin_base(base_dir, tmp_path / "work")


def test_adapter_validation(tmp_path, base_dir):
    base = pin_base(base_dir, tmp_path / "work")
    adapter = save_adapter(tmp_path / "adapter", 1)
    targets = validate_adapter(adapter, base.layout, num_experts=E)
    assert (targets.pairs, targets.linear, targets.expert_groups) == (7, 3, 4)

    def edited(**changes):
        path = adapter / "adapter_config.json"
        config = json.loads(path.read_text())
        config.update(changes)
        path.write_text(json.dumps(config))

    for changes, match in (
        ({"use_dora": True}, "standard LoRA"),
        ({"modules_to_save": ["lm_head"]}, "standard LoRA"),
        ({"target_parameters": None}, "does not match a base matrix"),
        ({"target_parameters": ["mlp.experts.gate_up_proj"]}, "2 LoRA wrappers"),
        (
            {"target_parameters": ["mlp.experts.gate_up_proj", "mlp.experts.w9"]},
            "does not match a base tensor",
        ),
    ):
        save_adapter(adapter, 1)
        edited(**changes)
        with pytest.raises(ValueError, match=match):
            validate_adapter(adapter, base.layout, num_experts=E)
    save_adapter(adapter, 1)
    with pytest.raises(ValueError, match="cover experts"):
        validate_adapter(adapter, base.layout, num_experts=E + 1)
    bf16_experts = {
        key: ("bfloat16", [16, 64]) if key.endswith("experts.0.gate_proj.weight") else v
        for key, v in base.layout.items()
    }
    with pytest.raises(ValueError, match="per-expert bfloat16"):
        validate_adapter(adapter, bf16_experts, num_experts=E)


@pytest.mark.parametrize("merge_aware", [False, True], ids=["reuse", "fresh"])
def test_merged_nvfp4_chain_matches_fresh_merges(tmp_path, base_dir, merge_aware):
    publisher, exporter = merged_publisher(tmp_path, base_dir)
    base_weights = load_checkpoint(base_dir)
    previous = None
    for step in (1, 2, 3):
        adapter = save_adapter(
            tmp_path / f"adapter-{step}", step, merge_aware=merge_aware
        )
        identity = publisher.publish_export(exporter, adapter, f"step-{step}")
        artifact = tmp_path / "published" / identity
        index = index_of(artifact)
        if step == 1:
            assert "delta_encoding" not in index["metadata"]
            actual = load_checkpoint(artifact)
        else:
            assert index["metadata"]["base_version"] == f"step-{step - 1}"
            changed = set(index["weight_map"])
            assert changed and not any("input_scale" in key for key in changed)
            assert not any(key.startswith("model.layers.2.mlp") for key in changed)
            assert not any("o_proj" in key or "mlp.gate." in key for key in changed)
            actual = apply_delta(artifact, previous)
        expected = fresh_merge(base_dir, adapter, tmp_path / f"expected-{step}")
        assert_bytes_equal(actual, expected)
        assert {k: (v.dtype, v.shape) for k, v in actual.items()} == {
            k: (v.dtype, v.shape) for k, v in base_weights.items()
        }
        stem = "model.layers.0.mlp.experts.1.down_proj.weight"
        assert not torch.equal(actual[stem], base_weights[stem])
        assert publisher.delta_count == step - 1
        previous = actual
    assert len(list((tmp_path / "work").glob(".baseline-*"))) == 1
    assert not list((tmp_path / "work").glob(".merge-*"))


def test_scale_mode_change_forces_full(tmp_path, base_dir):
    publisher, exporter = merged_publisher(tmp_path, base_dir)
    publisher.publish_export(exporter, save_adapter(tmp_path / "a1", 1), "step-1")
    publisher.publish_export(exporter, save_adapter(tmp_path / "a2", 2), "step-2")
    assert publisher.delta_count == 1
    adapter = save_adapter(tmp_path / "a3", 3, merge_aware=True)
    identity = publisher.publish_export(exporter, adapter, "step-3")
    assert publisher.delta_count == 0
    index = index_of(tmp_path / "published" / identity)
    assert "delta_encoding" not in index["metadata"]
    assert index["metadata"]["merge_fingerprint"] == (
        publisher.baseline.fingerprint_digest
    )
    assert publisher.baseline.fingerprint["scale_mode"] == "fresh"
    assert publisher.baseline.fingerprint["merge_compile"] == "eager"


class FailingDestination(Destination):
    def __init__(self, inner):
        self.inner, self.fail = inner, False

    def publish(self, staged):
        if self.fail:
            raise RuntimeError("upload failed")
        self.inner.publish(staged)


def test_failures_before_commit_leave_baseline(tmp_path, base_dir, monkeypatch):
    publisher, exporter = merged_publisher(tmp_path, base_dir)
    publisher.destination = destination = FailingDestination(publisher.destination)
    publisher.publish_export(exporter, save_adapter(tmp_path / "a1", 1), "step-1")
    committed = publisher.baseline
    snapshot = dict(committed.checksums)

    destination.fail = True
    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish_export(exporter, save_adapter(tmp_path / "a2", 2), "step-2")
    assert publisher.baseline is committed and committed.exists()
    assert publisher.pending is not None
    publisher.abandon()
    assert not (tmp_path / "work" / "step-2").exists()
    committed.verify()
    assert committed.checksums == snapshot

    real = lora_merge.merge_lora_sharded_efficient

    def corrupting(*args, shard_sink, **kwargs):
        def sink(tensors, destination, metadata):
            key = "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
            if key in tensors:
                tensors = {**tensors, key: tensors[key].float()}
            shard_sink(tensors, destination, metadata)

        return real(*args, shard_sink=sink, **kwargs)

    destination.fail = False
    monkeypatch.setattr(lora_merge, "merge_lora_sharded_efficient", corrupting)
    with pytest.raises(ValueError, match="pinned base layout"):
        publisher.publish_export(exporter, save_adapter(tmp_path / "a3", 3), "step-3")
    monkeypatch.setattr(lora_merge, "merge_lora_sharded_efficient", real)
    assert publisher.baseline is committed and publisher.pending is None
    assert not list((tmp_path / "work").glob(".staging-*"))
    assert not list((tmp_path / "work").glob(".merge-*"))

    identity = publisher.publish_export(
        exporter, save_adapter(tmp_path / "a4", 4), "step-4"
    )
    assert index_of(tmp_path / "published" / identity)["metadata"]["base_version"] == (
        "step-1"
    )


def test_dropped_expert_lora_is_refused(tmp_path, base_dir, monkeypatch):
    publisher, exporter = merged_publisher(tmp_path, base_dir)
    monkeypatch.setattr(lora_merge, "_per_expert_weights_are_packed", lambda _: False)
    with pytest.raises(RuntimeError, match="dropped the expert LoRA"):
        publisher.publish_export(exporter, save_adapter(tmp_path / "a1", 1), "step-1")
    assert publisher.baseline is None


def test_lora_export_with_and_without_deltas(tmp_path, base_dir):
    base = pin_base(base_dir, tmp_path / "work")
    exporter = LoRAExporter(base)
    publisher = ChainPublisher(
        ChainWriter(tmp_path / "work"), LocalDestination(tmp_path / "published")
    )
    previous = None
    for step in (1, 2):
        adapter = save_adapter(tmp_path / f"a{step}", step)
        identity = publisher.publish_export(exporter, adapter, f"lora-{step}")
        artifact = tmp_path / "published" / identity
        if step == 1:
            actual = load_checkpoint(artifact)
            assert json.loads((artifact / "config.json").read_text())["r"] == 4
        else:
            actual = apply_delta(artifact, previous)
        assert_bytes_equal(actual, load_checkpoint(adapter))
        previous = actual
    assert publisher.delta_count == 1
    assert publisher.baseline.fingerprint["export"] == "lora"
    assert "merge_compile" not in publisher.baseline.fingerprint

    identity = publisher.publish_export(
        exporter, save_adapter(tmp_path / "a3", 3, alpha=16), "lora-3"
    )
    assert publisher.delta_count == 0
    assert (
        "delta_encoding" not in index_of(tmp_path / "published" / identity)["metadata"]
    )

    full_only = ChainPublisher(
        ChainWriter(tmp_path / "work-full", delta=False),
        LocalDestination(tmp_path / "published-full"),
    )
    for step in (1, 2):
        adapter = save_adapter(tmp_path / f"a{step}", step)
        identity = full_only.publish_export(exporter, adapter, f"lora-{step}")
        artifact = tmp_path / "published-full" / identity
        assert "delta_encoding" not in index_of(artifact)["metadata"]
        assert_bytes_equal(load_checkpoint(artifact), load_checkpoint(adapter))


def test_coreweave_publishes_nvfp4_merge_chain(tmp_path, base_dir):
    from tests.integrations.coreweave.test_checkpoints import config

    client = MagicMock()
    publisher = LoRACheckpointPublisher(
        config(), client, tmp_path / "publish", s3=MagicMock(), base_model=str(base_dir)
    )
    first = publisher.publish(save_adapter(tmp_path / "a1", 1), None, 1)
    assert client.hot_load.call_args.kwargs["base_identity"] is None
    second = publisher.publish(save_adapter(tmp_path / "a2", 2), None, 2)
    assert client.hot_load.call_args.args[0] == second
    assert client.hot_load.call_args.kwargs["base_identity"] == first
    assert publisher.pinned.linked
    assert not list((tmp_path / "publish").glob(".original-*"))
    keys = [call.args[2] for call in publisher.s3.upload_file.call_args_list]
    assert keys[-1].endswith(f"{second}/{INDEX}")
