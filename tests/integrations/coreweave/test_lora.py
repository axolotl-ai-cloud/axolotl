"""Exercise real sharded merges, exact delta decoding, and baseline transactions."""

import json
import shutil
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import zstandard
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient
from axolotl.integrations.coreweave.async_publisher import retain_model_checkpoint
from axolotl.integrations.coreweave.lora import LoRACheckpointPublisher
from axolotl.integrations.coreweave.plugin import (
    CoreWeaveCheckpointCallback,
    load_weights,
)

from .test_checkpoints import config


@pytest.fixture
def model_and_adapter(tmp_path):
    torch.manual_seed(19)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    ).to(torch.bfloat16)
    base = tmp_path / "base"
    model.save_pretrained(base, max_shard_size="2KB")
    peft = get_peft_model(
        model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"])
    )

    def save(step):
        with torch.no_grad():
            for name, parameter in peft.named_parameters():
                if "lora_" in name:
                    parameter.fill_(0.031 * step)
        directory = tmp_path / f"checkpoint-{step}"
        peft.save_pretrained(directory)
        return directory

    return base, save


def decode(path, previous):
    result = {key: tensor.clone() for key, tensor in previous.items()}
    manifest = json.loads((path / "model.safetensors.index.json").read_text())
    for name in set(manifest["weight_map"].values()):
        with safe_open(path / name, framework="numpy") as shard:
            for key in shard.keys():
                delta = np.frombuffer(
                    zstandard.ZstdDecompressor().decompress(shard.get_tensor(key)),
                    dtype=np.uint8,
                )
                target = result[key].view(torch.uint8).numpy().reshape(-1)
                target[:] ^= delta
                assert shard.metadata()[key] == f"{zlib.adler32(target):08x}"
    return result


def test_sharded_delta_chain_matches_fresh_merges(tmp_path, model_and_adapter):
    base, save = model_and_adapter
    bucket = tmp_path / "bucket"

    def upload(source, _bucket, key):
        (bucket / key).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, bucket / key)

    s3 = MagicMock()
    s3.upload_file.side_effect = upload
    publisher = LoRACheckpointPublisher(
        config(max_deltas=2),
        MagicMock(),
        tmp_path / "publish",
        s3=s3,
        base_model=str(base),
    )
    previous = None
    for step in range(1, 5):
        adapter = save(step)
        identity = publisher.publish(adapter, None, step)
        assert not (publisher.directory / identity).exists()
        artifact = bucket / "weights" / identity
        if step in (1, 4):
            actual, _ = load_weights(artifact)
        else:
            actual = decode(artifact, previous)
            index = json.loads((artifact / "model.safetensors.index.json").read_text())
            assert set(index["weight_map"]) == {
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.1.self_attn.q_proj.weight",
            }
        expected_dir = tmp_path / f"expected-{step}"
        merge_lora_sharded_efficient(base, adapter, expected_dir)
        expected, _ = load_weights(expected_dir)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert torch.equal(
                actual[key].view(torch.uint8), expected[key].view(torch.uint8)
            ), key
        previous = actual
        assert len(list(publisher.directory.glob(".baseline-*"))) == 1
    calls = publisher.client.hot_load.call_count
    assert publisher.publish(adapter, None, 5) == identity
    assert publisher.client.hot_load.call_count == calls


def test_failed_upload_keeps_baseline_until_recovery(tmp_path, model_and_adapter):
    base, save = model_and_adapter
    client, s3 = MagicMock(), MagicMock()
    publisher = LoRACheckpointPublisher(
        config(), client, tmp_path / "publish", s3=s3, base_model=str(base)
    )
    identity = publisher.publish(save(1), None, 1)
    old = publisher.base
    s3.upload_file.side_effect = RuntimeError("upload failed")
    with pytest.raises(RuntimeError, match="upload failed"):
        publisher.publish(save(2), None, 2)
    assert publisher.identity == identity
    assert publisher.base == old and old.exists()
    assert publisher.pending[1].exists()
    assert (publisher.directory / publisher.pending[0]).is_dir()
    with pytest.raises(RuntimeError, match="pending"):
        publisher.publish(save(3), None, 3)
    publisher.recover("retried-operation")
    assert publisher.identity != identity
    assert publisher.base.exists() and not old.exists()
    assert not (publisher.directory / publisher.identity).exists()


def test_adapter_rotation_and_background_merge(tmp_path, model_and_adapter):
    base, save = model_and_adapter
    callback = CoreWeaveCheckpointCallback(config(), base_model=str(base))
    callback.publisher = LoRACheckpointPublisher(
        config(),
        MagicMock(),
        tmp_path / "publish",
        s3=MagicMock(),
        base_model=str(base),
    )
    directory = save(1)
    retained = retain_model_checkpoint(directory, tmp_path / "retained")
    assert (retained / "adapter_model.safetensors").stat().st_ino == (
        directory / "adapter_model.safetensors"
    ).stat().st_ino
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path)),
        SimpleNamespace(global_step=1, is_world_process_zero=True),
        None,
    )
    shutil.rmtree(directory)
    callback.on_train_end(None, None, None)
    assert callback.publisher.identity is not None
    assert not list((tmp_path / "coreweave-checkpoints" / ".queue").iterdir())


def test_base_is_pinned_and_bad_adapter_fails_before_upload(
    tmp_path, model_and_adapter
):
    base, save = model_and_adapter
    publisher = LoRACheckpointPublisher(
        config(),
        MagicMock(),
        tmp_path / "publish",
        s3=MagicMock(),
        base_model=str(base),
    )
    publisher.publish(save(1), None, 1)
    shutil.rmtree(base)
    adapter = save(2)
    publisher.publish(adapter, None, 2)
    cfg_path = adapter / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["use_dora"] = True
    cfg_path.write_text(json.dumps(cfg))
    publisher.s3.reset_mock()
    with pytest.raises(ValueError, match="standard LoRA"):
        publisher.publish(adapter, None, 3)
    publisher.s3.upload_file.assert_not_called()
