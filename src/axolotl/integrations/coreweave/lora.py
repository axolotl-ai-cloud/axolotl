"""Shardwise BF16 LoRA publication against an exact, committed disk baseline."""

import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from axolotl.cli.utils.lora_merge import merge_lora_sharded_efficient
from axolotl.utils.delta_checkpoint import write_checkpoint

from .checkpoints import CheckpointPublisher


def adapter_checkpoint(directory):
    """Keep adapter tensors on disk until the ordered publication worker runs."""
    return Path(directory), None


class LoRACheckpointPublisher(CheckpointPublisher):
    """Merge every adapter against the original base, never rounded prior merges."""

    def __init__(self, *args, base_model, revision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        self.revision = revision
        self.original = None
        self.layout = {}

    def _pin_base(self):
        if self.original is not None:
            return
        source = Path(self.base_model)
        if not source.is_dir():
            from huggingface_hub import snapshot_download

            source = Path(
                snapshot_download(
                    self.base_model,
                    revision=self.revision,
                    allow_patterns=["*.safetensors", "*.json"],
                )
            )
        config = json.loads((source / "config.json").read_text())
        if config.get("quantization_config"):
            raise ValueError("LoRA publication requires an unquantized BF16 base")
        index = source / "model.safetensors.index.json"
        if index.exists():
            expected = json.loads(index.read_text())["weight_map"]
            names = set(expected.values())
        else:
            expected = None
            names = {"model.safetensors"}
        self.directory.mkdir(parents=True, exist_ok=True)
        pinned = Path(tempfile.mkdtemp(prefix=".original-", dir=self.directory))
        layout = {}
        try:
            for name in names | {"config.json"}:
                if Path(name).name != name:
                    raise ValueError("Base shards must be in the model directory")
                shutil.copy2(source / name, pinned / name)
            if expected is not None:
                shutil.copy2(index, pinned / index.name)
            for name in names:
                with safe_open(pinned / name, framework="pt") as shard:
                    for key in shard.keys():
                        if key in layout or (
                            expected is not None and expected.get(key) != name
                        ):
                            raise ValueError("Invalid base checkpoint index")
                        tensor = shard.get_slice(key)
                        if tensor.get_dtype() != "BF16":
                            raise ValueError(
                                "LoRA publication requires BF16 base tensors"
                            )
                        layout[key] = tensor.get_shape()
            if expected is not None and layout.keys() != expected.keys():
                raise ValueError("Base checkpoint index is incomplete")
        except BaseException:
            shutil.rmtree(pinned)
            raise
        self.original, self.layout = pinned, layout

    def _validate_adapter(self, directory):
        config = json.loads((directory / "adapter_config.json").read_text())
        if (
            config.get("peft_type") != "LORA"
            or any(
                config.get(key)
                for key in (
                    "use_dora",
                    "modules_to_save",
                    "target_parameters",
                    "trainable_token_indices",
                    "layer_replication",
                    "lora_bias",
                    "alora_invocation_tokens",
                    "arrow_config",
                    "use_qalora",
                )
            )
            or config.get("bias", "none") != "none"
        ):
            raise ValueError("Only standard LoRA linear adapters are supported")
        if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
            raise ValueError("LoRA publication requires an unchanged original base")
        with safe_open(
            directory / "adapter_model.safetensors", framework="pt"
        ) as adapter:
            keys = set(adapter.keys())
            if not keys:
                raise ValueError("Empty LoRA adapter")
            for key in keys:
                prefix = "base_model.model."
                if not key.startswith(prefix) or not key.endswith(
                    (".lora_A.weight", ".lora_B.weight")
                ):
                    raise ValueError(f"Unsupported LoRA tensor: {key}")
                module, side = key[len(prefix) :].rsplit(".lora_", 1)
                base_key = module + ".weight"
                if base_key not in self.layout or len(self.layout[base_key]) != 2:
                    raise ValueError(f"LoRA tensor does not match a base matrix: {key}")
                other = "B" if side == "A.weight" else "A"
                if f"{prefix}{module}.lora_{other}.weight" not in keys:
                    raise ValueError(f"Incomplete LoRA pair: {key}")

    def publish(self, weights, model_config, step):
        if self.pending is not None:
            raise RuntimeError("Recover the pending checkpoint before publishing again")
        adapter = Path(weights)
        self._pin_base()
        self._validate_adapter(adapter)
        identity = f"step-{step:08d}-{uuid.uuid4().hex}"
        full = self.base is None or self.delta_count >= self.config.max_deltas
        staging = Path(tempfile.mkdtemp(prefix=".merge-", dir=self.directory))
        candidate, artifact = staging / "merged", staging / "artifact"
        artifact.mkdir()
        weight_map = {}
        seen = set()
        total_size = 0
        shard_number = 0

        def consume(tensors, destination, metadata):
            nonlocal total_size, shard_number
            for key, tensor in tensors.items():
                if key in seen or list(tensor.shape) != self.layout.get(key):
                    raise ValueError(
                        f"Merged checkpoint changed the base layout: {key}"
                    )
                seen.add(key)
            if any(tensor.dtype != torch.bfloat16 for tensor in tensors.values()):
                raise ValueError("Merged serving weights must be BF16")
            previous = (
                None if self.base is None else load_file(self.base / destination.name)
            )
            unchanged = (
                previous is not None
                and previous.keys() == tensors.keys()
                and all(
                    tensor.shape == previous[key].shape
                    and tensor.dtype == previous[key].dtype
                    and torch.equal(
                        tensor.view(torch.uint8), previous[key].view(torch.uint8)
                    )
                    for key, tensor in tensors.items()
                )
            )
            if unchanged:
                os.link(self.base / destination.name, destination)
            else:
                save_file(tensors, destination, metadata=metadata)
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            if full:
                os.link(destination, artifact / destination.name)
                weight_map.update(dict.fromkeys(tensors, destination.name))
            elif not unchanged:
                part = write_checkpoint(
                    tensors,
                    staging / "part",
                    identity=identity,
                    base=previous,
                    base_identity=self.identity,
                    max_shard_size=self.config.max_shard_size,
                )
                manifest = json.loads(
                    (part / "model.safetensors.index.json").read_text()
                )
                renamed = {}
                for name in set(manifest["weight_map"].values()):
                    renamed[name] = f"model-{shard_number:05d}.safetensors"
                    (part / name).rename(artifact / renamed[name])
                    shard_number += 1
                weight_map.update(
                    {key: renamed[name] for key, name in manifest["weight_map"].items()}
                )
                shutil.rmtree(part)

        try:
            merge_lora_sharded_efficient(
                self.original, adapter, candidate, shard_sink=consume
            )
            if seen != self.layout.keys():
                raise ValueError("Merged checkpoint is missing base tensors")
            if not full and not weight_map:
                return self.identity
            if full:
                shutil.copy2(candidate / "config.json", artifact / "config.json")
                metadata = {"total_size": total_size}
            else:
                metadata = {
                    "version": identity,
                    "base_version": self.identity,
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                }
            (artifact / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": metadata, "weight_map": weight_map})
            )
            cache = self.directory / f".baseline-{identity}"
            path = self.directory / identity
            artifact.rename(path)
            candidate.rename(cache)
            self.pending = (identity, cache, full)
        finally:
            shutil.rmtree(staging)
        self.client.ready_identity = None
        self.upload(path, identity)
        self.client.hot_load(identity, base_identity=None if full else self.identity)
        self._commit()
        return identity

    def _commit(self):
        previous = self.base
        super()._commit()
        if previous is not None:
            shutil.rmtree(previous)
