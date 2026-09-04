# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Tests that NVFP4 layout inspection reads local data instead of the HF hub.

``inspect_nvfp4_layout`` runs in ``pre_model_load`` for every NVFP4 architecture
(``qwen3_moe``, ``qwen3_next``, ``glm_moe_dsa``, ``qwen4_exp``). It used to always call
``HfApi().get_safetensors_metadata``, which made an air-gapped (``HF_HUB_OFFLINE=1``) run
impossible and exposed a fully-cached 170 GB checkpoint to hub ``429`` rate limiting.

These pin that a local snapshot dir and an already-cached hub repo are both served from
disk with no network call at all, that a genuinely uncached repo still falls back to the
hub API, and that the offline error names what is missing. CPU-only, no triton/CUDA.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

import axolotl

# Loaded by path: the package __init__ pulls in the triton kernels, and nothing here needs them.
_spec = importlib.util.spec_from_file_location(
    "nvfp4_moe_loading_offline_test",
    Path(axolotl.__file__ or "").parent
    / "integrations/kernels/libs/scattermoe_lora/nvfp4_moe_loading.py",
)
assert _spec is not None and _spec.loader is not None
nvfp4_moe_loading = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nvfp4_moe_loading)

_ROUTED = ("gate_proj", "up_proj", "down_proj")


def _nvfp4_module(prefix: str, n: int, k: int) -> dict[str, torch.Tensor]:
    """modelopt-style NVFP4 leaves: uint8 qdata (K halved) + e4m3 group scale + per-tensor."""
    return {
        f"{prefix}.weight": torch.zeros(n, k // 2, dtype=torch.uint8),
        f"{prefix}.weight_scale": torch.zeros(n, k // 16, dtype=torch.float8_e4m3fn),
        f"{prefix}.weight_scale_2": torch.ones(1, dtype=torch.float32),
    }


def _make_tensors(n_layers: int = 2, n_experts: int = 2) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros(8, 16, dtype=torch.bfloat16),
    }
    for layer in range(n_layers):
        for e in range(n_experts):
            for proj in _ROUTED:
                tensors.update(
                    _nvfp4_module(
                        f"model.layers.{layer}.mlp.experts.{e}.{proj}", 64, 128
                    )
                )
        tensors.update(_nvfp4_module(f"model.layers.{layer}.self_attn.q_proj", 32, 64))
        # fp8 module: carries a weight_scale but an fp8 (not uint8) weight, so NOT NVFP4
        tensors[f"model.layers.{layer}.self_attn.k_proj.weight"] = torch.zeros(
            32, 64, dtype=torch.float8_e4m3fn
        )
        tensors[f"model.layers.{layer}.self_attn.k_proj.weight_scale"] = torch.ones(
            1, dtype=torch.float32
        )
    return tensors


def _write_sharded(dirpath: Path, tensors: dict[str, torch.Tensor], n_shards: int = 2):
    """Write ``tensors`` across ``n_shards`` safetensors files plus an index json."""
    dirpath.mkdir(parents=True, exist_ok=True)
    names = sorted(tensors)
    weight_map = {}
    for i in range(n_shards):
        shard = f"model-{i + 1:05d}-of-{n_shards:05d}.safetensors"
        part = {n: tensors[n] for n in names[i::n_shards]}
        save_file(part, str(dirpath / shard))
        weight_map.update({n: shard for n in part})
    (dirpath / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    return sorted({v for v in weight_map.values()})


def _make_cache_repo(cache_dir: Path, repo_id: str, files: list[str], src: Path):
    """Lay out ``src``'s files the way huggingface_hub caches them, minus the blob symlinks."""
    repo_dir = cache_dir / ("models--" + repo_id.replace("/", "--"))
    sha = "0" * 40
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / "main").write_text(sha)
    for name in files:
        (snap / name).write_bytes((src / name).read_bytes())
    return snap


@pytest.fixture(name="no_network")
def fixture_no_network(monkeypatch):
    """Make any hub round trip an immediate, loud failure."""
    import huggingface_hub

    def _boom(*args, **kwargs):
        raise AssertionError("unexpected HF hub network call")

    monkeypatch.setattr(huggingface_hub, "HfApi", _boom)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _boom)


@pytest.fixture(name="ckpt")
def fixture_ckpt(tmp_path):
    src = tmp_path / "ckpt"
    shards = _write_sharded(src, _make_tensors())
    return src, shards


def test_local_dir_reads_headers_without_network(ckpt, no_network):
    src, _ = ckpt
    meta = nvfp4_moe_loading._safetensors_metadata(str(src))
    assert meta["model.layers.0.mlp.experts.0.gate_proj.weight"] == ("U8", (64, 64))
    assert meta["model.embed_tokens.weight"] == ("BF16", (8, 16))
    assert len(meta) == len(_make_tensors())


def test_local_dir_single_file_checkpoint(tmp_path, no_network):
    src = tmp_path / "single"
    src.mkdir()
    save_file(_make_tensors(n_layers=1), str(src / "model.safetensors"))
    meta = nvfp4_moe_loading._safetensors_metadata(str(src))
    assert meta["model.layers.0.mlp.experts.0.up_proj.weight"] == ("U8", (64, 64))


def test_cached_hub_repo_reads_from_cache_without_network(
    ckpt, tmp_path, monkeypatch, no_network
):
    import huggingface_hub

    src, shards = ckpt
    cache = tmp_path / "hub"
    _make_cache_repo(cache, "org/nvfp4", ["model.safetensors.index.json", *shards], src)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    assert nvfp4_moe_loading._local_shard_paths("org/nvfp4") is not None
    assert nvfp4_moe_loading._safetensors_metadata(
        "org/nvfp4"
    ) == nvfp4_moe_loading._safetensors_metadata(str(src))


def test_cached_hub_repo_resolves_files_without_network(
    ckpt, tmp_path, monkeypatch, no_network
):
    import huggingface_hub

    src, _ = ckpt
    cache = tmp_path / "hub"
    snap = _make_cache_repo(cache, "org/nvfp4", ["model.safetensors.index.json"], src)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))

    assert nvfp4_moe_loading._resolve_repo_file(
        "org/nvfp4", "model.safetensors.index.json"
    ) == str(snap / "model.safetensors.index.json")
    assert len(nvfp4_moe_loading._load_index("org/nvfp4")) == len(_make_tensors())


def test_partially_cached_hub_repo_falls_back_to_api(ckpt, tmp_path, monkeypatch):
    """Index cached but a shard missing: the hub metadata API is still the only answer."""
    import huggingface_hub

    src, shards = ckpt
    cache = tmp_path / "hub"
    _make_cache_repo(
        cache, "org/nvfp4", ["model.safetensors.index.json", shards[0]], src
    )
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    assert nvfp4_moe_loading._local_shard_paths("org/nvfp4") is None

    called = []

    class _FakeApi:
        def get_safetensors_metadata(self, repo_id):
            called.append(repo_id)
            tinfo = type("T", (), {"dtype": "U8", "shape": [64, 64]})()
            fmeta = type("F", (), {"tensors": {"a.weight": tinfo}})()
            return type("M", (), {"files_metadata": {"f": fmeta}})()

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    assert nvfp4_moe_loading._safetensors_metadata("org/nvfp4") == {
        "a.weight": ("U8", (64, 64))
    }
    assert called == ["org/nvfp4"]


def test_offline_uncached_repo_raises_named_error(tmp_path, monkeypatch, no_network):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub")
    )
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    with pytest.raises(OSError, match="org/absent"):
        nvfp4_moe_loading._safetensors_metadata("org/absent")


def test_inspect_layout_from_local_dir(ckpt, no_network):
    """The layout the shared adapters consume must be identical off-line."""
    src, _ = ckpt
    layout = nvfp4_moe_loading.inspect_nvfp4_layout(str(src))
    assert layout["routed_present"] is True
    assert layout["routed_projs"] == sorted(_ROUTED)
    assert layout["routed_sample_shapes"]["gate_proj"] == ("U8", (64, 64))
    # the fp8 k_proj must not be picked up; only the NVFP4 q_proj is non-routed
    assert layout["nonrouted_suffixes"] == ["self_attn.q_proj"]
    assert layout["qdata_names"] == ["weight"]
    assert layout["per_tensor_names"] == ["weight_scale_2"]
    assert layout["naming"] == "modelopt"
