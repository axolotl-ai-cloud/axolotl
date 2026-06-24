# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Stream-write a merged NVFP4 MoE checkpoint (step 3 of the expert-LoRA merge).

``write_merged_nvfp4_checkpoint`` reads an NVFP4-quantized MoE base checkpoint and a trained
ScatterMoE expert-LoRA adapter, and re-emits the SAME source NVFP4 schema with the LoRA delta
folded into each expert: per MoE layer it rebuilds the fused gate_up/down base NVFP4Tensors,
calls ``merge_layer_experts`` (dequant -> add bf16 delta -> un-fuse -> requant), and OVERWRITES
the base expert tensors at their existing key names. Every non-expert tensor (attention, norms,
embeddings, router, FP8 dense weights + scales) is copied byte-identical.

Bounded memory: each layer's merged experts are streamed straight to disk (never accumulated for
the whole model), and the output is re-sharded into size-bounded shards. Peak resident memory is
therefore ~one layer's working set plus one shard buffer, independent of model size. The index
weight_map is rebuilt; shard grouping and filenames may differ from the base, but every key name
and every non-expert value is preserved (loaders read via the index, not the shard layout).

Importable without triton: only torch + torchao + safetensors + huggingface_hub at top level;
the sibling merge / loading helpers are imported lazily inside functions so the CPU round-trip
test can load this module by file path without executing the scattermoe_lora package __init__
(which imports triton)."""

from __future__ import annotations

import json
import math
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# A triton-less CPU test loads this module by file path (so the scattermoe_lora package
# __init__, which imports triton, never runs) and injects the two sibling modules here; on a
# GPU host this stays None and the relative package imports are used.
_SIBLINGS: dict[str, object] | None = None


def _merge_helpers():
    """Resolve the sibling merge / loading helpers.

    Prefers test-injected sibling modules (loaded by file path) and otherwise falls back to the
    normal relative package imports used on a GPU host where triton is present."""
    if _SIBLINGS is not None:
        merge = _SIBLINGS["nvfp4_lora_merge"]
        loading = _SIBLINGS["nvfp4_moe_loading"]
    else:
        from . import nvfp4_lora_merge as merge, nvfp4_moe_loading as loading

    return {
        "checkpoint_keys_for": merge.checkpoint_keys_for,
        "extract_expert_lora_from_state_dict": merge.extract_expert_lora_from_state_dict,
        "merge_layer_experts": merge.merge_layer_experts,
        "_build_expert_nvfp4": loading._build_expert_nvfp4,
        "_detect_scheme": loading._detect_scheme,
        "_load_index": loading._load_index,
        "_resolve_repo_file": loading._resolve_repo_file,
        "_shard_open": loading._shard_open,
        "_nvfp4_cls": loading._nvfp4_cls,
    }


def is_nvfp4_moe_checkpoint(base_repo: str) -> bool:
    """True if ``base_repo`` is an NVFP4-quantized MoE checkpoint mergeable by the writer.

    Detects via the same index probe the runtime loader uses (a known per-expert NVFP4 naming
    scheme present in ``model.safetensors.index.json``) OR an ``hf_quant_config.json`` declaring
    an NVFP4 MoE quant algo. Robust to a missing/unreadable index or config (returns False rather
    than raising) so the CLI can call it unconditionally on any base. ``base_repo`` may be a local
    dir or an HF hub id."""
    h = _merge_helpers()
    try:
        wmap = h["_load_index"](base_repo)
        _, scheme = h["_detect_scheme"](wmap)
        if scheme is not None:
            return True
    except Exception:  # noqa: BLE001 - missing/unreadable index is just "not NVFP4 MoE"
        pass

    try:
        with open(h["_resolve_repo_file"](base_repo, "hf_quant_config.json")) as f:
            quant = json.load(f).get("quantization", {})
        algos = (
            str(quant.get("moe_quant_algo", "")),
            str(quant.get("quant_algo", "")),
        )
        return any("NVFP4" in a.upper() for a in algos)
    except Exception:  # noqa: BLE001 - missing/unreadable config is just "not NVFP4 MoE"
        return False


def _adapter_scaling(adapter_dir: str) -> float:
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    denom = math.sqrt(r) if use_rslora else r
    return alpha / denom


def _num_experts(config: dict) -> int:
    for key in ("n_routed_experts", "num_experts", "num_local_experts"):
        val = config.get(key)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError(
        "could not determine number of experts from base config "
        "(tried n_routed_experts, num_experts, num_local_experts)"
    )


def _load_adapter_state_dict(adapter_dir: str) -> dict[str, torch.Tensor]:
    st = os.path.join(adapter_dir, "adapter_model.safetensors")
    if os.path.isfile(st):
        sd: dict[str, torch.Tensor] = {}
        with safe_open(st, framework="pt") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu", weights_only=True)
    raise FileNotFoundError(
        f"no adapter_model.safetensors or adapter_model.bin in {adapter_dir}"
    )


def _moe_layers(wmap: dict, scheme: dict) -> list[int]:
    """Layer indices whose expert-0 gate_up[0] weight is present in the index."""
    base_fmt = scheme["base_fmt"]
    gate0 = scheme["gate_up"][0]
    layers = set()
    layer = 0
    misses = 0
    # Scan upward; allow gaps (dense layers) before giving up.
    while misses < 256:
        if base_fmt.format(layer=layer, e=0, proj=gate0) + ".weight" in wmap:
            layers.add(layer)
            misses = 0
        else:
            misses += 1
        layer += 1
    return sorted(layers)


_DEFAULT_MAX_SHARD_BYTES = 5 * 1024**3


class _ShardWriter:
    """Accumulate tensors and flush size-bounded safetensors shards so peak memory stays ~one shard.

    safetensors writes a whole file at once (no append), so the buffer is flushed to a shard once
    adding the next group would exceed ``max_shard_bytes`` (a single oversized group still gets its
    own shard). Tensors are added in atomic GROUPS that are never split across a shard boundary:
    the loader (``_build_expert_nvfp4``) reads each expert's weight/weight_scale/weight_scale_2 from
    the shard of its ``.weight`` key, so a triple must stay co-located. Shards are written under a
    temp name and renamed to ``model-{i}-of-{n}`` at ``finalize`` once the total count is known;
    ``finalize`` also writes the index with the recomputed total_size."""

    def __init__(self, output_dir: str, max_shard_bytes: int):
        self._output_dir = output_dir
        self._max = max_shard_bytes
        self._buf: dict[str, torch.Tensor] = {}
        self._buf_bytes = 0
        self._shard_keys: list[list[str]] = []
        self._total = 0

    def add_group(self, items: list[tuple[str, torch.Tensor]]) -> None:
        """Add a set of keys that MUST land in the same shard (e.g. an expert's weight + scales)."""
        group_bytes = sum(t.numel() * t.element_size() for _, t in items)
        if self._buf and self._buf_bytes + group_bytes > self._max:
            self._flush()
        for key, tensor in items:
            self._buf[key] = tensor
        self._buf_bytes += group_bytes
        self._total += group_bytes

    def _flush(self) -> None:
        tmp = os.path.join(
            self._output_dir, f"_shard-{len(self._shard_keys):05d}.safetensors"
        )
        save_file(self._buf, tmp, metadata={"format": "pt"})
        self._shard_keys.append(list(self._buf.keys()))
        self._buf = {}
        self._buf_bytes = 0

    def finalize(self) -> None:
        if self._buf:
            self._flush()
        n = len(self._shard_keys)
        weight_map: dict[str, str] = {}
        for idx, keys in enumerate(self._shard_keys):
            final = f"model-{idx + 1:05d}-of-{n:05d}.safetensors"
            os.rename(
                os.path.join(self._output_dir, f"_shard-{idx:05d}.safetensors"),
                os.path.join(self._output_dir, final),
            )
            for key in keys:
                weight_map[key] = final
        index = {"metadata": {"total_size": self._total}, "weight_map": weight_map}
        with open(
            os.path.join(self._output_dir, "model.safetensors.index.json"), "w"
        ) as f:
            json.dump(index, f, indent=2)


def write_merged_nvfp4_checkpoint(
    base_repo: str,
    adapter_dir: str,
    output_dir: str,
    *,
    device: str = "cpu",
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> None:
    """Merge an expert-LoRA adapter into an NVFP4 MoE base and write a new checkpoint.

    ``base_repo`` may be a local snapshot dir or an HF hub id. The output re-emits the base's
    NVFP4 source schema (per-proj unfused expert keys); merged expert tensors overwrite the base
    tensors at the same key names, all other tensors are copied byte-identical, and the index
    keeps the base's shard layout (recomputed total_size)."""
    h = _merge_helpers()
    NVFP4Tensor = h["_nvfp4_cls"]()
    if NVFP4Tensor is None:
        raise ImportError(
            "torchao NVFP4Tensor is required to write a merged NVFP4 checkpoint"
        )

    os.makedirs(output_dir, exist_ok=True)

    with open(h["_resolve_repo_file"](base_repo, "config.json")) as f:
        base_config = json.load(f)
    num_experts = _num_experts(base_config)
    scaling = _adapter_scaling(adapter_dir)

    adapter_sd = _load_adapter_state_dict(adapter_dir)
    extracted = h["extract_expert_lora_from_state_dict"](
        adapter_sd, num_experts, scaling
    )

    wmap = h["_load_index"](base_repo)
    _scheme_name, scheme = h["_detect_scheme"](wmap)
    if scheme is None:
        raise ValueError(f"no known NVFP4 MoE naming scheme found in {base_repo} index")
    base_fmt = scheme["base_fmt"]

    writer = _ShardWriter(output_dir, max_shard_bytes)
    regenerated: set[str] = set()

    # Merge each MoE layer and stream its merged experts straight out, so only one layer's working
    # set plus the current shard buffer is ever resident (never the whole model).
    for layer in _moe_layers(wmap, scheme):
        gqd, gscale, gpts = h["_build_expert_nvfp4"](
            base_repo, wmap, base_fmt, layer, scheme["gate_up"], num_experts, device
        )
        dqd, dscale, dpts = h["_build_expert_nvfp4"](
            base_repo, wmap, base_fmt, layer, scheme["down"], num_experts, device
        )
        base_gate_up = NVFP4Tensor(
            gqd, gscale, 16, torch.bfloat16, per_tensor_scale=gpts
        )
        base_down = NVFP4Tensor(dqd, dscale, 16, torch.bfloat16, per_tensor_scale=dpts)
        gup_lora = extracted.get((layer, "gate_up_proj"))
        down_lora = extracted.get((layer, "down_proj"))
        layer_out = h["merge_layer_experts"](
            base_gate_up, base_down, gup_lora, down_lora, scheme
        )
        for source_proj, per_expert in layer_out.items():
            for expert, tensors in per_expert.items():
                keys = h["checkpoint_keys_for"](scheme, layer, source_proj, expert)
                # The triple must stay in one shard (the loader reads weight_scale/weight_scale_2
                # from the shard of .weight). clone(): gate/up share one weight_scale_2 tensor
                # object and safetensors rejects keys aliasing the same storage.
                group = [
                    (keys[which], tensors[which].contiguous().clone()) for which in keys
                ]
                writer.add_group(group)
                regenerated.update(key for key, _ in group)

    # Copy every non-expert tensor unchanged, grouped by source shard so each is opened once.
    passthrough: dict[str, list[str]] = {}
    for key, src_shard in wmap.items():
        if key in regenerated:
            continue
        passthrough.setdefault(src_shard, []).append(key)
    for src_shard in sorted(passthrough):
        f = h["_shard_open"](base_repo, src_shard)
        for key in passthrough[src_shard]:
            writer.add_group([(key, f.get_tensor(key))])

    writer.finalize()

    _copy_aux_files(base_repo, output_dir, h["_resolve_repo_file"])


def _copy_aux_files(base_repo: str, output_dir: str, resolve) -> None:
    """Copy config + tokenizer/quant aux files from the base into the output dir."""
    aux = [
        "config.json",
        "hf_quant_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ]
    for name in aux:
        try:
            src = resolve(base_repo, name)
        except Exception:  # noqa: BLE001 - hub miss / not present  # nosec B112
            continue
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(output_dir, name))
