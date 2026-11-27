"""optmoe's ``nvfp4_lora`` precision: frozen packed NVFP4 experts plus a PEFT
``lora_target_parameters`` adapter, trained by the SM100 persistent kernels.

The experts stay the torchao ``NVFP4Tensor`` the kernels plugin loaded (EP-sliced to this
rank), out of FSDP's hands; per MoE layer a :class:`optmoe.nvfp4_lora.Nvfp4LoraExperts`
reuses their packed bytes and scales in place. Every optimizer step invalidates the
:class:`optmoe.nvfp4_lora.RequantCache`, so the next forward requantizes the merged
weights from the updated adapter (gradient accumulation reuses them; FSDP2 adapters are
unsharded when the forward reads them). ``expert_parallel_optmoe_nvfp4_merged_cache``
picks whether the cache holds every layer's forms across the step (``all``) or builds each
layer's forms just in time and keeps the last two layers (``layer``, the default).
"""

from __future__ import annotations

import importlib
import json
import os
import re
from typing import Any

import torch
from safetensors import safe_open
from torch.optim.optimizer import register_optimizer_step_post_hook

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

PRECISION = "nvfp4_lora"
_GATE_UP = "gate_up_proj"
_DOWN = "down_proj"
_X_SCALE_PROJS = ("gate_proj", "up_proj", "w1", "w3", "gate_up_proj")
_H_SCALE_PROJS = ("down_proj", "w2")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_INPUT_SCALE_RE = re.compile(
    r"layers\.(\d+)\.(?:.*\.)?experts\.(\d+)\.([A-Za-z0-9_]+)\.input_scale$"
)


_INDEXED_TARGET_RE = re.compile(r"(^|\.)\d+(\.|$)")


def _targets_expert_param(targets, name: str) -> bool:
    """Whether a target names ``name`` on every layer. A target with a layer index adapts only
    that layer, and one runtime precision serves all of them."""
    return any(
        (str(t) == name or str(t).endswith(f".{name}"))
        and not _INDEXED_TARGET_RE.search(str(t))
        for t in targets or ()
    )


def config_reason(cfg) -> str | None:
    """Config-only reasons the ``nvfp4_lora`` precision cannot run (``None`` = allowed)."""
    adapter = getattr(cfg, "adapter", None)
    if adapter != "lora":
        return f"nvfp4_lora trains a single lora adapter, adapter is {adapter!r}"
    if not getattr(cfg, "use_sonicmoe", False):
        return (
            "nvfp4_lora needs use_sonicmoe: true (the NVFP4 checkpoint loader and the "
            "expert-LoRA forward of the kernels plugin)"
        )
    if getattr(cfg, "lora_dropout", 0) or 0:
        return "nvfp4_lora needs lora_dropout: 0"
    targets = getattr(cfg, "lora_target_parameters", None) or ()
    if not (
        _targets_expert_param(targets, _GATE_UP)
        and _targets_expert_param(targets, _DOWN)
    ):
        return (
            "nvfp4_lora needs lora_target_parameters on experts.gate_up_proj and "
            "experts.down_proj of every layer (no per-layer targets)"
        )
    if getattr(cfg, "nvfp4_merge_aware", None):
        return (
            "nvfp4_merge_aware is on; merge-aware training is the SonicMoE path "
            "(set nvfp4_merge_aware: false for nvfp4_lora)"
        )
    if not base_is_nvfp4_modelopt(cfg):
        return "nvfp4_lora needs a modelopt NVFP4 base checkpoint"
    return None


def base_is_nvfp4_modelopt(cfg) -> bool:
    try:
        from axolotl.integrations.kernels.plugin import _base_is_nvfp4_modelopt

        return bool(_base_is_nvfp4_modelopt(cfg))
    except Exception:  # pylint: disable=broad-except
        return False


def is_nvfp4_tensor(param) -> bool:
    return type(param).__name__ == "NVFP4Tensor"


def experts_reason(experts, name: str) -> str | None:
    """Model-level reasons one Experts module cannot run as ``nvfp4_lora``."""
    for attr in (_GATE_UP, _DOWN):
        param = getattr(experts, attr, None)
        if param is None or not is_nvfp4_tensor(param):
            return f"{name}.{attr} is not a packed NVFP4Tensor"
        if getattr(param, "is_swizzled_scales", False):
            return f"{name}.{attr} has swizzled NVFP4 scales"
        if getattr(param, "block_size", 16) != 16:
            return f"{name}.{attr} has NVFP4 block size {param.block_size}, not 16"
        pts = getattr(param, "per_tensor_scale", None)
        if pts is None or pts.numel() not in (1, param.qdata.shape[0]):
            return f"{name}.{attr} has no per-expert NVFP4 per-tensor scale"
    return None


def expert_params(experts) -> set[torch.nn.Parameter]:
    return {getattr(experts, _GATE_UP), getattr(experts, _DOWN)}


def detach_frozen_experts(model) -> list[tuple]:
    """Pop the frozen packed NVFP4 expert params (this rank's EP slice, kept outside FSDP) off
    their modules so a whole-model ``to("meta")`` for cpu_ram_efficient_loading leaves them
    real. Returns ``(module, attr, param)`` for :func:`reattach_frozen_experts`."""
    if not active():
        return []
    from .shard import _detect_experts_modules

    detached = []
    for _name, experts in _detect_experts_modules(model):
        for attr in (_GATE_UP, _DOWN):
            param = experts._parameters.get(attr)
            if param is None or not is_nvfp4_tensor(param) or param.requires_grad:
                continue
            if param.is_meta:
                raise RuntimeError(
                    f"nvfp4_lora: {attr} of {_name} is on meta before FSDP wrapping; the "
                    "expert-parallel scatter should have given every rank its packed slice"
                )
            detached.append((experts, attr, experts._parameters.pop(attr)))
    return detached


def reattach_frozen_experts(detached: list[tuple]) -> None:
    for module, attr, param in detached:
        module._parameters[attr] = param


def move_frozen_experts(experts, device_type: str = "cuda") -> int:
    """Put the frozen expert params on the accelerator in place (FSDP moves only the params
    it manages, and the EP scatter leaves them on the loader's device). Returns the count moved."""
    device = torch.device(device_type, torch.cuda.current_device())
    moved = 0
    for attr in (_GATE_UP, _DOWN):
        param = getattr(experts, attr)
        if param.device == device:
            continue
        with torch.no_grad():
            on_device = param.to(device)
        replacement = torch.nn.Parameter(on_device, requires_grad=param.requires_grad)
        torch.utils.swap_tensors(param, replacement)
        moved += 1
    return moved


def layer_index(module_name: str) -> int | None:
    match = _LAYER_RE.search(f".{module_name}.")
    return int(match.group(1)) if match else None


def _checkpoint_file(repo_id: str, filename: str, revision: str | None = None) -> str:
    """A checkpoint file from a local snapshot dir, the HF cache, or the hub."""
    if os.path.isdir(repo_id):
        return os.path.join(repo_id, filename)
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo_id, filename, revision=revision)
    if isinstance(cached, str):
        return cached
    return hf_hub_download(repo_id, filename, revision=revision)


def _load_index(repo_id: str, revision: str | None = None) -> dict[str, str]:
    name = "model.safetensors.index.json"
    if os.path.isdir(repo_id) and not os.path.exists(os.path.join(repo_id, name)):
        with safe_open(
            os.path.join(repo_id, "model.safetensors"), framework="pt"
        ) as handle:
            return dict.fromkeys(handle.keys(), "model.safetensors")
    with open(_checkpoint_file(repo_id, name, revision)) as handle:
        return json.load(handle)["weight_map"]


def read_input_scales(
    repo_id: str, revision: str | None = None
) -> dict[tuple[int, int, str], float]:
    """``{(layer, expert, projection): input_scale}`` from the checkpoint's safetensors
    (modelopt W4A4 static activation scales) at the revision the weights load from; empty for
    weight-only exports."""
    try:
        weight_map = _load_index(repo_id, revision)
    except Exception as exc:  # pylint: disable=broad-except
        LOG.warning(
            f"nvfp4_lora: cannot read the safetensors index of {repo_id!r} ({exc}); "
            "activation scales default to 1.0"
        )
        return {}
    by_shard: dict[str, list[tuple[str, tuple[int, int, str]]]] = {}
    for key, shard in weight_map.items():
        match = _INPUT_SCALE_RE.search(key)
        if match is None:
            continue
        by_shard.setdefault(shard, []).append(
            (key, (int(match.group(1)), int(match.group(2)), match.group(3)))
        )
    scales: dict[tuple[int, int, str], float] = {}
    for shard, entries in by_shard.items():
        with safe_open(
            _checkpoint_file(repo_id, shard, revision), framework="pt"
        ) as handle:
            for key, index in entries:
                scales[index] = float(handle.get_tensor(key).float().reshape(-1)[0])
    return scales


def expert_adapter_scalings(model):
    """``(experts, scaling)`` per Experts module whose ``gate_up_proj`` and ``down_proj``
    carry an active PEFT ``target_parameters`` adapter with one shared scaling."""
    found: dict[int, tuple] = {}
    for module in model.modules():
        name = getattr(module, "parameter_name", None)
        if name is None or not hasattr(module, "lora_A"):
            continue
        adapters = getattr(module, "active_adapters", None) or ()
        scaling = getattr(module, "scaling", {}).get(adapters[0]) if adapters else None
        if scaling is None:
            continue
        base = module
        while hasattr(base, "base_layer") and hasattr(base, "lora_A"):
            base = base.base_layer
        found.setdefault(id(base), (base, {}))[1][name] = float(scaling)
    for base, scalings in found.values():
        if _GATE_UP in scalings and scalings.get(_DOWN) == scalings[_GATE_UP]:
            yield base, scalings[_GATE_UP]


def register_layer(module: str, entry, cache) -> None:
    # an optmoe without it registers the layer at its first forward
    register = getattr(importlib.import_module(module), "register_layer", None)
    if register is not None:
        register(entry, cache)


def unregister_layers(entries) -> None:
    """Drop the package's registry entries for these layers: it holds each layer and its cache
    strongly, and an ``nvfp4_lora`` layer owns the packed experts on the GPU."""
    try:
        registry = getattr(importlib.import_module("optmoe.lora"), "_LAYERS", None)
    except ImportError:
        return
    if registry is None:
        return
    for entry in entries:
        registry.pop(getattr(entry, "key", None), None)


class Nvfp4LoraState:
    """Per-process state of the ``nvfp4_lora`` path: the shared requant cache, one
    ``Nvfp4LoraExperts`` per MoE layer (built at train begin, else at its first forward)
    and the checkpoint's static activation scales."""

    def __init__(
        self,
        repo_id: str | None = None,
        backend: str | None = None,
        revision: str | None = None,
    ):
        from optmoe.nvfp4_lora import RequantCache

        self.cache = RequantCache()
        self.layers: dict[int, object] = {}
        self.backend = backend
        self.input_scales = read_input_scales(repo_id, revision) if repo_id else {}
        self._hook = register_optimizer_step_post_hook(
            lambda *_: self.cache.invalidate()
        )
        self._warned_scales = False
        self.bytes = 0
        if repo_id and not self.input_scales:
            LOG.warning(
                "nvfp4_lora: the checkpoint carries no expert input_scale tensors; "
                "activations quantize with per-tensor scale 1.0"
            )

    def close(self) -> None:
        self._hook.remove()
        unregister_layers(self.layers.values())
        self.layers.clear()
        self.cache.invalidate()

    def _activation_scales(self, experts, device):
        layer = getattr(experts, "_optmoe_layer_index", None)
        offset = getattr(experts, "local_expert_offset", 0) or 0
        count = experts.num_local_experts
        if layer is None or not self.input_scales:
            return 1.0, 1.0
        x_scales, h_scales = [], []
        for e in range(offset, offset + count):
            xs = [
                self.input_scales[(layer, e, p)]
                for p in _X_SCALE_PROJS
                if (layer, e, p) in self.input_scales
            ]
            hs = [
                self.input_scales[(layer, e, p)]
                for p in _H_SCALE_PROJS
                if (layer, e, p) in self.input_scales
            ]
            if not xs or not hs:
                if not self._warned_scales:
                    LOG.warning(
                        f"nvfp4_lora: layer {layer} expert {e} has no input_scale; "
                        "activations of that layer quantize with per-tensor scale 1.0"
                    )
                    self._warned_scales = True
                return 1.0, 1.0
            x_scales.append(max(xs))
            h_scales.append(max(hs))
        return (
            torch.tensor(x_scales, device=device, dtype=torch.float32),
            torch.tensor(h_scales, device=device, dtype=torch.float32),
        )

    @staticmethod
    def _device_ok(device) -> bool:
        return device.type == "cuda"

    def layer(self, experts, scaling: float):
        entry = self.layers.get(id(experts))
        if entry is not None:
            return entry
        from optmoe.nvfp4_lora import Nvfp4LoraExperts

        gate_up, down = getattr(experts, _GATE_UP), getattr(experts, _DOWN)
        for attr, param in ((_GATE_UP, gate_up), (_DOWN, down)):
            if not is_nvfp4_tensor(param):
                raise RuntimeError(
                    f"nvfp4_lora: experts.{attr} is {type(param).__name__}, not the packed "
                    "NVFP4Tensor the kernels consume (FSDP must leave the frozen experts "
                    "unsharded)"
                )
        device = gate_up.qdata.device
        if not self._device_ok(device):
            raise RuntimeError(
                f"nvfp4_lora: experts.gate_up_proj lives on {device}; the frozen NVFP4 "
                "experts must be on the GPU before the first forward"
            )
        num_experts = gate_up.qdata.shape[0]

        def pts(param):
            scale = param.per_tensor_scale.reshape(-1).float()
            return scale.expand(num_experts) if scale.numel() == 1 else scale

        before = torch.cuda.memory_allocated(device)
        entry = Nvfp4LoraExperts(
            gate_up.qdata,
            gate_up.scale,
            pts(gate_up),
            down.qdata,
            down.scale,
            pts(down),
            scaling=scaling,
            input_scale=self._activation_scales(experts, device),
            backend=self.backend,
        )
        torch.cuda.synchronize(device)
        added = torch.cuda.memory_allocated(device) - before
        self.bytes += added
        self.layers[id(experts)] = entry
        register_layer("optmoe.nvfp4_lora", entry, self.cache)
        if len(self.layers) == 1:
            LOG.info(
                f"nvfp4_lora: first MoE layer built ({num_experts} local experts, "
                f"hidden {entry.hidden}, intermediate {entry.inter}, requant backend "
                f"{entry.backend}, merged cache {self.cache.mode}); the layer adds "
                f"{added / 2**20:.0f} MiB besides the packed base"
                + (
                    " (a stored transposed base)"
                    if entry.transposed_base_bytes()
                    else ""
                )
            )
        return entry

    def prepare_layers(self, model) -> int:
        """Build and register every adapted layer whose packed experts are already on the
        GPU, so a compiled forward creates none."""
        built = 0
        for experts, scaling in expert_adapter_scalings(model):
            gate_up: Any = getattr(experts, _GATE_UP, None)
            if (
                is_nvfp4_tensor(gate_up)
                and is_nvfp4_tensor(getattr(experts, _DOWN, None))
                and self._device_ok(gate_up.qdata.device)
            ):
                self.layer(experts, scaling)
                built += 1
        return built

    def stats(self) -> dict:
        """Requant-cache counters (builds per kind, hits, evictions, resident bytes)."""
        return self.cache.stats()


_STATE: Nvfp4LoraState | None = None


def get_state() -> Nvfp4LoraState | None:
    return _STATE


def set_state(state: Nvfp4LoraState | None) -> None:
    global _STATE
    previous = _STATE
    _STATE = state
    if previous is not None and previous is not state:
        previous.close()


def active() -> bool:
    return _STATE is not None


def tag_layers(model) -> int:
    """Record each Experts module's decoder-layer index for the activation-scale lookup."""
    from .shard import _detect_experts_modules

    tagged = 0
    for name, experts in _detect_experts_modules(model):
        index = layer_index(name)
        if index is not None:
            experts._optmoe_layer_index = index
            tagged += 1
    return tagged


def _wrapper_rank(wrapper) -> int | None:
    ranks = getattr(wrapper, "r", None) or {}
    for adapter in getattr(wrapper, "active_adapters", None) or ():
        if adapter in ranks:
            return int(ranks[adapter])
    return None


def local_expert_lora(lora_a, lora_b, experts, wrapper=None):
    """This EP rank's slice of a PEFT expert adapter: ``(A, B, E_local, rank)``.

    An adapter applied after the EP weight slice is already local (``shard_expert_lora``
    flags its wrapper); a global one is cut to ``[offset : offset + E_local]`` experts
    (``lora_A [E*r, in]`` expert-major rows, ``lora_B [out, r*E]`` rank-major columns)."""
    e_local = experts.num_experts
    e_global = getattr(experts, "num_experts_global", e_local)
    rank = _wrapper_rank(wrapper)
    if rank is None:
        rank = lora_a.shape[0] // e_global
    if (
        e_global == e_local
        or getattr(wrapper, "_ep_lora_sharded", False)
        or lora_a.shape[0] == e_local * rank
    ):
        return lora_a, lora_b, e_local, rank
    offset = getattr(experts, "local_expert_offset", 0)
    a = lora_a[offset * rank : (offset + e_local) * rank, :]
    out = lora_b.shape[0]
    b = lora_b.reshape(out, rank, e_global)[:, :, offset : offset + e_local].reshape(
        out, rank * e_local
    )
    return a, b, e_local, rank


def map_adapter(lora_a, lora_b, num_experts: int, rank: int, in_features: int):
    """PEFT ``target_parameters`` layouts (``lora_A [E*r, in]`` expert-major, ``lora_B
    [out, r*E]`` rank-major) to optmoe's ``A [E, r, in]`` / ``B [E, out, r]`` views."""
    a = lora_a.reshape(num_experts, rank, in_features)
    b = lora_b.reshape(lora_b.shape[0], rank, num_experts).permute(2, 0, 1)
    return a, b.contiguous()


def forward(runtime, experts, hidden_states, topk_idx, topk_w):
    """The fused forward with the adapter the SonicMoE ParamWrapper fastpath handed over."""
    lora = getattr(experts, "_sonicmoe_lora", None) or {}
    if _GATE_UP not in lora or _DOWN not in lora:
        raise RuntimeError(
            "nvfp4_lora: the experts forward received no LoRA factors; the PEFT "
            "lora_target_parameters adapter on experts.gate_up_proj and experts.down_proj "
            "must be active, with use_sonicmoe: true so its ParamWrapper hands the factors "
            "to the experts module"
        )
    state = get_state()
    if state is None:
        raise RuntimeError("nvfp4_lora: state missing; was post_model_load skipped?")
    wrappers = getattr(experts, "_sonicmoe_lora_wrappers", None) or {}
    a1, b1, scaling1 = lora[_GATE_UP]
    a2, b2, scaling2 = lora[_DOWN]
    if float(scaling1) != float(scaling2):
        raise ValueError(
            "nvfp4_lora: gate_up_proj and down_proj adapters must share one LoRA "
            f"scaling, got {scaling1} and {scaling2}"
        )
    a1, b1, num_experts, rank1 = local_expert_lora(
        a1, b1, experts, wrappers.get(_GATE_UP)
    )
    a2, b2, _, rank2 = local_expert_lora(a2, b2, experts, wrappers.get(_DOWN))
    layer = state.layer(experts, float(scaling1))
    if layer.scaling != float(scaling1):
        raise ValueError("nvfp4_lora: the LoRA scaling changed after the first forward")
    a1, b1 = map_adapter(a1, b1, num_experts, rank1, layer.hidden)
    a2, b2 = map_adapter(a2, b2, num_experts, rank2, layer.inter)
    return runtime.forward_lora(
        hidden_states, topk_idx, topk_w, a1, b1, a2, b2, layer, state.cache
    )
