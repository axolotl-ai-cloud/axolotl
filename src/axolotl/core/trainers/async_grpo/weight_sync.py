"""Async GRPO's weight sync to vLLM.

Merged sync streams ``W + ΔW`` computed out of place, bitwise equal to the
``lora_merge_aware`` forward, instead of merging into the trained weights; adapter
sync retires each adapter leaving the staleness window rather than deleting it; and
``AliasingVLLMClient`` serves the synced adapter under a fixed name.
"""

from __future__ import annotations

import contextlib
import functools
import os
import shutil
import time
from collections.abc import Callable

import requests
import torch
from accelerate.utils import is_peft_model
from peft.utils.constants import SAFETENSORS_WEIGHTS_NAME
from torch.distributed.tensor import DTensor

from axolotl.core.trainers.async_grpo.trl_compat import (
    AsyncGRPOTrainer,
    VLLMClient,
    iter_vllm_named_params,
    save_lora_adapter,
)
from axolotl.core.trainers.mixins.lora_merge_check import (
    LoraMergeCheckMixin,
)
from axolotl.utils.lora_merge_precision import (
    MergeRetention,
    lora_delta,
    served_dtype_for,
)

# Kept beside the versioned adapter directories, dot-prefixed like TRL's staging ones.
RETIRED_ADAPTER_WEIGHTS = f".retired-{SAFETENSORS_WEIGHTS_NAME}"


class AliasingVLLMClient(VLLMClient):
    """TRL's vLLM client that also loads every synced adapter under ``alias``.

    vLLM keys its prefix cache on the adapter name, which the alias keeps across
    versions, so a sync drains in-flight requests (pausing with ``mode=wait`` also
    clears the cache) rather than freezing them under the old adapter.
    """

    def __init__(self, *args, alias: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.alias = alias

    def pause(self, timeout: int = 1800) -> None:
        response = requests.post(
            f"{self.server_url}/pause", params={"mode": "wait"}, timeout=timeout
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Pausing vLLM failed: {response.status_code}, {response.text}"
            )

    def load_lora_adapter(
        self, lora_name: str, lora_path: str, timeout: int = 1800
    ) -> None:
        super().load_lora_adapter(lora_name, lora_path, timeout)
        self.unload_lora_adapter(self.alias, timeout)
        super().load_lora_adapter(self.alias, lora_path, timeout)


def _full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.detach()


def _lora_delta(module, adapter: str) -> torch.Tensor:
    """FP32 delta of ``adapter``. A LoRA linear's comes from its gathered factors, as
    the ``lora_merge_aware`` forward and PEFT's merge compute it; other layers'
    from PEFT."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    if isinstance(module, LoraLinear) and not module.fan_in_fan_out:
        return lora_delta(
            _full(module.lora_A[adapter].weight),
            _full(module.lora_B[adapter].weight),
            module.scaling[adapter],
        )
    return _full(module.get_delta_weight(adapter)).float()


def _measures_merge(module, adapters: list[str]) -> bool:
    """Whether ``measure_lora_merge_retention`` would count this layer's rounding:
    a plain single-adapter LoRA linear that does not already train merge-aware."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    return (
        isinstance(module, LoraLinear)
        and not module.fan_in_fan_out
        and len(adapters) == 1
        and adapters[0] not in module.lora_variant
        and getattr(module, "_axolotl_lora_merge_aware_dtype", None) is None
    )


def _merged_weight(
    module,
    adapters: list[str],
    weight: torch.Tensor,
    retention: MergeRetention | None = None,
) -> torch.Tensor:
    """``weight`` with ``adapters`` folded in and rounded once, leaving ``weight``
    itself untouched; with ``retention``, also how much of the delta the rounding
    keeps."""
    base = _full(weight).float()
    merged = base
    for adapter in adapters:
        if adapter not in module.lora_variant:
            merged = merged + _lora_delta(module, adapter)
            continue
        # Variants (DoRA) rebuild the merge from the module's own sharded weights.
        if isinstance(weight, DTensor) or len(adapters) > 1:
            raise NotImplementedError(
                f"Merged weight sync of a {type(module.lora_variant[adapter]).__name__} "
                "adapter is only supported for a single adapter without FSDP."
            )
        merged = module.lora_variant[adapter].merge_safe(module, adapter, merged)
    if retention is not None and _measures_merge(module, adapters):
        served = served_dtype_for(weight.dtype)
        retention.update(
            merged - base, merged.to(served).float() - base.to(served).float()
        )
        retention.modules += 1
    return merged.to(weight.dtype)


def _merged_bias(
    module,
    adapters: list[str],
    bias: torch.Tensor,
    retention: MergeRetention | None = None,
) -> torch.Tensor:
    merged = _full(bias).float()
    for adapter in adapters:
        if module.lora_bias.get(adapter):
            lora_bias = _full(module.lora_B[adapter].bias).float()
            merged = merged + lora_bias * module.scaling[adapter]
    return merged.to(bias.dtype)


def _out_of_place_merges(model) -> dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    """Map each base parameter a LoRA layer adapts to the function that merges it."""
    from peft.tuners.lora import LoraLayer

    merges: dict[int, Callable[[torch.Tensor], torch.Tensor]] = {}
    for module in model.modules():
        if not isinstance(module, LoraLayer) or module.disable_adapters:
            continue
        present = set(module.lora_A) | set(module.lora_embedding_A)
        adapters = [a for a in module.active_adapters if a in present]
        if not adapters:
            continue
        if module.merged:
            raise RuntimeError(
                f"{type(module).__name__} already has its adapter merged, so its base "
                "weight would be sent with the adapter applied twice."
            )
        base_layer = module.get_base_layer()
        weight = getattr(base_layer, getattr(module, "parameter_name", "weight"), None)
        if not isinstance(weight, torch.Tensor):
            raise NotImplementedError(
                f"Merged weight sync does not support {type(module).__name__} layers."
            )
        merges[id(weight)] = functools.partial(_merged_weight, module, adapters)
        if any(module.lora_bias.get(a) for a in adapters):
            merges[id(base_layer.bias)] = functools.partial(
                _merged_bias, module, adapters
            )
    return merges


class WeightSyncMixin(LoraMergeCheckMixin, AsyncGRPOTrainer):
    """TRL's weight sync, skipped after steps that trained nothing, merging out of
    place and retiring adapters instead of deleting them."""

    model_version: int
    # Whether this process wrote the weights that retired adapters link to.
    _wrote_retired_weights = False

    def _sync_weight(self):
        if not self._weights_updated:
            # Only steps that trained nothing since the last sync: same policy.
            self._metrics["train"]["perf/weight_sync_skipped_total"].append(1.0)
            return
        super()._sync_weight()
        self._weights_updated = False

    def _sync_weight_merged(self, t0: float) -> tuple[float, float, float]:
        """TRL's merged sync, without folding the adapter into the trained weights.

        TRL merges the adapter in place for the send and unmerges it afterwards. In
        bf16 the round trip does not restore the base weights, so every sync nudges the
        frozen model; `_streaming_iter` streams ``W + ΔW`` computed out of place instead.
        """
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.pause()
        t_pause = time.time()
        self.accelerator.wait_for_everyone()
        t_barrier = time.time()
        if self.accelerator.is_main_process and self.weight_transfer:
            self.weight_transfer.send_weights(self._streaming_iter())
        else:
            # The other ranks still take part in each parameter's all-gather.
            for _ in self._streaming_iter():
                pass
        t_transfer = time.time()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.weight_transfer:
                self.weight_transfer.resume()
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _sync_weight_lora(self, t0: float) -> tuple[float, float, float]:
        """TRL's adapter sync, except that an adapter leaving the staleness window is
        retired rather than deleted.

        TRL pauses vLLM with ``mode=keep``, so requests for an adapter it unloaded can
        stay queued. vLLM's adapter cache evicts it all the same, so serving them
        reloads it from disk, and a missing directory kills the engine.
        """
        version = self.model_version + 1
        name = f"{self._lora_name}-v{version}"
        adapter_dir = os.path.join(self._lora_dir, name)
        model = self.accelerator.unwrap_model(self.model)
        if self.accelerator.is_main_process:
            self.vllm_client.pause()
        t_pause = time.time()
        self.accelerator.wait_for_everyone()
        t_barrier = time.time()
        # Every rank gathers its shards of the adapter; only the main process writes.
        save_lora_adapter(model, self.accelerator, self._adapter_name, adapter_dir)
        t_transfer = time.time()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Loaded before the version moves: rollouts request the adapter it names.
            self.vllm_client.load_lora_adapter(name, adapter_dir)
            self.vllm_client.resume()
            stale_version = version - (self.args.max_staleness + 1)
            if stale_version > 0:
                self.vllm_client.unload_lora_adapter(
                    f"{self._lora_name}-v{stale_version}"
                )
            if stale_version > 1:
                self._retire_adapter(
                    os.path.join(
                        self._lora_dir, f"{self._lora_name}-v{stale_version - 1}"
                    )
                )
            self.model_version = version
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        return t_pause, t_barrier, t_transfer

    def _retire_adapter(self, adapter_dir: str) -> None:
        """Point ``adapter_dir``'s weights at one kept copy, so it stays loadable
        without its own copy of the weights.

        Requests still naming it are over ``max_staleness`` versions old, so their
        samples are dropped and any weights of the adapter's shape can serve them.
        """
        weights = os.path.join(adapter_dir, SAFETENSORS_WEIGHTS_NAME)
        if os.path.islink(weights) or not os.path.isfile(weights):
            return
        kept = os.path.join(self._lora_dir, RETIRED_ADAPTER_WEIGHTS)
        if not self._wrote_retired_weights:
            # Rewritten by each process, so it has the shape of this run's adapter.
            shutil.copyfile(weights, f"{kept}.tmp")
            os.replace(f"{kept}.tmp", kept)
            self._wrote_retired_weights = True
        link = f"{weights}.tmp"
        with contextlib.suppress(FileNotFoundError):
            os.remove(link)
        os.symlink(os.path.relpath(kept, adapter_dir), link)
        # Atomic: vLLM reading the adapter sees either file, never neither.
        os.replace(link, weights)

    def _streaming_iter(self):
        model = self.accelerator.unwrap_model(self.model)
        if not is_peft_model(model):
            yield from super()._streaming_iter()
            return
        merges = _out_of_place_merges(model)
        device = self.accelerator.device
        retention = MergeRetention() if self._lora_merge_check_due() else None
        # The same walk as the weight manifest TRL sends the server at init.
        for name, param in iter_vllm_named_params(model):
            with torch.no_grad():
                merge = merges.get(id(param))
                tensor = merge(param, retention) if merge is not None else _full(param)
            yield name, tensor.to(device)
        if retention is not None:
            self._record_lora_merge_check(retention)
