"""Persistence for qualified static native-NVFP4 merge-aware adapters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import TrainerCallback

from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (
    build_native_merge_aware_metadata,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _target_name(name):
    prefix = "base_model.model."
    if prefix in name:
        name = name.split(prefix, maxsplit=1)[1]
    return f"{name}.weight"


def capture_static_native_metadata(model, start_step=None):
    weights = {}
    for name, module in getattr(model, "named_modules", lambda: ())():
        if not hasattr(module, "_axolotl_native_nvfp4_orig_forward"):
            continue
        weight = module.get_base_layer().weight
        if type(weight).__name__ == "NVFP4Tensor":
            weights[_target_name(name)] = weight
    return build_native_merge_aware_metadata(weights, start_step) if weights else None


def write_native_metadata(adapter_dir, metadata):
    path = Path(adapter_dir) / "adapter_config.json"
    if not path.exists() or not metadata:
        return False
    config = json.loads(path.read_text())
    config["nvfp4_merge_aware"] = metadata
    path.write_text(json.dumps(config, indent=2))
    return True


def clear_native_metadata(adapter_dir):
    path = Path(adapter_dir) / "adapter_config.json"
    if path.exists():
        config = json.loads(path.read_text())
        if config.pop("nvfp4_merge_aware", None) is not None:
            path.write_text(json.dumps(config, indent=2))


def update_native_metadata_validity(model):
    valid = getattr(model, "_axolotl_native_nvfp4_metadata_valid", True) and not any(
        getattr(m, "_axolotl_merge_aware_unsupported", False) for m in model.modules()
    )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.distributed.get_backend() == "nccl"
            else torch.device("cpu")
        )
        flag = torch.tensor(int(not valid), device=device)
        torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
        valid = not bool(flag.item())
    model._axolotl_native_nvfp4_metadata_valid = valid
    return valid


def native_metadata_valid_for_save(model):
    return getattr(model, "_axolotl_native_nvfp4_metadata_valid", False) and not any(
        getattr(module, "_axolotl_merge_aware_unsupported", False)
        for module in model.modules()
    )


def prepare_sharded_native_metadata(model):
    """Capture original recipes before distributed wrapping changes weight storage."""
    from peft.tuners.lora.layer import Linear as LoraLinear

    from axolotl.monkeypatch.torchao_nvfp4_merge import _native_merge_aware_reason

    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    source = not distributed or torch.distributed.get_rank() == 0
    payload = [None]
    if source:
        try:
            weights = {}
            for name, module in model.named_modules():
                if not isinstance(module, LoraLinear):
                    continue
                weight = module.get_base_layer().weight
                if type(weight).__name__ != "NVFP4Tensor":
                    continue
                adapters = [a for a in module.active_adapters if a in module.lora_A]
                if (
                    len(adapters) != 1
                    or _native_merge_aware_reason(module, adapters)
                    or weight.ndim != 2
                    or weight.act_quant_kwargs is not None
                ):
                    continue
                weights[_target_name(name)] = weight
            payload[0] = {
                "metadata": build_native_merge_aware_metadata(weights, 0)
                if weights
                else None
            }
        except (RuntimeError, TypeError, ValueError) as error:
            payload[0] = {"error": str(error)}
    if distributed:
        torch.distributed.broadcast_object_list(payload, src=0)
    result = payload[0]
    if result.get("error"):
        model._axolotl_native_nvfp4_metadata_valid = False
        LOG.warning(
            "NVFP4 MERGE WARNING: cannot capture the original sharded quantizer recipe (%s); "
            "adapter saves will not carry a merge-aware guarantee.",
            result["error"],
        )
    model._axolotl_native_nvfp4_metadata = result.get("metadata")


def persist_native_metadata_after_save(model, adapter_dir):
    """Write the cached recipe on the saving rank without entering collectives."""
    metadata = getattr(model, "_axolotl_native_nvfp4_metadata", None)
    if not metadata:
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else int(os.environ.get("RANK", "0"))
    )
    if rank != 0:
        return
    if native_metadata_valid_for_save(model):
        write_native_metadata(adapter_dir, metadata)
    else:
        clear_native_metadata(adapter_dir)


class NativeNVFP4MergeMetadataCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is not None:
            update_native_metadata_validity(model)

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        valid = update_native_metadata_validity(model)
        if state.is_world_process_zero:
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if valid:
                write_native_metadata(
                    path, getattr(model, "_axolotl_native_nvfp4_metadata", None)
                )
            else:
                clear_native_metadata(path)
