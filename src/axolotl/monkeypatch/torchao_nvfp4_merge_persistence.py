"""Persistence for qualified static native-NVFP4 merge-aware adapters."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import TrainerCallback

from axolotl.monkeypatch.torchao_nvfp4_merge_metadata import (
    build_native_merge_aware_metadata,
)


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
