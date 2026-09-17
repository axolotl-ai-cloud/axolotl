"""CPU-staged NF4 loading through Transformers' checkpoint conversion pipeline."""

import copy
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.utils import parametrize

from axolotl.utils.dict import DictDefault
from axolotl.utils.nf4_loading import (
    nf4_loading_device,
    nf4_loading_group,
    nf4_phase,
    record_progress,
)

if TYPE_CHECKING:
    from transformers import BitsAndBytesConfig, PretrainedConfig, PreTrainedModel
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

from axolotl.utils.nf4 import (
    BnbNF4Parametrization,
    checkpoint_nf4_linear,
    quantize_bnb_4bit,
    quantize_torchao_nf4,
)


def load_nf4_model(
    loader: type["PreTrainedModel"] | type["_BaseAutoModelClass"],
    model_config: "PretrainedConfig",
    model_kwargs: dict[str, Any],
    cfg: DictDefault,
    quantization_device: torch.device | str | None = None,
) -> nn.Module:
    """Load quantized CPU weights on rank zero and matching meta parameters on peers.

    Args:
        loader: Transformers model factory exposing from_pretrained.
        model_config: Resolved base-model configuration.
        model_kwargs: Loader kwargs, including the resolved quantization configuration.
        cfg: Axolotl loading and quantization settings.
        quantization_device: Override the Accelerate device for chunk conversion.

    Returns:
        A frozen model ready for adapter creation and FSDP2 sharding.
    """
    with nf4_loading_group(cfg) as control_group:
        return _load_nf4_model(
            loader, model_config, model_kwargs, cfg, quantization_device, control_group
        )


def _load_nf4_model(
    loader, model_config, model_kwargs, cfg, quantization_device, control_group
):
    import torch.distributed as dist
    from accelerate import init_empty_weights

    from axolotl.monkeypatch.moe_quant import patch_peft_target_parameters_matching
    from axolotl.monkeypatch.peft.nf4 import patch_nf4_merge

    if cfg.fsdp_config:
        from axolotl.monkeypatch.accelerate.fsdp2_nf4 import patch_nf4_adapter_state

        patch_nf4_adapter_state()
    patch_nf4_merge()
    patch_peft_target_parameters_matching()
    from axolotl.monkeypatch.moe_quant import _moe_load_state

    _moe_load_state["count"] = 0
    _moe_load_state["expert_param_order"] = {}

    quantization = model_kwargs.pop("quantization_config", None)
    distributed = bool(cfg.fsdp_config)
    main = not distributed or dist.get_rank() == 0

    def empty_model():
        with init_empty_weights():
            options = {"dtype": model_kwargs.get("dtype", cfg.torch_dtype)}
            for name in ("attn_implementation", "experts_implementation"):
                if name in model_kwargs:
                    options[name] = model_kwargs[name]
            if hasattr(loader, "from_config"):
                return loader.from_config(
                    model_config,
                    trust_remote_code=cfg.trust_remote_code or False,
                    **options,
                )
            return loader._from_config(model_config, **options)

    error = None
    model = None
    structures = []
    if main:
        try:
            device = quantization_device or nf4_loading_device()
            with (
                nf4_phase("NF4 checkpoint loading and quantization"),
                staged_nf4_loading(
                    cfg, device=device, quantization_config=quantization
                ),
            ):
                model = loader.from_pretrained(
                    cfg.base_model, config=model_config, **model_kwargs
                )
            # before the status exchange, so peers never block on a broadcast a
            # failed rank zero cannot reach
            if distributed:
                with nf4_phase("NF4 metadata preparation"):
                    _collect_nf4_structures(model, structures)
        except Exception as exc:
            if not distributed:
                raise
            error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            model = empty_model()
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    if distributed:
        _raise_on_any_rank_failure(error, control_group)
        # the broadcast shapes are packed NF4 storage, so peers cannot re-derive which
        # parameters were experts; take rank zero's classification instead
        payload = [
            structures,
            _moe_load_state["expert_param_order"] if main else None,
            _moe_load_state["count"] if main else None,
        ]
        with nf4_phase("NF4 metadata broadcast"):
            dist.broadcast_object_list(
                payload, src=0, group=control_group, device=torch.device("cpu")
            )
        if not main:
            try:
                _apply_nf4_structures(model, payload, _moe_load_state)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        del payload, structures
        _raise_on_any_rank_failure(error, control_group)
    model.requires_grad_(False)
    model._axolotl_staged_nf4 = True
    return model


def _raise_on_any_rank_failure(error, control_group):
    """Abort every rank with the real cause when any rank failed."""
    import torch.distributed as dist

    statuses: list = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(statuses, error, group=control_group)
    failures = [f"rank {rank}: {msg}" for rank, msg in enumerate(statuses) if msg]
    if failures:
        raise RuntimeError("NF4 loading failed on " + "; ".join(failures))


def _apply_nf4_structures(model, payload, moe_load_state):
    moe_load_state["expert_param_order"] = payload[1]
    moe_load_state["count"] = payload[2]
    for path, name, transform, shape, dtype in payload[0]:
        module = model.get_submodule(path)
        setattr(
            module,
            name,
            nn.Parameter(
                torch.empty(shape, dtype=dtype, device="meta"), requires_grad=False
            ),
        )
        if isinstance(module, nn.Linear) and name == "weight":
            checkpoint_nf4_linear(module)
        parametrize.register_parametrization(module, name, transform, unsafe=True)


def _collect_nf4_structures(model, structures):
    for path, module in model.named_modules():
        for name, chain in getattr(module, "parametrizations", {}).items():
            transform = chain[0]
            memo = {
                id(buffer): torch.empty_like(buffer, device="meta")
                for buffer in transform.buffers()
            }
            structures.append(
                (
                    path,
                    name,
                    copy.deepcopy(transform, memo),
                    tuple(chain.original.shape),
                    chain.original.dtype,
                )
            )


def uses_staged_nf4(cfg: DictDefault) -> bool:
    return bool(
        cfg.load_in_4bit
        and (
            cfg.get("nf4_backend") == "torchao"
            or (str(cfg.fsdp_version) == "2" and cfg.qlora_sharded_model_loading)
        )
    )


@contextmanager
def staged_nf4_loading(
    cfg: DictDefault,
    device: torch.device | str | None = None,
    quantization_config: "BitsAndBytesConfig | None" = None,
) -> Iterator[None]:
    """Temporarily intercept Transformers loading to convert selected weights in chunks."""
    import transformers.core_model_loading as loading
    import transformers.modeling_utils as modeling

    from axolotl.loaders.nf4_prefetch import prefetch_nf4_weights
    from axolotl.monkeypatch.moe_quant import _moe_load_state

    original = loading.set_param_for_module
    device = device or nf4_loading_device()
    storage = "cpu"
    backend = cfg.get("nf4_backend") or "bitsandbytes"
    from axolotl.utils.nf4 import nf4_should_quantize, nf4_skip_modules

    quantization = (
        quantization_config.to_dict()
        if quantization_config is not None
        else dict(cfg.bnb_config_kwargs or {})
    )
    skips = nf4_skip_modules(cfg.model_config_type, quantization)

    def set_param(model, target_name, param_value, *args, **kwargs):
        value = param_value
        path, _, name = target_name.rpartition(".")
        module = model.get_submodule(path) if path else model
        expert = (
            cfg.quantize_moe_experts
            and value.ndim >= 3
            and "expert" in target_name.lower()
        )
        linear = isinstance(module, nn.Linear) and name == "weight"
        original(model, target_name, value, *args, **kwargs)
        if value.is_meta or not nf4_should_quantize(
            target_name, linear=linear, expert=expert, skips=skips
        ):
            return
        if expert and path not in _moe_load_state["expert_param_order"]:
            _moe_load_state["expert_param_order"][path] = list(module._parameters)
        if backend == "torchao":
            data, transform = quantize_torchao_nf4(
                value, device=device, storage_device=storage
            )
        else:
            data, state = quantize_bnb_4bit(
                value,
                device=device,
                storage_device=storage,
                blocksize=quantization.get("blocksize", 64),
                quant_type=quantization.get("bnb_4bit_quant_type", "nf4"),
                compress_statistics=quantization.get("bnb_4bit_use_double_quant", True),
            )
            transform = BnbNF4Parametrization(state)
        setattr(module, name, nn.Parameter(data, requires_grad=False))
        if linear:
            checkpoint_nf4_linear(module)
        parametrize.register_parametrization(module, name, transform, unsafe=True)
        record_progress(value.numel() * value.element_size())
        if expert:
            _moe_load_state["count"] += 1

    # Transformers must materialize rank zero even when its FSDP meta-loading gate is enabled.
    with (
        patch.dict(
            os.environ,
            {
                "FSDP_CPU_RAM_EFFICIENT_LOADING": "false",
                "HF_ENABLE_PARALLEL_LOADING": "false",
                "HF_DEACTIVATE_ASYNC_LOAD": "true",
            },
        ),
        prefetch_nf4_weights(int(cfg.get("nf4_prefetch_memory_mb", 1024)) * 1024**2),
        patch.object(loading, "set_param_for_module", set_param),
        patch.object(modeling, "caching_allocator_warmup", lambda *a, **k: None),
    ):
        yield
