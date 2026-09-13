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
from axolotl.utils.nf4_loading import nf4_loading_device, nf4_loading_group, nf4_phase

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
    if main:
        try:
            from axolotl.loaders.nf4_cache import (
                load_nf4_cache,
                nf4_cache_path,
                save_nf4_cache,
            )
            from axolotl.utils.logging import get_logger

            device = quantization_device or nf4_loading_device()
            cache = nf4_cache_path(
                cfg, model_config, model_kwargs, quantization, device
            )
            if cache is not None and cache.is_file():
                get_logger(__name__).info("Loading packed NF4 cache: %s", cache)
                model = load_nf4_cache(cache, empty_model)
            else:
                with (
                    nf4_phase("NF4 checkpoint loading and quantization"),
                    staged_nf4_loading(
                        cfg, device=device, quantization_config=quantization
                    ),
                ):
                    model = loader.from_pretrained(
                        cfg.base_model, config=model_config, **model_kwargs
                    )
                if cache is not None:
                    save_nf4_cache(cache, model)
                    get_logger(__name__).info("Saved packed NF4 cache: %s", cache)
        except Exception as exc:
            if not distributed:
                raise
            error = f"{type(exc).__name__}: {exc}"
    if distributed:
        status = [error]
        dist.broadcast_object_list(
            status, src=0, group=control_group, device=torch.device("cpu")
        )
        if status[0] is not None:
            raise RuntimeError(f"Rank-zero NF4 loading failed: {status[0]}")
    if not main:
        model = empty_model()
    if distributed:
        structures = []
        if main:
            with nf4_phase("NF4 metadata preparation"):
                _collect_nf4_structures(model, structures)
        payload = [structures]
        with nf4_phase("NF4 metadata broadcast"):
            dist.broadcast_object_list(
                payload, src=0, group=control_group, device=torch.device("cpu")
            )
        if not main:
            from axolotl.monkeypatch.moe_quant import _moe_load_state

            for path, name, transform, shape, dtype in payload[0]:
                module = model.get_submodule(path)
                if "expert" in path:
                    _moe_load_state["expert_param_order"].setdefault(
                        path, list(module._parameters)
                    )
                    _moe_load_state["count"] += 1
                setattr(
                    module,
                    name,
                    nn.Parameter(
                        torch.empty(shape, dtype=dtype, device="meta"),
                        requires_grad=False,
                    ),
                )
                if isinstance(module, nn.Linear) and name == "weight":
                    checkpoint_nf4_linear(module)
                parametrize.register_parametrization(
                    module, name, transform, unsafe=True
                )
        del payload, structures
    model.requires_grad_(False)
    model._axolotl_staged_nf4 = True
    return model


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
    skips.update(cfg.lora_modules_to_save or [])

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
