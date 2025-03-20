"""
module to handle loading model on cpu/meta device for FSDP
"""

import os
import time
from typing import List, Optional, Type, Union

import safetensors
import torch
from accelerate import init_empty_weights
from bitsandbytes.nn import Linear4bit, Params4bit
from fastcore.parallel import parallel
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.quantizers import AutoHfQuantizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, hub


def _replace_linear(
    model: nn.Module,
    linear_replacement: Type[nn.Module],
    quant_config: Union[dict, None] = None,
    skip_modules=None,
    **kwargs,
):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    if skip_modules is None:
        skip_modules = ["lm_head"]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            _replace_linear(
                module, linear_replacement, quant_config, skip_modules, **kwargs
            )

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            if issubclass(linear_replacement, Linear4bit):
                model._modules[name] = (  # pylint: disable=protected-access
                    linear_replacement(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        **kwargs,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported linear replacement: {type(linear_replacement)}"
                )
    return model


def load_and_quantize(
    module: nn.Module,
    name: str,
    value: Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
    skip_names: Optional[List[str]] = None,
    to_cpu: bool = False,
    to_meta: bool = False,
    verbose: bool = False,
    quant_method: str = "bnb",
):
    """
    Loads `value` tensor into submodule of `module`, optionally skipping `skip_names` and converting to `dtype`.

    Quantizes `Params4bit` on `device` then places on "cpu" if to_cpu=True or "meta" if to_meta=True.
    """

    if not skip_names:
        skip_names = []

    def place_on_device(value):
        if to_meta:
            device = "meta"
        elif to_cpu:
            device = "cpu"
        return value.to(device=device, dtype=dtype)

    if any(skip_name in name for skip_name in skip_names):
        if verbose:
            print(f"Skipping {name} because it is in skip_names")
        return

    module_key, _, value_key = name.rpartition(".")
    try:
        submodule = module.get_submodule(module_key)
    except AttributeError as exc:
        print(f"Module {module_key} not found:\n{exc}")
        return

    try:
        if quant_method == "bnb":
            param = submodule.get_parameter(value_key)
            if isinstance(param, Params4bit):
                # With `sync_module_states=True`, a meta device Params4bit needs to be the same
                # shape as the quantized Params4bit with an initialized quant_state. However,
                # FSDP only syncs parameters and buffers, so the quant_state isn't copied. This
                # workaround quantizes Params4bit to initialize quant_state on all ranks, then
                # replaces Params4bit's data with a meta tensor to free memory on non-rank 0.
                value = type(param)(
                    value.to(device=device, dtype=dtype).data, **param.__dict__
                ).cuda(device)
                if to_meta:
                    value = type(param)(value.data.to("meta"), **value.__dict__)
                elif to_cpu:
                    value = type(param)(value.data.to("cpu"), **value.__dict__)
            else:
                value = type(param)(place_on_device(value).data)

    except AttributeError:
        # it's a buffer
        value = place_on_device(value)

    setattr(submodule, value_key, value)


def n_loading_workers(quant_method: str, param_count: float):
    devprops = torch.cuda.get_device_properties(torch.cuda.current_device())
    left = int(os.cpu_count() / torch.cuda.device_count())
    model_params_b = 70
    right = int(
        (4 if quant_method == "hqq" else 8)
        * (devprops.total_memory / 1e9 / 40)
        * (model_params_b / (param_count / 1e9))
    )
    return min(left, right)


def load_sharded_model(
    model_name,
    model_config,
    cfg,
    torch_dtype=torch.bfloat16,
    low_memory=True,
):
    if (low_memory and cfg.local_rank == 0) or not low_memory:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            use_cache=False,
            torch_dtype=torch.float32,
            _attn_implementation=model_config._attn_implementation,  # pylint: disable=protected-access
            trust_remote_code=cfg.trust_remote_code,
        )
        dtype = torch_dtype if not cfg.float32 else None
        model.to(dtype=dtype, device="cpu" if low_memory else cfg.local_rank)
    else:
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                model_config,
                torch_dtype=torch_dtype,
                trust_remote_code=cfg.trust_remote_code,
            )
    return model


def load_sharded_model_quant(
    model_name,
    model_config,
    cfg,
    compute_dtype=torch.bfloat16,
    quant_storage=torch.float32,
    low_memory=True,
    verbose=False,
    loading_workers=2,
    quantization_config=None,
):
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            model_config,
            trust_remote_code=cfg.trust_remote_code,
        )
        if hasattr(model, "transformer"):
            model.transformer = _replace_linear(
                model.transformer,
                Linear4bit,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                quant_storage=quant_storage,
                compress_statistics=True,  # bnb_4bit_use_double_quant
                skip_modules=[
                    "lm_head",
                    "embed_out",
                ],
            )
        else:
            # this is the more common case with HF transformers
            # TODO can we detect the model arch and dynamically set skip_modules
            model.model = _replace_linear(
                model.model,
                Linear4bit,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                quant_storage=quant_storage,
                compress_statistics=True,  # bnb_4bit_use_double_quant
                skip_modules=[
                    "lm_head",
                    "embed_out",
                ],
            )
    model.is_loaded_in_4bit = True

    # Grab the safetensors files that hold the weights
    try:
        idx = hub.cached_file(model_name, SAFE_WEIGHTS_INDEX_NAME)
        files, _ = hub.get_checkpoint_shard_files(model_name, idx)
    except OSError:
        try:
            # This means the model doesn't have a model.safetensors.index.json because it is not sharded
            files = []
            files.append(hub.cached_file(model_name, SAFE_WEIGHTS_NAME))
        except OSError as exc:
            # This means the model probably doesn't have a safetensors file
            raise exc

    # Load in the weights, using our custom load_and_quantize method which quantizes Params4bit on the fly
    # and then places each layer on CPU or meta if using low_memory to minimize GPU memory usage
    def load_and_quantize_parallel(name_param, model, **kwargs):
        name, param = name_param
        load_and_quantize(model, name, param, **kwargs)

    quant_method = "bnb"
    param_count = sum((p.numel() for n, p in model.named_parameters()))

    n_workers = (
        n_loading_workers(quant_method, param_count)
        if loading_workers == -1
        else loading_workers
    )
    if cfg.local_rank == 0 and verbose:
        print(f"Using n_workers: {n_workers} for loading")

    start = time.time()
    for filename in tqdm(
        files,
        desc="Loading & Quantizing Model Shards",
        disable=cfg.local_rank != 0,
        position=0,
    ):
        weights = safetensors.torch.load_file(filename)
        parallel(
            load_and_quantize_parallel,
            iter(weights.items()),
            n_workers=n_workers,
            threadpool=True,
            model=model,
            dtype=quant_storage,
            device=cfg.local_rank,
            skip_names=[],
            to_cpu=(low_memory and cfg.local_rank == 0),
            to_meta=(low_memory and cfg.local_rank != 0),
            verbose=verbose,
            quant_method=quant_method,
        )

    # these attributes are needed to inform transformers/peft of the quantization
    model.is_quantized = True
    model.quantization_method = "bitsandbytes"
    model.hf_quantizer = AutoHfQuantizer.from_config(quantization_config)

    if cfg.local_rank == 0 and verbose:
        print(f"Loaded model weights in {time.time() - start:.3f} seconds")
    # cleanup any extra memory usage from parallel loading
    torch.cuda.empty_cache()

    return model
