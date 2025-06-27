"""
monkeypatch for accelerate fsdp2 fix when modifying ordereddict during interation, and saving full state dicts
"""

import sys

import torch
import torch.nn as nn

from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def fsdp2_load_full_state_dict(
    accelerator, model: torch.nn.Module, full_sd: dict, offload_to_cpu: bool = False
):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.
    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    from torch.distributed.tensor import distribute_tensor

    LOG.info("Broadcasting full state dict to all ranks...")
    import time

    start_time = time.time()
    # Rank 0 distributes the full state dict to other ranks

    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = (
            is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
        )

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    # param_names = sorted(meta_sharded_sd.keys())
    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}
    for param_name, full_tensor in full_sd.items():
        sharded_meta_param = meta_sharded_sd.get(param_name)
        # if accelerator.is_main_process:
        #     full_tensor = full_tensor.cuda()
        #     # if not hasattr(sharded_meta_param, "device_mesh"):
        #     dist.broadcast(full_tensor, src=0,
        #                    group=sharded_meta_param.device_mesh.get_group())
        # else:
        #     full_tensor = torch.empty_like(sharded_meta_param, device="cuda")
        #     dist.broadcast(full_tensor, src=0,
        #                    group=sharded_meta_param.device_mesh.get_group())
        # Clear immediately
        # elif not hasattr(sharded_meta_param, "device_mesh"):
        #     full_tensor = torch.empty_like(sharded_meta_param, device="cuda")
        #     dist.broadcast(full_tensor, src=0, group=sharded_meta_param.device_mesh.get_group())
        # else:
        #     # Non-main ranks: distribute_tensor with src_data_rank=0 handles this
        #     full_tensor = torch.empty_like(sharded_meta_param, device="cuda")
        #     dist.broadcast(full_tensor, src=0,
        #                    group=sharded_meta_param.device_mesh.get_group())

        # distribute_tensor will internally handle the broadcasting/scattering from rank 0

        # else:
        # if hasattr(sharded_meta_param, "device_mesh"):
        full_tensor = full_tensor.to(sharded_meta_param.dtype).to(torch.device("cuda"))
        # LOG.info(f"param name: {param_name}")
        # LOG.info(f"sharded_meta_param: {sharded_meta_param}")
        # torch.distributed.barrier()
        if hasattr(sharded_meta_param, "device_mesh"):
            sharded_param = distribute_tensor(
                full_tensor,
                sharded_meta_param.device_mesh,
                sharded_meta_param.placements,
                src_data_rank=0,  # This is the default, but being explicit
            )
        else:
            sharded_param = full_tensor
        # else:
        #     sharded_param = full_tensor

        # if "embed_tokens" in param_name or "lm_head" in param_name:
        #     print(f"param_name: {param_name}\nfull_tensor: {full_tensor}\nsharded_param: {sharded_param}\nsharded_meta_param: {sharded_meta_param}")
        if offload_to_cpu:
            sharded_param = sharded_param.cpu()

        sharded_sd[param_name] = nn.Parameter(sharded_param)
        del full_tensor
        full_sd[param_name] = None
    # for param_name in param_names:
    #     mesh = meta_sharded_sd[param_name].device_mesh
    #     if accelerator.is_main_process:
    #         full_param = full_sd[param_name].detach().cuda()
    #         dist.broadcast(full_param, src=0, group=mesh.get_group())
    #         sharded_tensor = distribute_tensor(
    #             full_param, mesh, sharded_sd[param_name].placements
    #         )
    #         to_contiguous, casting_dtype = _infer_parameter_dtype(
    #             model,
    #             param_name,
    #             full_param,
    #         )
    #         sharded_tensor = _cast_and_contiguous(
    #             sharded_tensor, to_contiguous, casting_dtype
    #         )
    #         sharded_sd[param_name] = sharded_tensor
    #     else:
    #         full_tensor = torch.empty(
    #             sharded_sd[param_name].size(),
    #             device="cuda",
    #             dtype=sharded_sd[param_name].dtype,
    #         )
    #         dist.broadcast(full_tensor, src=0, group=mesh.get_group())
    #         sharded_tensor = distribute_tensor(
    #             full_tensor, sharded_param.device_mesh, sharded_param.placements
    #         )
    #         to_contiguous, casting_dtype = _infer_parameter_dtype(
    #             model,
    #             param_name,
    #             full_tensor,
    #         )
    #         sharded_tensor = _cast_and_contiguous(
    #             sharded_tensor, to_contiguous, casting_dtype
    #         )
    #         sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    # raise ValueError("Stop here")
    model.load_state_dict(sharded_sd, assign=True, strict=True)
    end_time = time.time()
    LOG.info(
        f"Time taken to load full state dict: {(end_time - start_time):.2f} seconds"
    )
    log_gpu_memory_usage(LOG, "Memory usage after broadcasting full state dict", 0)
    # raise ValueError("Stop here")
    return model


def get_state_dict(self, model, unwrap=True):
    """
    Returns the state dictionary of a model sent through [`Accelerator.prepare`] potentially without full
    precision.

    Args:
        model (`torch.nn.Module`):
            A PyTorch model sent through [`Accelerator.prepare`]
        unwrap (`bool`, *optional*, defaults to `True`):
            Whether to return the original underlying state_dict of `model` or to return the wrapped state_dict

    Returns:
        `dict`: The state dictionary of the model potentially without full precision.

    Example:

    ```python
    >>> import torch
    >>> from accelerate import Accelerator

    >>> accelerator = Accelerator()
    >>> net = torch.nn.Linear(2, 2)
    >>> net = accelerator.prepare(net)
    >>> state_dict = accelerator.get_state_dict(net)
    ```
    """
    from accelerate import DistributedType
    from accelerate.utils import compare_versions

    if self.distributed_type == DistributedType.DEEPSPEED:
        zero3_sharding = self.deepspeed_config["zero_optimization"]["stage"] == 3
        tp_sharding = (
            self.deepspeed_config.get("tensor_parallel", {}).get("autotp_size", 0) > 1
        )
        if zero3_sharding or tp_sharding:
            if model.zero_gather_16bit_weights_on_model_save():
                if tp_sharding and not compare_versions("deepspeed", ">=", "0.16.4"):
                    raise ImportError(
                        "Deepspeed TP requires deepspeed >= 0.16.4, Please update DeepSpeed via `pip install deepspeed -U`."
                    )
                state_dict = (
                    model._consolidated_16bit_state_dict()  # pylint: disable=protected-access
                    if tp_sharding
                    else model._zero3_consolidated_16bit_state_dict()  # pylint: disable=protected-access
                )
            else:
                raise ValueError(
                    "Cannot get 16bit model weights because `stage3_gather_16bit_weights_on_model_save` in DeepSpeed config is False. "
                    "To save the model weights in 16bit, set `stage3_gather_16bit_weights_on_model_save` to True in DeepSpeed config file or "
                    "set `zero3_save_16bit_model` to True when using `accelerate config`. "
                    "To save the full checkpoint, run `model.save_checkpoint(save_dir)` and use `zero_to_fp32.py` to recover weights."
                )
        else:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

            state_dict = clone_tensors_for_torch_save(
                self.unwrap_model(model).state_dict()
            )
    elif self.is_fsdp2:
        # https://github.com/pytorch/torchtune/blob/main/torchtune/training/_distributed.py#L465
        state_dict = {}
        sharded_state_dict = model.state_dict()
        for param_name, param in sharded_state_dict.items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))

            param = param.full_tensor()
            if torch.distributed.get_rank() == 0:
                state_dict[param_name] = param.cpu()
            torch.distributed.barrier()
    elif self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import FullStateDictConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import StateDictType

        full_state_dict_config = FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, full_state_dict_config
        ):
            state_dict = model.state_dict()
    else:
        if unwrap:
            model = self.unwrap_model(model)
        state_dict = model.state_dict()

    return state_dict


def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare

    Returns:
        `torch.nn.Module`: Prepared model
    """
    import copy
    import functools

    from accelerate.utils import get_module_children_bottom_up, is_compiled_module
    from accelerate.utils.fsdp_utils import fsdp2_prepare_auto_wrap_policy
    from accelerate.utils.modeling import get_non_persistent_buffers
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )

    is_type_fsdp = isinstance(model, FSDPModule) or (
        is_compiled_module(model) and isinstance(model._orig_mod, FSDPModule)
    )
    if is_type_fsdp:
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin

    original_sd = model.state_dict()

    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )

    # We need the `auto_wrap_policy` original type to create a custom poilicy function for sharding
    # This is because `fully_shard` doesn't support old auto wrap policies, rather we have to imitate the behaviour
    auto_wrap_policy_type = None
    if fsdp2_plugin.auto_wrap_policy is transformer_auto_wrap_policy:
        auto_wrap_policy_type = "transformer"
    elif fsdp2_plugin.auto_wrap_policy is size_based_auto_wrap_policy:
        auto_wrap_policy_type = "size"

    # We set `auto_wrap_policy` to `functools.partial` to avoid creating it again
    # This is because of `apply_activation_checkpointing` which will can reuse this function
    fsdp2_plugin.set_auto_wrap_policy(model)

    if fsdp2_plugin.activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        # Apply activation checkpointing before applying `fully_shard`
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            auto_wrap_policy=fsdp2_plugin.auto_wrap_policy,
        )

    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
    }

    model_has_params4bit = False
    for name, param in model.named_parameters():
        # this is a temporary fix whereby loading models with bnb params cannot be moved from
        # GPU to a meta device due with FSDP2 because torch operations don't return the original class type
        # bypassing the move to meta will still cause the VRAM spike, but at least it still will load
        if param.__class__.__name__ == "Params4bit":
            model_has_params4bit = True
            break

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # Context: `fully_shard` moves the model to GPU if it was on CPU, however it can also be on `meta` and then it stays there even after `fully_shard`
        # For this reason, we need to move the model to `meta` device, as then sharding happens on `meta` device
        # If we kept the model on CPU (`cpu_ram_efficient_loading` has model be on CPU on all ranks, though non-main ranks only have `torch.emtpy`), `fully_shard` would move it to GPU
        # Afterwards, when we call `fsdp2_load_full_state_dict`, us creating the state_dict would result into briefly having two copies of model state_dict on the GPU -> VRAM spike

        # We need to keep the original non-persistent buffers, as those MAY not be in the state_dict, resulting in them staying on meta device
        # Also, these buffers aren't getting sharded by default
        # We get the FQNs of all non-persistent buffers, to re-register them after
        non_persistent_buffer_fqns = get_non_persistent_buffers(
            model, recurse=True, fqns=True
        )
        original_non_persistent_buffers = copy.deepcopy(
            {k: v for k, v in model.named_buffers() if k in non_persistent_buffer_fqns}
        )
        # We move the model to meta device, as then sharding happens on meta device
        model = model.to(torch.device("meta"))
        # We need to re-tie the weights, not exactly sure why, but if we don't do this, reference to `lm_head/embed_tokens` stay hanging -> more VRAM usage
        # We assume `transformers` models have a `tie_weights` method if they support it
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    is_peft_model = "Peft" in model.__class__.__name__
    if is_peft_model:
        from peft.tuners.lora import LoraLayer
    else:
        LoraLayer = None

    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(
        fsdp2_plugin, auto_wrap_policy_type, model
    )
    log_bias_dtype_mismatch = False
    if auto_wrap_policy is not None:
        for module in get_module_children_bottom_up(model)[:-1]:
            ignored_params = []
            if is_peft_model and isinstance(module, LoraLayer):
                for active_adapter in module.active_adapters:
                    # LOG.info(f"module: {module}")
                    # Linear4Bit will keep it's bias term in fp32. If the weight dtype is in bf16 we are not able to
                    # wrap this. Therefore we must ensure the bias has the same dtype as the weight
                    if module.base_layer.bias is not None:
                        if (
                            module.base_layer.weight.dtype
                            != module.base_layer.bias.dtype
                        ):
                            log_bias_dtype_mismatch = True
                            module.base_layer.bias.data = (
                                module.base_layer.bias.data.to(
                                    module.base_layer.weight.dtype
                                )
                            )

                    if module.lora_A:
                        fully_shard(module.lora_A[active_adapter], **fsdp2_kwargs)
                    if module.lora_B:
                        fully_shard(module.lora_B[active_adapter], **fsdp2_kwargs)
                    if module.lora_embedding_A:
                        fully_shard(
                            module.lora_embedding_A[active_adapter], **fsdp2_kwargs
                        )
                    if module.lora_embedding_B:
                        fully_shard(
                            module.lora_embedding_B[active_adapter], **fsdp2_kwargs
                        )
                ignored_params = {
                    p
                    for name, p in module.named_parameters()
                    if "magnitude_vector" in name
                }
            if auto_wrap_policy(module):
                fully_shard(module, ignored_params=ignored_params, **fsdp2_kwargs)
    fully_shard(model, **fsdp2_kwargs)
    if log_bias_dtype_mismatch:
        LOG.warning(
            "Found dtype mismatch between weight and bias in QLoRA linear layers. Since mixed dtypes are not supported in a "
            "single FSDP param group, the bias dtype has been coerced to match the weight dtype."
        )
    if fsdp2_plugin.cpu_ram_efficient_loading:
        offload_to_cpu = isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy)
        fsdp2_load_full_state_dict(
            accelerator, model, original_sd, offload_to_cpu=offload_to_cpu
        )

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # We re-register the buffers, as they may not be in the state_dict
        for fqn, buffer_tensor in original_non_persistent_buffers.items():
            buffer_tensor = buffer_tensor.to(accelerator.device)

            if "." in fqn:
                parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
                parent_module = model.get_submodule(parent_fqn)
            else:
                local_buffer_name = fqn
                parent_module = model

            parent_module.register_buffer(
                local_buffer_name, buffer_tensor, persistent=False
            )

        # We need to tie the weights again, as call to `load_full_state_dict` breaks the tie
        # Needs to be called both here and above
        # removing this call makes the have slightly different loss
        # removing the call above leads to extra memory usage as explained in the comment above
        if hasattr(model, "tie_weights"):
            model.tie_weights()
    return model


def patch_accelerate_fsdp2():
    import accelerate

    # from accelerate.utils import fsdp_utils

    accelerate.accelerator.fsdp2_prepare_model = fsdp2_prepare_model

    accelerate.Accelerator.get_state_dict = get_state_dict
    setattr(
        sys.modules["accelerate"],
        "Accelerator.get_state_dict",
        get_state_dict,
    )
