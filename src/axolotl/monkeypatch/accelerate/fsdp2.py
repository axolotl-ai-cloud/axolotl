"""
monkeypatch for accelerate fsdp2 fix when modifying ordereddict during interation, and saving full state dicts
"""

import sys

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. This is done by broadcasting the
    parameters from rank 0 to all other ranks. This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    import torch.distributed as dist
    from torch.distributed.tensor import distribute_tensor

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

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

    param_names = sorted(meta_sharded_sd.keys())

    for param_name in param_names:
        mesh = meta_sharded_sd[param_name].device_mesh
        if accelerator.is_main_process:
            full_param = full_sd[param_name].detach().cuda()
            dist.broadcast(full_param, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_param, mesh, sharded_sd[param_name].placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_param,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor
        else:
            full_tensor = torch.empty(
                sharded_sd[param_name].size(),
                device="cuda",
                dtype=sharded_sd[param_name].dtype,
            )
            dist.broadcast(full_tensor, src=0, group=mesh.get_group())
            sharded_tensor = distribute_tensor(
                full_tensor, mesh, sharded_sd[param_name].placements
            )
            to_contiguous, casting_dtype = _infer_parameter_dtype(
                model,
                param_name,
                full_tensor,
            )
            sharded_tensor = _cast_and_contiguous(
                sharded_tensor, to_contiguous, casting_dtype
            )
            sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    model.load_state_dict(sharded_sd, assign=True)
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


def patch_accelerate_fsdp2():
    import accelerate
    from accelerate.utils import fsdp_utils

    fsdp_utils.fsdp2_load_full_state_dict = fsdp2_load_full_state_dict
    setattr(
        sys.modules["accelerate.utils.fsdp_utils"],
        "fsdp2_load_full_state_dict",
        fsdp2_load_full_state_dict,
    )

    accelerate.Accelerator.get_state_dict = get_state_dict
    setattr(
        sys.modules["accelerate"],
        "Accelerator.get_state_dict",
        get_state_dict,
    )
