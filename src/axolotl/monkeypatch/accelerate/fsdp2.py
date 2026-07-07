"""
monkeypatch for accelerate fsdp2 fix when modifying ordereddict during interation, and saving full state dicts
"""

import contextlib
import copy
import functools
import gc
import os
import sys

import torch
import torch.distributed as dist
from torch import nn

from axolotl.utils.bench import log_gpu_memory_usage
from axolotl.utils.fp32_norms import get_fp32_norm_patterns, shard_norms_fp32
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _nvfp4_local_tensor_cls(p):
    """Return the NVFP4Tensor class if ``p`` is a DTensor whose local shard is an NVFP4Tensor."""
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        return None
    lt = getattr(p, "_local_tensor", None)
    return NVFP4Tensor if isinstance(lt, NVFP4Tensor) else None


def _broadcast_nvfp4_param(sharded_meta_param, full_nvfp4, is_main, device, nvfp4_cls):
    """Scatter rank-0's full NVFP4Tensor expert param to each rank's shard.

    ``distribute_tensor`` calls ``c10d.scatter_``, which torchao's NVFP4Tensor doesn't implement.
    Instead scatter the PLAIN component tensors (qdata uint8, scale e4m3, per_tensor_scale fp32 — all
    collective-capable) along the same Shard placement, then rebuild a local NVFP4Tensor shard and
    wrap it back into a DTensor matching the sharded model param."""
    from torch.distributed.tensor import DTensor, distribute_tensor

    mesh = sharded_meta_param.device_mesh
    placements = sharded_meta_param.placements
    local_meta = sharded_meta_param._local_tensor
    e_global = sharded_meta_param.shape[0]
    block_size = local_meta.block_size
    dtype = local_meta.dtype

    def _scatter_component(name, ref_local):
        # Direct per-shard scatter: send each rank ONLY its dim-0 (expert-axis) shard. Avoids the
        # full-global placeholder that non-rank0 otherwise allocates (256 experts to receive 32),
        # the generic distribute_tensor path, and its trailing .clone(). Valid for the common
        # Shard(0) + even-division + full-world mesh (GLM-5.2: 256 experts / 8 ranks); falls back to
        # distribute_tensor for anything else (uneven, replicate, sub-group meshes).
        group = mesh.get_group()
        world = dist.get_world_size(group)
        p0 = placements[0] if len(placements) == 1 else None
        # Direct per-shard scatter on the param's mesh group — the full WORLD (pure data-parallel) OR a
        # dp_shard/cp SUBGROUP (EP composition). Source is the group's rank-0 (== the is_main rank set by
        # the caller). Avoids distribute_tensor, which deadlocks scattering from src_data_rank=0 on a
        # sub-mesh. Falls back to distribute_tensor only for non-Shard(0)/uneven placements.
        if (
            p0 is not None
            and getattr(p0, "dim", None) == 0
            and ref_local.shape[0] * world == e_global
        ):
            src_rank = dist.get_process_group_ranks(group)[0]
            local = torch.empty_like(ref_local, device=device)
            chunks = None
            if is_main:
                full = getattr(full_nvfp4, name)
                chunks = [c.contiguous().to(device) for c in full.chunk(world, dim=0)]
            dist.scatter(local, scatter_list=chunks, src=src_rank, group=group)
            del chunks
            return local
        # Fallback: generic distribute_tensor (uneven / replicate / sub-group).
        gshape = (e_global,) + tuple(ref_local.shape[1:])
        if is_main:
            full = getattr(full_nvfp4, name).to(device)
        else:
            full = torch.empty(gshape, device=device, dtype=ref_local.dtype)
        dt = distribute_tensor(full, mesh, placements, src_data_rank=0)
        return dt._local_tensor.clone()

    local_qdata = _scatter_component("qdata", local_meta.qdata)
    local_scale = _scatter_component("scale", local_meta.scale)

    local_pts = None
    pts_ref = getattr(local_meta, "per_tensor_scale", None)
    if pts_ref is not None:
        if pts_ref.dim() >= 1 and pts_ref.shape[0] == local_meta.qdata.shape[0]:
            # per-expert scale shards along dim 0 like qdata/scale
            local_pts = _scatter_component("per_tensor_scale", pts_ref)
        else:
            # replicated scalar — plain broadcast
            if is_main:
                local_pts = full_nvfp4.per_tensor_scale.to(device)
            else:
                local_pts = torch.empty(
                    pts_ref.shape, device=device, dtype=pts_ref.dtype
                )
            dist.broadcast(local_pts, src=0)

    local_nvfp4 = nvfp4_cls(
        local_qdata, local_scale, block_size, dtype, per_tensor_scale=local_pts
    )
    return DTensor.from_local(local_nvfp4, mesh, placements, run_check=False)


def _ep_expert_from_local(sharded_meta_param, full_local, nvfp4_cls):
    """Build an EP-sharded NVFP4 expert DTensor from THIS rank's already-correct local copy — NO
    collective. shard_expert_weights scattered each rank its ep-group's full ``[E_local]`` NVFP4
    (``full_local``); slice this rank's dp-axis shard (``[E_local // dp]`` at its position in the
    mesh group) and wrap it with ``from_local``. Used instead of a per-subgroup scatter, which
    deadlocks against the interleaved full-mesh non-expert loads (the receiver ranks race ahead of
    the source ranks)."""
    from torch.distributed.tensor import DTensor

    mesh = sharded_meta_param.device_mesh
    placements = sharded_meta_param.placements
    local_meta = sharded_meta_param._local_tensor
    dev = mesh.device_type
    e_dp = local_meta.qdata.shape[0]  # this rank's dp-local expert count
    dp_rank = dist.get_group_rank(mesh.get_group(), dist.get_rank())
    s = slice(dp_rank * e_dp, (dp_rank + 1) * e_dp)
    qd = full_local.qdata[s].to(dev)
    sc = full_local.scale[s].to(dev)
    pts = getattr(full_local, "per_tensor_scale", None)
    local_pts = None
    if pts is not None:
        local_pts = (
            pts[s].to(dev)
            if (pts.dim() >= 1 and pts.shape[0] == full_local.qdata.shape[0])
            else pts.to(dev)
        )
    local_nv = nvfp4_cls(
        qd, sc, local_meta.block_size, local_meta.dtype, per_tensor_scale=local_pts
    )
    return DTensor.from_local(local_nv, mesh, placements, run_check=False)


def fsdp2_load_full_state_dict(
    _accelerator, model: torch.nn.Module, full_sd: dict, offload_to_cpu: bool = False
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
    LOG.info("Broadcasting full state dict to all ranks...")
    import time

    start_time = time.time()

    # EP-sharded expert params hold RANK-SPECIFIC content (each rank owns experts
    # [r*E_local, (r+1)*E_local)). The default rank0-authoritative load below would broadcast
    # rank 0's shard to every rank, replicating experts[0:E_local] everywhere and silently
    # destroying expert parallelism. shard_expert_weights() already scattered the correct shard
    # to every rank BEFORE the model was moved to meta, so each rank's own `full_sd` entry is
    # already correct — load it locally with no cross-rank broadcast. Match via the `.experts.`
    # infix (excludes `shared_experts`, survives the `_checkpoint_wrapped_module` infix) tied to
    # the propagated DDP-ignore list.
    _ep_ignore = getattr(model, "_ddp_params_and_buffers_to_ignore", None) or []
    _ep_tails = tuple(
        sorted(
            {"." + n.split(".experts.", 1)[1] for n in _ep_ignore if ".experts." in n}
        )
    )

    def _is_ep_expert_param(name: str) -> bool:
        return bool(_ep_tails) and ".experts." in name and name.endswith(_ep_tails)

    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    for param_name, sharded_meta_param in meta_sharded_sd.items():
        # Pure-EP: the EP-sharded experts are excluded from the FSDP wrap (ignored_params), so they
        # stay PLAIN per-rank meta params here. shard_expert_weights already scattered each rank's
        # correct [E_local] slice before the meta move, so each rank's own `full_sd` entry is right —
        # load it locally with NO rank-0 broadcast (which would replicate experts[0:E_local]). Only
        # applies when the param is plain (not a DTensor); EP×dp_shard composition keeps them as
        # DTensors loaded via the mesh path below.
        if _is_ep_expert_param(param_name) and not hasattr(
            sharded_meta_param, "device_mesh"
        ):
            own = full_sd[param_name]
            own = own.to(torch.device("cuda"))
            if offload_to_cpu:
                own = own.cpu()
            sharded_sd[param_name] = nn.Parameter(
                own, requires_grad=sharded_meta_param.requires_grad
            )
            full_sd[param_name] = None
            continue

        nvfp4_cls = _nvfp4_local_tensor_cls(sharded_meta_param)

        full_tensor = None
        # Skip the dtype cast for NVFP4 (its components are scattered raw, not cast to bf16).
        if _accelerator.is_main_process and nvfp4_cls is None:
            full_tensor = full_sd[param_name]
            full_tensor = full_tensor.to(sharded_meta_param.dtype)

        if nvfp4_cls is not None:
            # NVFP4Tensor params can't go through distribute_tensor (c10d.scatter_ unimplemented on
            # the subclass); scatter their plain qdata/scale/per_tensor_scale components instead.
            device_mesh = sharded_meta_param.device_mesh
            # Source each NVFP4 expert from its OWN mesh-group's rank-0, not the global rank 0. Under
            # EP×dp_shard / EP×cp composition the experts live on a dp_shard SUBGROUP mesh, and
            # shard_expert_weights scattered each rank its ep-group's real slice — so the subgroup's
            # rank-0 holds the data (global rank 0 is outside ep-groups 1..N's subgroups). For full-world
            # params the subgroup rank-0 IS global rank 0, so this reduces to is_main_process.
            if _is_ep_expert_param(param_name):
                # EP-sharded NVFP4 experts: shard_expert_weights already scattered THIS rank its
                # ep-group's full [E_local] NVFP4 into full_sd. Build the DTensor by slicing the dp-axis
                # shard out of that local copy and wrapping it with from_local — NO collective. The
                # per-subgroup scatter (any flavor) deadlocks because it interleaves with the full-mesh
                # non-expert loads, racing the receiver ranks ahead of the source ranks; from_local
                # sidesteps every collective.
                sharded_param = _ep_expert_from_local(
                    sharded_meta_param, full_sd[param_name], nvfp4_cls
                )
            else:
                # Non-EP NVFP4 on the full data-parallel mesh: only rank 0 has data -> scatter from it.
                full_nvfp4 = (
                    full_sd[param_name] if _accelerator.is_main_process else None
                )
                sharded_param = _broadcast_nvfp4_param(
                    sharded_meta_param,
                    full_nvfp4,
                    _accelerator.is_main_process,
                    device_mesh.device_type,
                    nvfp4_cls,
                )
        elif (
            ".experts." in param_name
            and (".lora_A." in param_name or ".lora_B." in param_name)
            and hasattr(sharded_meta_param, "device_mesh")
            and dist.get_world_size(sharded_meta_param.device_mesh.get_group())
            < dist.get_world_size()
        ):
            # EP×dp_shard/cp composition: the routed-expert LoRA adapter lives on a dp_shard SUBGROUP
            # mesh and holds only THIS ep-group's E_local experts, but full_sd carries the GLOBAL
            # (all-experts) adapter (4096-row lora_A etc.). The generic broadcast below would mismatch
            # sizes (rank-0 sends the global tensor, receivers allocate the E_local size) and replicate
            # rank-0's experts onto ranks owning a different ep-slice. Broadcast rank-0's global adapter
            # (a consistent shape on every rank), then slice THIS rank's ep-experts + dp_shard and
            # from_local. The ep slice is expert-aware (lora_B's experts are not contiguous in its flat
            # r*E dim), so it mirrors shard_expert_lora rather than a plain chunk.
            from torch.distributed.tensor import DTensor

            from axolotl.integrations.expert_parallel.shard import (
                ep_adapter_load_local_shard,
            )

            mesh = sharded_meta_param.device_mesh
            placements = sharded_meta_param.placements
            dp_size = mesh.size()
            ep_size = dist.get_world_size() // dp_size
            ep_dim = 0 if ".lora_A." in param_name else 1
            dev = mesh.device_type
            gshape = list(sharded_meta_param.size())
            gshape[ep_dim] *= ep_size
            if _accelerator.is_main_process:
                g = full_tensor.to(dev)
            else:
                g = torch.empty(gshape, device=dev, dtype=sharded_meta_param.dtype)
            dist.broadcast(g, src=0)
            ep_coord = min(dist.get_process_group_ranks(mesh.get_group())) // dp_size
            dp_rank = dist.get_group_rank(mesh.get_group(), dist.get_rank())
            local = ep_adapter_load_local_shard(
                g,
                ep_dim,
                model._ep_num_experts_global,
                ep_coord,
                ep_size,
                placements,
                dp_size,
                dp_rank,
            )
            sharded_param = DTensor.from_local(local, mesh, placements, run_check=False)
        elif hasattr(sharded_meta_param, "device_mesh"):
            # Generic sharded params (the whole non-EP model, and EP composition's NON-expert weights)
            # live on a FULL mesh that includes global rank 0, so distribute_tensor(src_data_rank=0)
            # broadcasts rank-0's data and shards it correctly — including FSDP's padding for uneven
            # sizes, which a manual chunk + from_local gets wrong (it builds a DTensor with the raw
            # global size and mismatches the model param). EP-composition's rank-0-EXCLUDING subgroup
            # params (experts, and the routed-expert LoRA adapter) are handled by their own branches
            # above, so they never reach here.
            from torch.distributed.tensor import distribute_tensor

            device_mesh = sharded_meta_param.device_mesh
            if _accelerator.is_main_process:
                full_tensor = full_tensor.to(device_mesh.device_type)
            else:
                full_tensor = torch.empty(
                    sharded_meta_param.size(),
                    device=device_mesh.device_type,
                    dtype=sharded_meta_param.dtype,
                )
            sharded_param = distribute_tensor(
                full_tensor,
                device_mesh,
                sharded_meta_param.placements,
                src_data_rank=0,
            )
            # Clone the local shard to allow full_tensor to be freed.
            if (
                sharded_param._local_tensor.untyped_storage().size()
                > sharded_param._local_tensor.nelement()
                * sharded_param._local_tensor.element_size()
            ):
                sharded_param = sharded_param.clone()
        else:
            # Non-sharded parameters
            if _accelerator.is_main_process:
                sharded_param = full_tensor.to(torch.device("cuda"))
            else:
                # broadcast manually
                sharded_param = torch.empty_like(
                    sharded_meta_param,
                    device=torch.device("cuda"),
                    dtype=sharded_meta_param.dtype,
                )
            dist.broadcast(sharded_param, src=0)

        if offload_to_cpu:
            sharded_param = sharded_param.cpu()

        sharded_sd[param_name] = nn.Parameter(sharded_param)

        del full_tensor
        full_sd[param_name] = None

    model.load_state_dict(sharded_sd, assign=True, strict=True)
    end_time = time.time()
    LOG.debug(
        f"Time taken to load full state dict: {(end_time - start_time):.2f} seconds"
    )
    log_gpu_memory_usage(LOG, "Memory usage after broadcasting full state dict", 0)
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
                    model._consolidated_16bit_state_dict()
                    if tp_sharding
                    else model._zero3_consolidated_16bit_state_dict()
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
        from torch.distributed.tensor import DTensor

        state_dict = {}
        sharded_state_dict = model.state_dict()
        is_rank_zero = torch.distributed.get_rank() == 0
        for param_name, param in sharded_state_dict.items():
            if param.is_cpu:
                param = param.to(torch.device("cuda"))

            if isinstance(param, DTensor):
                param = param.full_tensor()

            if is_rank_zero:
                state_dict[param_name] = param.cpu()
            # Drop the GPU-resident gathered tensor before the next iteration
            # allocates the next one; otherwise the caching allocator holds
            # both reservations and we accumulate ~model-size of VRAM.
            del param
            torch.distributed.barrier()

        # Release the sharded view and force the allocator to give back the
        # gather buffers.
        del sharded_state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            StateDictType,
        )

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


def patch_peft_param_wrapper_for_fsdp2():
    """Patch PEFT's _LoraParameterProxy.forward for FSDP2 DTensor compatibility.

    PEFT's ParamWrapper applies LoRA via torch.nn.utils.parametrize, which adds
    delta_weight to the base weight W inside _LoraParameterProxy.forward().
    Under FSDP2, W may be a DTensor (from FSDP unshard) while delta_weight is a
    regular Tensor (or vice versa), causing a RuntimeError on mixed types.

    This patch promotes the non-DTensor operand to match the DTensor's spec
    using DTensor.from_local(), which is free for Replicate placement (just
    metadata wrapping, no communication).
    """
    from peft.tuners.lora.layer import _LoraParameterProxy

    if getattr(_LoraParameterProxy, "_axolotl_fsdp2_patched", False):
        return

    _original_forward = _LoraParameterProxy.forward

    # NOTE: Replaces (not wraps) forward; assumes original is just `W + self.delta_weight`.
    def _patched_forward(self, W):
        from torch.distributed.tensor import DTensor

        delta = self.delta_weight
        w_is_dt = isinstance(W, DTensor)
        d_is_dt = isinstance(delta, DTensor)

        with torch.nn.utils.parametrize.cached():
            if w_is_dt == d_is_dt:
                return W + delta
            if w_is_dt:
                return W + DTensor.from_local(delta, W.device_mesh, W.placements)
            return DTensor.from_local(W, delta.device_mesh, delta.placements) + delta

    _LoraParameterProxy.forward = _patched_forward
    _LoraParameterProxy._axolotl_fsdp2_patched = True
    LOG.info("Patched PEFT _LoraParameterProxy.forward for FSDP2 DTensor compatibility")


def _process_lora_module_for_fsdp(module, fsdp2_kwargs):
    """Helper function to process LoRA modules for FSDP2."""
    from peft.tuners.lora.layer import ParamWrapper
    from torch.distributed.fsdp import fully_shard

    # Skip ParamWrapper — its lora_A/B must not be independently sharded.
    # The parent decoder layer's FSDP wrapper handles unsharding them.
    # TODO: review if we even need to shard them separately in first place.
    if isinstance(module, ParamWrapper):
        return False

    log_bias_dtype_mismatch = False

    # Linear4Bit will keep it's bias term in fp32. If the weight dtype is in bf16 we are not able to
    # wrap this. Therefore we must ensure the bias has the same dtype as the weight
    if hasattr(module.base_layer, "bias") and module.base_layer.bias is not None:
        if module.base_layer.weight.dtype != module.base_layer.bias.dtype:
            log_bias_dtype_mismatch = True
            module.base_layer.bias.data = module.base_layer.bias.data.to(
                module.base_layer.weight.dtype
            )

    for active_adapter in module.active_adapters:
        if module.lora_A:
            fully_shard(module.lora_A[active_adapter], **fsdp2_kwargs)
        if module.lora_B:
            fully_shard(module.lora_B[active_adapter], **fsdp2_kwargs)
        if module.lora_magnitude_vector:
            fully_shard(module.lora_magnitude_vector[active_adapter], **fsdp2_kwargs)

    # lora_embedding_A/B are ParameterDicts containing nn.Parameter (Tensors),
    # not nn.Module. fully_shard() only accepts nn.Module, so we cannot shard
    # individual embedding Parameters. Instead, shard the entire LoraLayer module. fully_shard() can be used hierarchically because it does not
    # override groups already assigned by fully_shard(), so modules
    # where fully_shard() was already called are not affected [see https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html]
    if module.lora_embedding_A or module.lora_embedding_B:
        from torch.distributed.fsdp import FSDPModule

        if not isinstance(module, FSDPModule):
            fully_shard(module, **fsdp2_kwargs)

    return log_bias_dtype_mismatch


def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare

    Returns:
        `torch.nn.Module`: Prepared model
    """
    from accelerate.utils import get_module_children_bottom_up, is_compiled_module
    from accelerate.utils.fsdp_utils import fsdp2_prepare_auto_wrap_policy
    from accelerate.utils.modeling import get_non_persistent_buffers
    from peft import PeftModel
    from peft.tuners.lora import LoraLayer
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
    if fsdp2_plugin.auto_wrap_policy is transformer_auto_wrap_policy:
        pass  # auto_wrap_policy_type = "transformer"
    elif fsdp2_plugin.auto_wrap_policy is size_based_auto_wrap_policy:
        pass  # auto_wrap_policy_type = "size"

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

    mesh = getattr(accelerator.state, "device_mesh", None)

    # Disable memory pinning if requested
    offload_to_cpu = isinstance(fsdp2_plugin.cpu_offload, CPUOffloadPolicy)
    if (
        offload_to_cpu
        and os.environ.get("FSDP_CPU_OFFLOAD_PIN_MEMORY", "").lower() == "false"
    ):
        fsdp2_plugin.cpu_offload.pin_memory = False

    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
        "mesh": (
            mesh[tuple(accelerator.state.parallelism_config.fsdp_dim_names)]
            if mesh is not None
            else None
        ),
    }
    model_has_params4bit = False
    for _, param in model.named_parameters():
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

    is_peft_model = isinstance(model, PeftModel)

    # Patch PEFT's _LoraParameterProxy for DTensor compatibility if any
    # ParamWrapper modules exist (used for target_parameters / 3D expert params).
    if is_peft_model:
        from peft.tuners.lora.layer import ParamWrapper

        if any(isinstance(m, ParamWrapper) for m in model.modules()):
            patch_peft_param_wrapper_for_fsdp2()

    # EP+FSDP: pre-wrap experts on `dp_shard` before the outer auto-wrap so
    # the walker skips them. See `expert_parallel/README.md`.
    # EP composition shards the experts on the non-ep axis so they don't sit replicated per rank. With
    # dp_shard that's the data axis; with pure EP×cp (no dp_shard) the cp ranks of an ep-group hold the
    # SAME experts (cp shards the sequence, not the experts), so FSDP-shard them on cp and let the MoE
    # forward all-gather — otherwise each rank keeps its full ep-group slice (e.g. 64 experts at ep=4)
    # and OOMs on the first forward's all-gather.
    from axolotl.integrations.expert_parallel.plugin import expert_shard_axis

    _ep_shard_axis = expert_shard_axis(
        getattr(mesh, "mesh_dim_names", None) if mesh is not None else None
    )
    if _ep_shard_axis is not None:
        from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
        from axolotl.integrations.expert_parallel.shard import shard_expert_lora

        # Realign target_parameters expert LoRA with the EP-sharded weights before
        # FSDP wraps the experts (PEFT sized it for the global expert count). Stash the
        # exact ep group so the save path gathers across the same axis (re-resolving at
        # save time can pick up a stale/size-1 mesh).
        model._ep_lora_group = mesh["ep"].get_group()
        shard_expert_lora(model, mesh["ep"].size())
        ExpertParallelPlugin.fully_shard_experts(
            model, mesh[_ep_shard_axis], fsdp2_kwargs
        )
    elif getattr(model, "_ddp_params_and_buffers_to_ignore", None):
        # Pure EP (ep_size == world_size): no ep×dp_shard mesh is built, so the experts were
        # manually EP-sharded (shard_expert_weights scattered each rank's real [E_local] slice) but
        # there is NO dp_shard axis to FSDP them onto. Left to the outer fully_shard(mesh=None) they
        # would be re-sharded on the flat world mesh and replicated from rank 0, destroying EP. Exclude
        # the EP-sharded expert base params from the FSDP wrap entirely so each rank keeps its own
        # [E_local] slice as a plain param; fsdp2_load_full_state_dict restores them per-rank.
        ep_ignored = {
            p
            for n, p in model.named_parameters()
            if ".experts." in n
            and ".shared_experts." not in n
            and n.rsplit(".", 1)[-1]
            in ("gate_up_proj", "down_proj", "gate_up_proj_bias", "down_proj_bias")
        }
        if ep_ignored:
            fsdp2_kwargs["ignored_params"] = (
                set(fsdp2_kwargs.get("ignored_params") or set()) | ep_ignored
            )
            LOG.info(
                f"expert_parallel (pure EP): excluded {len(ep_ignored)} EP-sharded expert "
                "param(s) from the FSDP wrap (kept as plain per-rank slices)."
            )

    auto_wrap_policy = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    log_bias_dtype_mismatch = False
    fp32_norm_patterns = get_fp32_norm_patterns(model)
    if fp32_norm_patterns:
        shard_norms_fp32(
            model,
            patterns=fp32_norm_patterns,
            fully_shard_kwargs=fsdp2_kwargs,
        )

    # Pre-quantized / mixed-dtype models (e.g. an NVFP4 checkpoint loaded for LoRA) carry
    # non-float Parameters and keep-fp32 modules that the generic FSDP2 path can't shard
    # uniformly. Engage the quantized capability path ONLY when such params exist; pure-bf16
    # models take the original generic path unchanged. The nonfloat ``nn.Parameter.__new__``
    # patch is process-global and is restored via the context manager's ``finally`` (an
    # exception in ``fully_shard`` can no longer leave it patched).
    from axolotl.monkeypatch.accelerate.fsdp2_quantized import (
        cast_residual_fp32,
        model_has_float_logical_quantized_params,
        model_has_nonfloat_params,
        nonfloat_param_guard,
        shard_fp32_modules,
    )

    # Apply the quantized dtype/cast/sharding policy ONLY for float-logical torchao subclasses
    # (NVFP4Tensor/Float8Tensor/MXTensor) — the pre-quantized checkpoint case this path is for.
    # Plain bnb Params4bit QLoRA is excluded so cast_residual_fp32 does not downcast its fp32 LoRA.
    # The nn.Parameter.__new__ guard is separate: it is needed for ANY plain non-float param (uint8
    # packed, which includes bnb Params4bit) and stays gated on that.
    _quantized = model_has_float_logical_quantized_params(model)
    _needs_nonfloat_guard = model_has_nonfloat_params(model)
    _guard = (
        nonfloat_param_guard(model)
        if _needs_nonfloat_guard
        else contextlib.nullcontext()
    )
    with _guard:
        if _quantized:
            # keep-fp32 modules (registered by model adapters, e.g. DSV4 mHC) get their own
            # fp32 shard group; remaining plain fp32 (PEFT LoRA) is cast to the compute dtype.
            shard_fp32_modules(model, fsdp2_kwargs)
            cast_residual_fp32(model)

        if auto_wrap_policy is not None:
            for module in get_module_children_bottom_up(model)[:-1]:
                if is_peft_model and isinstance(module, LoraLayer):
                    module_log_bias_mismatch = _process_lora_module_for_fsdp(
                        module, fsdp2_kwargs
                    )
                    log_bias_dtype_mismatch |= module_log_bias_mismatch
                if auto_wrap_policy(module) and not isinstance(module, FSDPModule):
                    fully_shard(module, **fsdp2_kwargs)

        fully_shard(model, **fsdp2_kwargs)

    if log_bias_dtype_mismatch:
        LOG.warning(
            "Bias dtype mismatch detected in LoRA base linear layer. Bias parameters have been cast to weight dtype."
        )

    if fsdp2_plugin.cpu_ram_efficient_loading:
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


def patch_tied_keys_for_meta_device():
    """Patch _adjust_tied_keys_with_tied_pointers to skip meta tensors.

    Meta tensors all share data_ptr()==0, causing every parameter to be incorrectly
    grouped as "tied". Skipping them is safe since they have no real storage.
    """
    from collections import defaultdict

    from transformers import PreTrainedModel

    # recent transformers replaced data_ptr tie detection (_adjust_tied_keys_with_tied_pointers) with
    # config-driven get_expanded_tied_weights_keys; absent on the pinned version, so bail (nothing to patch).
    if not hasattr(PreTrainedModel, "_adjust_tied_keys_with_tied_pointers"):
        return

    def _patched_adjust_tied_keys_with_tied_pointers(self, missing_keys):
        param_pointers = defaultdict(list)
        for param_name, param_value in self.state_dict().items():
            if param_value.is_meta:
                continue
            param_pointers[param_value.data_ptr()].append(param_name)

        tied_param_names = [
            names
            for names in param_pointers.values()
            if len(names) > 1
            and not any(name in self.all_tied_weights_keys.keys() for name in names)
            and not all(name in missing_keys for name in names)
        ]

        tied_weights_keys_by_pointers = {
            param_name: group[0]
            for group in tied_param_names
            for param_name in group[1:]
        }
        self.all_tied_weights_keys.update(tied_weights_keys_by_pointers)

    PreTrainedModel._adjust_tied_keys_with_tied_pointers = (
        _patched_adjust_tied_keys_with_tied_pointers
    )


def patch_initialize_missing_keys_for_fsdp():
    """Patch _initialize_missing_keys to skip re-initialization on FSDP non-rank-0.

    When using cpu_ram_efficient_loading, non-rank-0 processes load weights on
    meta device and move them to CPU as empty tensors. Without this patch,
    initialize_weights() re-initializes ALL parameters (via guarded init
    functions), which is slow and uses extra RAM per process.

    The fix marks all params/buffers with _is_hf_initialized=True before calling
    the original method, so guarded init functions (init.normal_, init.zeros_,
    etc.) become no-ops on non-rank-0 processes. The real weights arrive later
    via FSDP broadcast from rank 0.

    Upstream fix: https://github.com/huggingface/transformers/pull/44473
    Remove this patch once transformers includes the fix in a stable release.
    """
    from transformers import PreTrainedModel
    from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0

    if getattr(PreTrainedModel._initialize_missing_keys, "_axolotl_patched", False):
        return

    _original_initialize_missing_keys = PreTrainedModel._initialize_missing_keys

    def _patched_initialize_missing_keys(self, is_quantized: bool) -> None:
        if is_fsdp_enabled() and not is_local_dist_rank_0():
            for key in self.state_dict():
                try:
                    param_or_buffer = self.get_parameter_or_buffer(key)
                    param_or_buffer._is_hf_initialized = True
                except AttributeError:
                    pass  # may happen when handling pre-quantized weights
            self._is_hf_initialized = True

        _original_initialize_missing_keys(self, is_quantized)

    PreTrainedModel._initialize_missing_keys = _patched_initialize_missing_keys
    PreTrainedModel._initialize_missing_keys._axolotl_patched = True


def patch_move_missing_keys_meta_for_fsdp():
    """Stop transformers materializing the FULL model on every non-rank-0 rank during load.

    ``_move_missing_keys_from_meta_to_device`` has a branch ``is_fsdp_enabled() and not
    is_local_dist_rank_0() and not is_quantized`` that moves EVERY meta parameter to real CPU
    storage (``torch.zeros_like(param, device="cpu")``). For an unrecognized-quantizer checkpoint
    (NVFP4-modelopt → ``is_quantized=False``) on a large model, that puts the whole model on each
    non-rank-0 rank → ``world_size``× CPU RAM → OOM. axolotl's ``fsdp2_prepare_model`` immediately
    does ``model.to("meta")`` then broadcasts rank 0's weights, so the materialized params are
    pure waste. Keep params on meta (FSDP fills them); only buffers — computed in ``__init__`` and
    not broadcast — get real CPU storage.

    Caller MUST restrict this to frozen-base (adapter) runs: leaving base params on meta on
    non-rank-0 deadlocks the FSDP2 optimizer-state all-gather at checkpoint save for a FULL
    fine-tune (rank-0 real DTensors vs non-rank-0 meta). LoRA/qLoRA carry no base optimizer state,
    so the gather never touches these params."""
    from transformers import PreTrainedModel
    from transformers.integrations import (
        is_deepspeed_zero3_enabled,
        is_fsdp_enabled,
    )
    from transformers.modeling_utils import (
        _load_parameter_into_model,
        get_device,
        is_local_dist_rank_0,
    )

    if getattr(
        PreTrainedModel._move_missing_keys_from_meta_to_device,
        "_axolotl_patched",
        False,
    ):
        return

    def _patched_move_missing_keys(
        self, missing_keys, device_map, device_mesh, hf_quantizer
    ):
        is_quantized = hf_quantizer is not None
        if is_deepspeed_zero3_enabled() and not is_quantized:
            return

        if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
            # Params: leave on meta — FSDP broadcasts rank 0's real weights into them. (Upstream
            # materialized them all to cpu zeros here, OOMing large models.) Buffers still need real
            # storage, but they are small.
            for key, buffer in self.named_buffers():
                if buffer.is_meta:
                    value = torch.zeros_like(buffer, device="cpu")
                    _load_parameter_into_model(self, key, value)
            return

        for key in missing_keys - self.all_tied_weights_keys.keys():
            param = self.get_parameter_or_buffer(key)
            param_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(param, device=param_device)
            if device_mesh is not None:
                from transformers.modeling_utils import shard_and_distribute_module

                shard_and_distribute_module(
                    self,
                    value,
                    param,
                    key,
                    None,
                    False,
                    device_mesh.get_local_rank(),
                    device_mesh,
                )
            else:
                _load_parameter_into_model(self, key, value)
        for key, buffer in self.named_non_persistent_buffers():
            buffer_device = get_device(device_map, key, valid_torch_device=True)
            value = torch.empty_like(buffer, device=buffer_device)
            _load_parameter_into_model(self, key, value)

    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move_missing_keys
    PreTrainedModel._move_missing_keys_from_meta_to_device._axolotl_patched = True
    LOG.info(
        "Patched transformers _move_missing_keys_from_meta_to_device: non-rank-0 params stay "
        "on meta (FSDP broadcast fills them) instead of full-model CPU materialization"
    )


def patch_accelerate_fsdp2():
    import accelerate

    accelerate.accelerator.fsdp2_prepare_model = fsdp2_prepare_model
    accelerate.Accelerator.get_state_dict = get_state_dict
    setattr(
        sys.modules["accelerate"],
        "Accelerator.get_state_dict",
        get_state_dict,
    )
