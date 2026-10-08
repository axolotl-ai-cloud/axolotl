# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FSDP2 ``FULL_STATE_DICT`` training checkpoints that keep every EP group's experts.

Under expert parallelism each EP rank holds a different ``[offset:offset+E_local]`` block
of experts under the same parameter name, FSDP-sharded on a mesh without the ``ep`` axis.
accelerate's ``save_fsdp_model`` / ``save_fsdp_optimizer`` gather full state dicts over that
mesh only and rank 0 writes them, so ``pytorch_model_fsdp.bin`` and ``optimizer.bin`` would
hold EP group 0's experts alone; on load rank 0 broadcasts its file and every EP group
would take group 0's expert weights and optimizer moments.

The functions here are drop-in replacements for accelerate's four FSDP checkpoint
functions (same signatures). Saving gathers each expert tensor across ``ep`` (one tensor at
a time) so the files hold the true full ``[E_global, ...]`` model and optimizer state, the
same shapes as the final ``model.safetensors`` export. Loading hands torch's distributed
``set_*_state_dict`` an ``E_local`` slice of each expert tensor (so dense params, step
counts and param groups load exactly as before) and then copies each rank's own expert
block in, one tensor at a time. Because the files are full and each rank slices by its own
expert offset, they don't depend on the ep / dp_shard layout they were saved with.
Anything other than a full-parameter FSDP2 ``FULL_STATE_DICT`` checkpoint is delegated to
accelerate unchanged.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

EXPERT_PARAM_NAMES = (
    "gate_up_proj",
    "down_proj",
    "gate_up_proj_bias",
    "down_proj_bias",
)


@dataclass(frozen=True)
class EPExpertParam:
    """One EP-sharded expert parameter: this rank holds experts ``[offset:offset+e_local]``."""

    fqn: str
    param: torch.nn.Parameter
    offset: int
    e_local: int
    e_global: int


def ep_sharded_expert_params(model) -> list[EPExpertParam]:
    """The EP-sharded expert parameters of ``model``, keyed by their canonical state-dict FQN.

    Ordered by ``model.named_parameters()``, so every rank walks them in the same order."""
    from torch.distributed.checkpoint.state_dict import _get_fqns

    from .shard import _detect_experts_modules

    sharded = {}
    for _name, module in _detect_experts_modules(model):
        e_global = getattr(module, "num_experts_global", None)
        e_local = getattr(module, "num_local_experts", None)
        if e_global is None or e_local is None or e_local >= e_global:
            continue
        offset = getattr(module, "local_expert_offset", None)
        if offset is None:
            raise RuntimeError(
                "expert_parallel: EP-sharded experts module without local_expert_offset"
            )
        for attr in EXPERT_PARAM_NAMES:
            param = getattr(module, attr, None)
            if isinstance(param, torch.nn.Parameter):
                sharded[id(param)] = (param, offset, e_local, e_global)

    params = []
    for name, param in model.named_parameters():
        if id(param) not in sharded:
            continue
        fqns = _get_fqns(model, name)
        if len(fqns) != 1:
            raise RuntimeError(f"expert_parallel: expected one FQN for {name}: {fqns}")
        _param, offset, e_local, e_global = sharded.pop(id(param))
        params.append(EPExpertParam(next(iter(fqns)), param, offset, e_local, e_global))
    return params


def all_gather_ep_experts(local: torch.Tensor, ep_group) -> torch.Tensor:
    """Concatenate each EP rank's ``[E_local, ...]`` block into ``[E_global, ...]``.

    ``local`` must already be whole across the experts' own (non-``ep``) FSDP mesh."""
    local = local.contiguous()
    chunks = [torch.empty_like(local) for _ in range(dist.get_world_size(ep_group))]
    dist.all_gather(chunks, local, group=ep_group)
    return torch.cat(chunks, dim=0)


def _gather_full(tensor: torch.Tensor, ep_group) -> torch.Tensor:
    from .shard import _gather_adapter_tensor

    return all_gather_ep_experts(_gather_adapter_tensor(tensor), ep_group)


def _optimizer_expert_states(optimizer, expert: EPExpertParam) -> list[str]:
    """Names of the per-parameter optimizer states laid out along the experts dim
    (``exp_avg``, ``exp_avg_sq``, ...), in a rank-independent order. Scalars such as
    ``step`` are the same on every rank and load as usual."""
    state = optimizer.state.get(expert.param, {})
    return sorted(
        key
        for key, value in state.items()
        if torch.is_tensor(value)
        and value.dim() > 0
        and value.shape[0] == expert.e_local
    )


def _copy_ep_block(target: torch.Tensor, full: torch.Tensor, expert: EPExpertParam):
    """Copy this rank's experts of the full ``[E_global, ...]`` tensor into ``target``
    (a parameter or optimizer state, possibly a DTensor sharded on the non-ep mesh)."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    block = full[expert.offset : expert.offset + expert.e_local]
    if tuple(block.shape) != tuple(target.shape):
        raise RuntimeError(
            f"expert_parallel: checkpoint block {tuple(block.shape)} does not match "
            f"{expert.fqn} {tuple(target.shape)}"
        )
    with torch.no_grad():
        if isinstance(target, DTensor):
            shape, offset = compute_local_shape_and_global_offset(
                block.shape, target.device_mesh, target.placements
            )
            slices = tuple(slice(o, o + s) for s, o in zip(shape, offset, strict=True))
            local = target.to_local()
            local.copy_(block[slices].to(local.device))
        else:
            target.copy_(block.to(target.device))


def _take_full_experts(
    entries: dict, keys, expert: EPExpertParam, stash: dict, ident
) -> None:
    """Swap ``entries[key]``'s full ``[E_global, ...]`` tensor for this rank's
    ``[E_local, ...]`` block (the shape torch's loader expects) and stash the full one."""
    for key in keys:
        value = entries.get(key)
        if not isinstance(value, torch.Tensor) or value.dim() == 0:
            continue
        if value.shape[0] not in (expert.e_global, expert.e_local):
            continue  # not laid out along the experts dim; loads as usual
        if value.shape[0] == expert.e_local:
            raise RuntimeError(
                f"expert_parallel: checkpoint tensor {expert.fqn} {key} holds "
                f"{expert.e_local} of {expert.e_global} experts. It was written before EP "
                "checkpoints gathered every EP group's experts and holds EP group 0's "
                "experts only, so resuming from it would give every EP group group 0's "
                "experts. Resume from a checkpoint saved with this fix, or start a new "
                "run from the final model export."
            )
        stash[ident(key)] = value
        entries[key] = value[expert.offset : expert.offset + expert.e_local]


def _raise_on_every_rank(split) -> None:
    """Run ``split`` and raise its error on every rank (only the ranks that read the file
    can detect a bad checkpoint; the others would otherwise wait in the next collective)."""
    error = None
    try:
        split()
    except RuntimeError as exc:
        error = str(exc)
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    first = next((e for e in errors if e is not None), None)
    if first is not None:
        raise RuntimeError(first)


def _comm_device() -> torch.device:
    if dist.get_backend() == "gloo" or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _scatter_from_rank0(stash: dict, targets: dict) -> None:
    """Broadcast each stashed full expert tensor from global rank 0, one at a time, and
    copy every rank's own block into its target. ``targets`` maps a stash key to
    ``(expert, target_tensor)`` and must hold the same keys on every rank."""
    meta: list = [
        [(ident, tuple(t.shape), t.dtype) for ident, t in stash.items()]
        if dist.get_rank() == 0
        else None
    ]
    dist.broadcast_object_list(meta, src=0)
    device = _comm_device()
    entries: list = meta[0]
    for ident, shape, dtype in entries:
        if ident not in targets:
            raise RuntimeError(f"expert_parallel: no target for checkpoint {ident}")
        expert, target = targets[ident]
        if dist.get_rank() == 0:
            full = stash[ident].to(device).contiguous()
        else:
            full = torch.empty(shape, dtype=dtype, device=device)
        dist.broadcast(full, src=0)
        _copy_ep_block(target, full, expert)
        del full


def _is_full_param_fsdp2_full_state_dict(fsdp_plugin, model, adapter_only) -> bool:
    from accelerate.utils.modeling import is_peft_model
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    return (
        getattr(fsdp_plugin, "fsdp_version", 1) == 2
        and fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT
        and not (adapter_only and is_peft_model(model))
    )


def _set_full_state_dict_flags(fsdp_plugin, accelerator) -> None:
    # as accelerate does: gather to rank 0 on CPU (and broadcast from it on load)
    is_multi_process = accelerator.num_processes > 1
    fsdp_plugin.state_dict_config.offload_to_cpu = is_multi_process
    fsdp_plugin.state_dict_config.rank0_only = is_multi_process


def _file(directory, name: str, index: int) -> str:
    return os.path.join(
        directory, f"{name}.bin" if index == 0 else f"{name}_{index}.bin"
    )


def save_fsdp_model(
    fsdp_plugin,
    accelerator,
    model,
    output_dir,
    model_index=0,
    adapter_only=False,
    use_dcp=True,
    *,
    ep_group,
):
    """accelerate's ``save_fsdp_model`` with every EP group's experts in the file."""
    from accelerate.utils import fsdp_utils
    from accelerate.utils.constants import FSDP_MODEL_NAME
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    experts = ep_sharded_expert_params(model)
    if not experts or not _is_full_param_fsdp2_full_state_dict(
        fsdp_plugin, model, adapter_only
    ):
        return fsdp_utils.save_fsdp_model(
            fsdp_plugin,
            accelerator,
            model,
            output_dir,
            model_index,
            adapter_only,
            use_dcp,
        )

    os.makedirs(output_dir, exist_ok=True)
    _set_full_state_dict_flags(fsdp_plugin, accelerator)
    sd_options = fsdp_utils._prepare_sd_options(fsdp_plugin)
    state_dict = get_model_state_dict(model, options=sd_options)
    for expert in experts:
        full = _gather_full(expert.param, ep_group)  # collective: every rank
        if expert.fqn in state_dict:
            state_dict[expert.fqn] = full.to(state_dict[expert.fqn].device)
        del full
    if accelerator.process_index == 0:
        output_model_file = _file(output_dir, FSDP_MODEL_NAME, model_index)
        LOG.info(f"Saving model with all EP groups' experts to {output_model_file}")
        torch.save(state_dict, output_model_file)


def load_fsdp_model(
    fsdp_plugin,
    accelerator,
    model,
    input_dir,
    model_index=0,
    adapter_only=False,
    use_dcp=True,
):
    """accelerate's ``load_fsdp_model`` restoring each EP rank's own experts."""
    from accelerate.utils import fsdp_utils
    from accelerate.utils.constants import FSDP_MODEL_NAME
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    experts = ep_sharded_expert_params(model)
    if not experts or not _is_full_param_fsdp2_full_state_dict(
        fsdp_plugin, model, adapter_only
    ):
        return fsdp_utils.load_fsdp_model(
            fsdp_plugin,
            accelerator,
            model,
            input_dir,
            model_index,
            adapter_only,
            use_dcp,
        )

    accelerator.wait_for_everyone()
    _set_full_state_dict_flags(fsdp_plugin, accelerator)
    sd_options = fsdp_utils._prepare_sd_options(fsdp_plugin)
    from_rank0 = sd_options.broadcast_from_rank0
    state_dict = {}
    if not from_rank0 or accelerator.is_main_process:
        input_model_file = _file(input_dir, FSDP_MODEL_NAME, model_index)
        LOG.info(f"Loading model from {input_model_file}")
        state_dict = torch.load(input_model_file, weights_only=True)

    stash: dict = {}

    def split():
        for expert in experts:
            _take_full_experts(
                state_dict, [expert.fqn], expert, stash, lambda _k, e=expert: e.fqn
            )

    _raise_on_every_rank(split)
    load_result = set_model_state_dict(model, state_dict, options=sd_options)
    if from_rank0:
        # rank 0's block went to every rank above; overwrite with each rank's own
        _scatter_from_rank0(stash, {e.fqn: (e, e.param) for e in experts})
    accelerator.wait_for_everyone()
    return load_result


def save_fsdp_optimizer(
    fsdp_plugin,
    accelerator,
    optimizer,
    model,
    output_dir,
    optimizer_index=0,
    use_dcp=True,
    *,
    ep_group,
):
    """accelerate's ``save_fsdp_optimizer`` with every EP group's expert states in the file."""
    from accelerate.utils import fsdp_utils
    from accelerate.utils.constants import OPTIMIZER_NAME
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    experts = ep_sharded_expert_params(model)
    if not experts or not _is_full_param_fsdp2_full_state_dict(
        fsdp_plugin, model, False
    ):
        return fsdp_utils.save_fsdp_optimizer(
            fsdp_plugin,
            accelerator,
            optimizer,
            model,
            output_dir,
            optimizer_index,
            use_dcp,
        )

    os.makedirs(output_dir, exist_ok=True)
    sd_options = fsdp_utils._prepare_sd_options(fsdp_plugin)
    optim_state = get_optimizer_state_dict(model, optimizer, options=sd_options)
    saved_states = optim_state.get("state", {}) if optim_state else {}
    for expert in experts:
        entry = saved_states.get(expert.fqn)
        for key in _optimizer_expert_states(optimizer, expert):
            full = _gather_full(optimizer.state[expert.param][key], ep_group)
            if entry is not None and key in entry:
                entry[key] = full.to(entry[key].device)
            del full
    if accelerator.process_index == 0:
        output_optimizer_file = _file(output_dir, OPTIMIZER_NAME, optimizer_index)
        LOG.info(
            f"Saving optimizer state with all EP groups' experts to {output_optimizer_file}"
        )
        torch.save(optim_state, output_optimizer_file)


def load_fsdp_optimizer(
    fsdp_plugin,
    accelerator,
    optimizer,
    model,
    input_dir,
    optimizer_index=0,
    adapter_only=False,
    use_dcp=True,
):
    """accelerate's ``load_fsdp_optimizer`` restoring each EP rank's own expert states."""
    from accelerate.utils import fsdp_utils
    from accelerate.utils.constants import OPTIMIZER_NAME
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    experts = ep_sharded_expert_params(model)
    if not experts or not _is_full_param_fsdp2_full_state_dict(
        fsdp_plugin, model, adapter_only
    ):
        return fsdp_utils.load_fsdp_optimizer(
            fsdp_plugin,
            accelerator,
            optimizer,
            model,
            input_dir,
            optimizer_index,
            adapter_only,
            use_dcp,
        )

    accelerator.wait_for_everyone()
    sd_options = fsdp_utils._prepare_sd_options(fsdp_plugin)
    optim_state = None
    if (
        accelerator.process_index == 0
        or not fsdp_plugin.optim_state_dict_config.rank0_only
    ):
        input_optimizer_file = _file(input_dir, OPTIMIZER_NAME, optimizer_index)
        LOG.info(f"Loading optimizer state from {input_optimizer_file}")
        optim_state = torch.load(input_optimizer_file, weights_only=True)

    stash: dict = {}

    def split():
        saved_states = (optim_state or {}).get("state", {})
        for expert in experts:
            entry = saved_states.get(expert.fqn)
            if entry is not None:
                _take_full_experts(
                    entry, list(entry), expert, stash, lambda k, e=expert: (e.fqn, k)
                )

    _raise_on_every_rank(split)
    set_optimizer_state_dict(model, optimizer, optim_state, options=sd_options)
    if sd_options.broadcast_from_rank0:
        targets = {
            (e.fqn, key): (e, optimizer.state[e.param][key])
            for e in experts
            for key in _optimizer_expert_states(optimizer, e)
        }
        _scatter_from_rank0(stash, targets)
    accelerator.wait_for_everyone()


@contextmanager
def ep_fsdp_checkpoint_functions(ep_group):
    """Point ``transformers.trainer``'s FSDP checkpoint functions at the EP-aware ones for
    the duration of the block (the Trainer calls them by their module-global names)."""
    import functools

    import transformers.trainer as hf_trainer

    replacements = {
        "save_fsdp_model": functools.partial(save_fsdp_model, ep_group=ep_group),
        "save_fsdp_optimizer": functools.partial(
            save_fsdp_optimizer, ep_group=ep_group
        ),
        "load_fsdp_model": load_fsdp_model,
        "load_fsdp_optimizer": load_fsdp_optimizer,
    }
    missing = object()
    previous = {name: getattr(hf_trainer, name, missing) for name in replacements}
    for name, function in replacements.items():
        setattr(hf_trainer, name, function)
    try:
        yield
    finally:
        for name, function in previous.items():
            if function is missing:
                delattr(hf_trainer, name)
            else:
                setattr(hf_trainer, name, function)
