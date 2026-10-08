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

The same holds for a ``target_parameters`` expert LoRA (``lora_A`` / ``lora_B`` on the
experts' PEFT ParamWrappers), which shard_expert_lora cuts to each rank's experts: the
adapter-only checkpoint (``pytorch_model_fsdp.bin``, which the Trainer resumes from in
preference to ``adapter_model.safetensors``) and its optimizer state held EP group 0's
expert LoRA alone.

The functions here are drop-in replacements for accelerate's four FSDP checkpoint
functions (same signatures). Saving gathers each EP-sharded tensor across ``ep`` (one
tensor at a time) so the files hold the true full model / adapter and optimizer state, in
the same layout as the final export (``[E_global, ...]`` experts; PEFT's ``[E*r, in]``
``lora_A`` and ``[out, r*E]`` ``lora_B``). Loading hands torch's distributed
``set_*_state_dict`` this rank's block of each such tensor (so dense params, step counts
and param groups load exactly as before) and then copies each rank's own block in, one
tensor at a time. Because the files are full and each rank slices by its own expert
offset, they don't depend on the ep / dp_shard layout they were saved with. Anything other
than an FSDP2 ``FULL_STATE_DICT`` checkpoint is delegated to accelerate unchanged.

The checkpoint contents, tensor layouts, the handling of checkpoints written before
this fix, and the limits are documented under "Training checkpoints and resume"
in src/axolotl/integrations/expert_parallel/README.md
(https://docs.axolotl.ai/docs/custom_integrations.html#training-checkpoints-and-resume).
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
class EPShardedParam:
    """One EP-sharded parameter: this rank holds experts ``[offset:offset+e_local]``.

    ``dim`` is the axis packing the experts and ``rank`` the number of entries per expert
    on it: the routed expert weights are ``[E, ...]`` (dim 0, rank 1); a
    ``target_parameters`` expert LoRA is PEFT's expert-major ``lora_A`` ``[E*r, in]``
    (dim 0, rank r) or rank-major ``lora_B`` ``[out, r*E]`` (dim 1, rank r).

    The layouts are tabulated under "What a checkpoint holds" in the expert-parallel
    README (see the module docstring)."""

    fqn: str
    param: torch.nn.Parameter
    offset: int
    e_local: int
    e_global: int
    dim: int = 0
    rank: int = 1

    @property
    def local_size(self) -> int:
        return self.e_local * self.rank

    @property
    def full_size(self) -> int:
        return self.e_global * self.rank

    def gather(self, whole: torch.Tensor, ep_group) -> torch.Tensor:
        """Assemble every EP rank's block (``whole`` across the non-ep mesh) into the
        full tensor, in the same layout as the final model / adapter export."""
        if self.dim == 0:
            return all_gather_ep_experts(whole, ep_group)
        from .shard import gather_expert_lora_full

        return gather_expert_lora_full(whole, "B", self.e_global, ep_group)

    def block(self, full: torch.Tensor) -> torch.Tensor:
        """This rank's block of the full tensor (the inverse of :meth:`gather`)."""
        if self.dim == 0:
            return full[
                self.offset * self.rank : (self.offset + self.e_local) * self.rank
            ]
        out_dim = full.shape[0]
        return full.reshape(out_dim, self.rank, self.e_global)[
            :, :, self.offset : self.offset + self.e_local
        ].reshape(out_dim, self.local_size)

    def holds_experts(self, value) -> bool:
        """Whether ``value`` (a parameter / optimizer-state tensor) is laid out like this
        parameter along the experts axis, local or full."""
        return (
            isinstance(value, torch.Tensor)
            and value.dim() > self.dim
            and value.shape[self.dim] in (self.local_size, self.full_size)
        )


def _ep_sharded_by_id(model) -> dict:
    from .shard import _detect_experts_modules, _is_param_wrapper, _real_experts_base

    def layout(module):
        e_global = getattr(module, "num_experts_global", None)
        e_local = getattr(module, "num_local_experts", None)
        if e_global is None or e_local is None or e_local >= e_global:
            return None
        offset = getattr(module, "local_expert_offset", None)
        if offset is None:
            raise RuntimeError(
                "expert_parallel: EP-sharded experts module without local_expert_offset"
            )
        return offset, e_local, e_global

    sharded: dict = {}
    for _name, module in _detect_experts_modules(model):
        if (found := layout(module)) is None:
            continue
        for attr in EXPERT_PARAM_NAMES:
            param = getattr(module, attr, None)
            if isinstance(param, torch.nn.Parameter):
                sharded[id(param)] = (*found, 0, 1)
    # expert LoRA (``target_parameters``): shard_expert_lora flags the wrappers whose
    # adapters it cut to this rank's experts
    for _name, wrapper in model.named_modules():
        if not (
            _is_param_wrapper(wrapper) and getattr(wrapper, "_ep_lora_sharded", False)
        ):
            continue
        base = _real_experts_base(wrapper)
        if base is None or (found := layout(base)) is None:
            continue
        ranks = getattr(wrapper, "r", {})
        for attr, dim in (("lora_A", 0), ("lora_B", 1)):
            for adapter, linear in getattr(wrapper, attr, {}).items():
                weight = getattr(linear, "weight", None)
                if isinstance(weight, torch.nn.Parameter) and adapter in ranks:
                    sharded[id(weight)] = (*found, dim, int(ranks[adapter]))
    return sharded


def ep_sharded_params(model) -> list[EPShardedParam]:
    """The EP-sharded expert weights and expert LoRA of ``model``, keyed by their canonical
    state-dict FQN, in ``model.named_parameters()`` order (the same on every rank)."""
    from torch.distributed.checkpoint.state_dict import _get_fqns

    sharded = _ep_sharded_by_id(model)
    params = []
    for name, param in model.named_parameters():
        if id(param) not in sharded:
            continue
        fqns = _get_fqns(model, name)
        if len(fqns) != 1:
            raise RuntimeError(f"expert_parallel: expected one FQN for {name}: {fqns}")
        offset, e_local, e_global, dim, rank = sharded.pop(id(param))
        params.append(
            EPShardedParam(
                next(iter(fqns)), param, offset, e_local, e_global, dim, rank
            )
        )
    return params


def all_gather_ep_experts(local: torch.Tensor, ep_group) -> torch.Tensor:
    """Concatenate each EP rank's ``[E_local, ...]`` block into ``[E_global, ...]``.

    ``local`` must already be whole across the experts' own (non-``ep``) FSDP mesh."""
    local = local.contiguous()
    chunks = [torch.empty_like(local) for _ in range(dist.get_world_size(ep_group))]
    dist.all_gather(chunks, local, group=ep_group)
    return torch.cat(chunks, dim=0)


def _gather_full(sharded: EPShardedParam, tensor: torch.Tensor, ep_group):
    from .shard import _gather_adapter_tensor

    return sharded.gather(_gather_adapter_tensor(tensor).contiguous(), ep_group)


def _optimizer_expert_states(optimizer, sharded: EPShardedParam) -> list[str]:
    """Names of the per-parameter optimizer states laid out along the experts axis
    (``exp_avg``, ``exp_avg_sq``, ...), in a rank-independent order. Scalars such as
    ``step`` are the same on every rank and load as usual."""
    state = optimizer.state.get(sharded.param, {})
    return sorted(key for key, value in state.items() if sharded.holds_experts(value))


def _copy_ep_block(target: torch.Tensor, full: torch.Tensor, sharded: EPShardedParam):
    """Copy this rank's block of the full tensor into ``target`` (a parameter or
    optimizer state, possibly a DTensor sharded on the non-ep mesh)."""
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    block = sharded.block(full)
    if tuple(block.shape) != tuple(target.shape):
        raise RuntimeError(
            f"expert_parallel: checkpoint block {tuple(block.shape)} does not match "
            f"{sharded.fqn} {tuple(target.shape)}"
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
    entries: dict, keys, sharded: EPShardedParam, stash: dict, ident
) -> None:
    """Swap ``entries[key]``'s full tensor for this rank's block (the shape torch's loader
    expects) and stash the full one. A tensor holding only one EP group's experts comes
    from a checkpoint written before this fix and is refused; recovery is documented
    under "Checkpoints from earlier versions" in the expert-parallel README."""
    for key in keys:
        value = entries.get(key)
        if not isinstance(value, torch.Tensor) or not sharded.holds_experts(value):
            continue  # not laid out along the experts axis; loads as usual
        if value.shape[sharded.dim] == sharded.local_size:
            raise RuntimeError(
                f"expert_parallel: checkpoint tensor {sharded.fqn} {key} holds "
                f"{sharded.e_local} of {sharded.e_global} experts. It was written before "
                "EP checkpoints gathered every EP group's experts and holds EP group 0's "
                "experts only, so resuming from it would give every EP group group 0's "
                "experts. Resume from a checkpoint saved with this fix, or start a new "
                "run from the final model export (for LoRA, the checkpoint's own "
                "adapter_model.safetensors holds every expert and loads with "
                "lora_model_dir, without the optimizer state)."
            )
        stash[ident(key)] = value
        entries[key] = sharded.block(value)


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


def _is_fsdp2_full_state_dict(fsdp_plugin) -> bool:
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    return (
        getattr(fsdp_plugin, "fsdp_version", 1) == 2
        and fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT
    )


def _saved_params(model, adapter_only) -> list[EPShardedParam]:
    """The EP-sharded params a model checkpoint holds: all of them, or only the trainable
    ones for accelerate's adapter-only (PEFT) save, which skips the frozen base."""
    from accelerate.utils.modeling import is_peft_model

    params = ep_sharded_params(model)
    if adapter_only and is_peft_model(model):
        params = [p for p in params if p.param.requires_grad]
    return params


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

    experts = _saved_params(model, adapter_only)
    if not experts or not _is_fsdp2_full_state_dict(fsdp_plugin):
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
    state_dict = fsdp_utils._get_model_state_dict(model, adapter_only, sd_options)
    for expert in experts:
        full = _gather_full(expert, expert.param, ep_group)  # collective: every rank
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

    experts = _saved_params(model, adapter_only)
    if not experts or not _is_fsdp2_full_state_dict(fsdp_plugin):
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
    load_result = fsdp_utils._set_model_state_dict(
        model, state_dict, adapter_only, sd_options
    )
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

    experts = [p for p in ep_sharded_params(model) if p.param.requires_grad]
    if not experts or not _is_fsdp2_full_state_dict(fsdp_plugin):
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
            full = _gather_full(expert, optimizer.state[expert.param][key], ep_group)
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

    experts = [p for p in ep_sharded_params(model) if p.param.requires_grad]
    if not experts or not _is_fsdp2_full_state_dict(fsdp_plugin):
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
    the duration of the block (the Trainer calls them by their module-global names).

    Behaviour and limits: "Training checkpoints and resume" in
    src/axolotl/integrations/expert_parallel/README.md
    (https://docs.axolotl.ai/docs/custom_integrations.html#training-checkpoints-and-resume)."""
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
