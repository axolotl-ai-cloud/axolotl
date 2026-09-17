# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Generic expert-weight sharding for `@use_experts_implementation` modules.

After this runs (in `post_model_build`, before FSDP wraps), each rank's Experts
modules hold only their local slice of the experts dim. The registered
`deep_ep_*` forward function then handles dispatch -> local compute -> combine.
"""

from __future__ import annotations

import gc

import torch
import torch.distributed as dist

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _replace_with_slice(module, attr_name: str, start: int, end: int) -> None:
    """Replace `module.{attr_name}` with a fresh-allocation slice [start:end].

    `tensor[start:end]` on a contiguous storage returns a VIEW that shares the
    same underlying allocation. `.contiguous()` is a no-op on an already-
    contiguous view, so wrapping the view in a new Parameter does NOT release
    the original full-size storage — refcount on the storage stays >= 1 from
    the new view.

    `.clone()` forces a fresh allocation. After we drop the old Parameter,
    the original storage's refcount drops to zero and PyTorch's caching
    allocator reclaims the memory on the next `empty_cache()`.
    """
    old = getattr(module, attr_name)
    new_data = old.data[start:end].detach().clone().contiguous()
    requires_grad = old.requires_grad
    # Drop the old Parameter from the module's _parameters registry first so
    # nothing is holding a reference when we assign the new one.
    if attr_name in module._parameters:
        del module._parameters[attr_name]
    setattr(
        module,
        attr_name,
        torch.nn.Parameter(new_data, requires_grad=requires_grad),
    )
    # `old` goes out of scope on return.


def _scatter_expert_from_rank0(module, attr_name, e_local, dp_size):
    """Populate ``module.{attr_name}`` with THIS rank's real ``[e_local]`` expert slice by scattering
    GLOBAL rank-0's full expert tensor over the WORLD group. Under cpu_ram_efficient_loading only global
    rank 0 materializes real weights, so it is the single source (sourcing from an ep-subgroup's rank-0
    would crash on the meta ranks). Each rank ``r`` receives its ep-group's slice
    ``full[(r//dp_size)*e_local : ((r//dp_size)+1)*e_local]`` — the ``dp_size`` ranks within an ep-group
    get the SAME ep slice (the dp axis FSDP-shards it across them later). ``dp_size == 1`` is pure EP
    (each rank its own [e_local]); ``dp_size > 1`` is EP×dp_shard / EP×cp composition. Handles torchao
    NVFP4Tensor (qdata/scale/per_tensor_scale) and plain tensors; runs on GPU (NCCL), result moved to
    the param's original device."""
    import torch.distributed as dist

    old = getattr(module, attr_name)
    nv = old.data
    dev = torch.device("cuda", torch.cuda.current_device())
    world = dist.get_world_size()
    is_src = dist.get_rank() == 0
    dst_device = (
        old.data.device if old.data.device.type != "meta" else torch.device("cpu")
    )
    requires_grad = old.requires_grad

    def _scatter_dim0(full_comp, like):
        out = torch.empty((e_local, *like.shape[1:]), dtype=like.dtype, device=dev)
        chunks = None
        if is_src:
            chunks = [
                full_comp[(r // dp_size) * e_local : (r // dp_size + 1) * e_local]
                .contiguous()
                .to(dev)
                for r in range(world)
            ]
        dist.scatter(out, scatter_list=chunks, src=0)
        return out

    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    except ImportError:
        NVFP4Tensor = None

    if NVFP4Tensor is not None and isinstance(nv, NVFP4Tensor):
        full_q = nv.qdata if is_src else None
        local_q = _scatter_dim0(full_q, nv.qdata)
        # scale (e4m3) isn't a collective dtype on all backends -> scatter the uint8 view.
        full_s = nv.scale.view(torch.uint8) if is_src else None
        local_s = _scatter_dim0(full_s, nv.scale.view(torch.uint8)).view(nv.scale.dtype)
        pts = getattr(nv, "per_tensor_scale", None)
        local_pts = None
        if pts is not None:
            if pts.dim() >= 1 and pts.shape[0] == nv.qdata.shape[0]:
                local_pts = _scatter_dim0(pts if is_src else None, pts)
            else:  # replicated scalar
                local_pts = pts.to(dev).clone()
                dist.broadcast(local_pts, src=0)
        local_nv = NVFP4Tensor(
            local_q.to(dst_device),
            local_s.to(dst_device),
            nv.block_size,
            nv.dtype,
            per_tensor_scale=(
                local_pts.to(dst_device) if local_pts is not None else None
            ),
        )
        new_param = torch.nn.Parameter(local_nv, requires_grad=requires_grad)
    else:
        local = _scatter_dim0(nv if is_src else None, nv)
        new_param = torch.nn.Parameter(
            local.to(dst_device), requires_grad=requires_grad
        )

    if attr_name in module._parameters:
        del module._parameters[attr_name]
    setattr(module, attr_name, new_param)


def _detect_experts_modules(model):
    """Yield (name, module) pairs for every module that looks like an Experts class.

    Detection: 3D `gate_up_proj` and `down_proj` parameters with experts on dim 0.
    This is the canonical layout enforced by `@use_experts_implementation`.
    Mixtral's `ModuleList[MixtralBlockSparseTop2MLP]` does NOT match — out of scope
    for v1.
    """
    for name, module in model.named_modules():
        # A PEFT ParamWrapper delegates `gate_up_proj` to its `base_layer`, so it
        # would match too and double-count the experts (double grad-scale, redundant
        # fully_shard). Yield only the real experts module (the wrapped base_layer).
        if _is_param_wrapper(module):
            continue
        gp = getattr(module, "gate_up_proj", None)
        dp = getattr(module, "down_proj", None)
        if gp is None or dp is None:
            continue
        if not (
            isinstance(gp, torch.nn.Parameter) and isinstance(dp, torch.nn.Parameter)
        ):
            continue
        if gp.dim() != 3 or dp.dim() != 3:
            continue
        yield name, module


def shard_expert_weights(model, ep_group) -> int:
    """Slice expert weights along dim 0 per the EP rank.

    Args:
        model: A built (but not yet FSDP-wrapped) HuggingFace model.
        ep_group: `torch.distributed.ProcessGroup` for EP, or `None` for
            single-rank (no-op).

    Returns:
        Number of Experts modules sharded (0 if EP disabled or none found).

    Raises:
        ValueError: if any Experts module's `num_experts` is not divisible by
            the EP world size.

    DDP composition: the sharded params hold DIFFERENT content per rank, so we
    add their fully-qualified names to `model._ddp_params_and_buffers_to_ignore`
    to prevent the startup broadcast from copying rank 0's slice everywhere.
    FSDP composition is handled in `ExpertParallelPlugin.fully_shard_experts`.
    """
    if ep_group is None:
        return 0

    ep_size = dist.get_world_size(ep_group)
    if ep_size <= 1:
        return 0

    ep_rank = dist.get_rank(ep_group)
    sharded = 0
    ignore_names: list[str] = []

    for name, module in _detect_experts_modules(model):
        gp = module.gate_up_proj
        E = gp.shape[0]
        if E % ep_size != 0:
            raise ValueError(
                f"Expert module {name!r}: num_experts={E} not divisible by "
                f"ep_size={ep_size}. Adjust the model config or ep_size."
            )
        E_local = E // ep_size
        start = ep_rank * E_local

        # Scatter global rank-0's REAL experts to every rank's ep-group slice. Under
        # cpu_ram_efficient_loading only global rank 0 has data; plain-slicing (the old path) kept
        # only rank 0's own ep-group's slice and zeroed the rest, so ep-groups 1..N had dead experts.
        # dp_size>1 (EP×dp_shard / EP×cp) gives the dp ranks within an ep-group the same ep slice; the
        # FSDP dp-axis shards it across them afterwards. See _scatter_expert_from_rank0.
        dp_size = dist.get_world_size() // ep_size
        with torch.no_grad():
            _scatter_expert_from_rank0(module, "gate_up_proj", E_local, dp_size)
            _scatter_expert_from_rank0(module, "down_proj", E_local, dp_size)
            for bias_name in ("gate_up_proj_bias", "down_proj_bias"):
                bias = getattr(module, bias_name, None)
                if (
                    isinstance(bias, torch.nn.Parameter)
                    and bias.dim() >= 1
                    and bias.shape[0] == E
                ):
                    _scatter_expert_from_rank0(module, bias_name, E_local, dp_size)

        # Stash metadata the registered fn needs.
        module.local_expert_offset = start
        module.num_local_experts = E_local
        module.num_experts_global = E
        # Single global expert count for the cpu_ram_efficient load path (all routed-expert modules
        # share it); used to reshape the global expert-LoRA adapter when slicing per-rank shards.
        model._ep_num_experts_global = E
        # Override the kernel's view of num_experts for grouped_mm/scattermoe bucketing.
        module.num_experts = E_local

        # Mark sharded params as DDP-ignored — they hold rank-specific content
        # and must NOT be broadcast from rank 0 at DDP construction.
        ignore_names.append(f"{name}.gate_up_proj")
        ignore_names.append(f"{name}.down_proj")
        for bias_name in ("gate_up_proj_bias", "down_proj_bias"):
            if isinstance(getattr(module, bias_name, None), torch.nn.Parameter):
                ignore_names.append(f"{name}.{bias_name}")

        sharded += 1

    if sharded:
        # Drop refs to any leftover full-size storage and return cached memory
        # to the allocator. Without this, the original `from_pretrained`
        # allocations stay reserved and `memory_allocated()` doesn't drop.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Append (don't overwrite) — other systems may have set this too.
        existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []))
        model._ddp_params_and_buffers_to_ignore = existing + ignore_names
        LOG.info(
            f"Sharded {sharded} Experts module(s) along the experts dim "
            f"(ep_rank={ep_rank}, ep_size={ep_size}). "
            f"Marked {len(ignore_names)} param(s) as DDP-ignored."
        )
    return sharded


def _is_param_wrapper(module) -> bool:
    """True for a PEFT ParamWrapper, including after ``fully_shard`` renames its class
    (e.g. ``FSDPParamWrapper``) — FSDP2 sets ``__class__`` to a subclass, so ``isinstance``
    still holds while a ``type(...).__name__ == 'ParamWrapper'`` check would miss it."""
    try:
        from peft.tuners.lora.layer import ParamWrapper

        if isinstance(module, ParamWrapper):
            return True
    except (ImportError, AttributeError):
        pass
    return "ParamWrapper" in type(module).__name__


def _real_experts_base(wrapper):
    """Walk a (possibly chained) PEFT ParamWrapper down to the real experts module."""
    base = getattr(wrapper, "base_layer", None)
    while base is not None and _is_param_wrapper(base):
        base = getattr(base, "base_layer", None)
    return base


def _slice_expert_lora_param(linear, dim: int, e_global: int, start: int, end: int):
    """Replace ``linear.weight`` with its local-experts slice.

    ``dim`` is the axis carrying the ``E*r`` packing:
      * lora_A: ``[E*r, in]`` expert-major  -> slice rows ``[start*r:end*r]`` (dim 0).
      * lora_B: ``[out, r*E]`` rank-major    -> reshape ``[out, r, E]``, take experts, flatten.
    Returns the rank ``r`` (so the caller can keep ``in_features``/``out_features`` consistent).
    """
    w = linear.weight
    if dim == 0:  # lora_A: expert-major rows
        r = w.shape[0] // e_global
        new = w.data[start * r : end * r, :].detach().clone()
        linear.out_features = new.shape[0]
    else:  # lora_B: rank-major columns [out, r, E]
        out_dim = w.shape[0]
        r = w.shape[1] // e_global
        new = (
            w.data.reshape(out_dim, r, e_global)[:, :, start:end]
            .reshape(out_dim, r * (end - start))
            .detach()
            .clone()
        )
        linear.in_features = new.shape[1]
    linear.weight = torch.nn.Parameter(new, requires_grad=w.requires_grad)
    return r


def ep_adapter_load_local_shard(
    global_adapter, ep_dim, e_global, ep_coord, ep_size, placements, dp_size, dp_rank
):
    """Slice an EP-composition expert-LoRA adapter from rank-0's GLOBAL (all-experts) tensor down to
    THIS rank's local FSDP shard — the inverse of ``shard_expert_lora`` + the FSDP dp/cp sharding, used
    by the cpu_ram_efficient load path. First take this ep-group's experts, then this rank's dp/cp shard
    along each Shard placement.

    ``ep_dim`` is the adapter's expert axis: 0 for lora_A's expert-major ``[E*r, in]`` rows, 1 for
    lora_B's ``[out, r*E]`` columns. lora_B's experts are the LAST axis of the ``[out, r, E]`` view (NOT
    contiguous in the flat ``r*E`` dim), so a plain ``chunk`` on dim 1 would pick a rank-component, not
    this ep-group's experts — hence the reshape-slice that mirrors :func:`_slice_expert_lora_param`.
    """
    from torch.distributed.tensor import Shard

    e_local = e_global // ep_size
    start, end = ep_coord * e_local, (ep_coord + 1) * e_local
    if ep_dim == 0:  # lora_A: [E*r, in] expert-major rows
        r = global_adapter.shape[0] // e_global
        ep_slice = global_adapter[start * r : end * r, :]
    else:  # lora_B: [out, r*E] -> [out, r, E], experts on the last axis
        out_dim = global_adapter.shape[0]
        r = global_adapter.shape[1] // e_global
        ep_slice = global_adapter.reshape(out_dim, r, e_global)[
            :, :, start:end
        ].reshape(out_dim, r * e_local)
    local = ep_slice
    for placement in placements:
        if isinstance(placement, Shard):
            local = local.chunk(dp_size, dim=placement.dim)[dp_rank]
    return local.contiguous()


def shard_expert_lora(model, ep_size: int) -> int:
    """Slice PEFT ``target_parameters`` expert LoRA to each rank's local experts.

    PEFT sizes the LoRA for a 3D ``experts.{gate_up,down}_proj`` from the parameter's
    own dim-0 (the *global* expert count) at adapter-application time, before EP's
    weight slice takes effect on the parameter PEFT wrapped. Left alone, the fused
    EP kernel (``num_experts = E_local``) and the FSDP2 parametrize merge both see a
    full-expert LoRA against a local-expert weight -> shape mismatch. This realigns
    the LoRA with the EP-sharded weights (same ``[offset:offset+E_local]`` slice) and
    registers the ``1/ep_size`` expert grad-scale on the new params. Idempotent.

    Run AFTER PEFT applies the adapter and BEFORE FSDP wraps. Returns the count of
    LoRA params sliced.
    """
    if ep_size <= 1:
        return 0

    scale = 1.0 / ep_size

    def _scale_hook(p):
        if p.grad is not None:
            p.grad.mul_(scale)

    n = 0
    for _name, wrapper in model.named_modules():
        if not _is_param_wrapper(wrapper):
            continue
        if getattr(wrapper, "_ep_lora_sharded", False):
            continue
        base = _real_experts_base(wrapper)
        e_local = getattr(base, "num_local_experts", None) if base is not None else None
        e_global = (
            getattr(base, "num_experts_global", None) if base is not None else None
        )
        if e_local is None or e_global is None or e_local >= e_global:
            continue
        start = base.local_expert_offset
        end = start + e_local

        for adapters, dim in (
            (getattr(wrapper, "lora_A", {}), 0),
            (getattr(wrapper, "lora_B", {}), 1),
        ):
            for ad in list(adapters.keys()):
                _slice_expert_lora_param(adapters[ad], dim, e_global, start, end)
                adapters[ad].weight.register_post_accumulate_grad_hook(_scale_hook)
                n += 1
        wrapper._ep_lora_sharded = True

    if n:
        LOG.info(
            f"Sharded {n} expert-LoRA param(s) to local experts "
            f"(ep_size={ep_size}, grad-scale=1/{ep_size})."
        )
    return n


def save_ep_lora_adapter(model, output_dir: str, ep_group) -> bool:
    """Write a complete LoRA adapter when experts are EP-sharded.

    The attention/router LoRA is replicated across EP, but ``target_parameters`` expert
    LoRA is EP-sharded (each rank holds ``[offset:offset+E_local]``). A plain save would
    persist only the local rank's experts. This gathers each adapter param to a full
    tensor (FSDP all-gather via ``full_tensor`` + EP all-gather for expert LoRA), renames
    to PEFT adapter keys, and writes ``adapter_model.safetensors`` on rank 0. Returns
    ``True`` if it handled the save.
    """
    from pathlib import Path

    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file

    # The EP-sharded 3D expert params and the global expert count. Gather straight from the
    # ParamWrappers by parameter_name (chaining `base_layer` through FSDP-wrapped units to reach
    # the experts module via `_real_experts_base` is brittle once the outer wrapper is its own
    # FSDP unit).
    expert_param_names: set = set()
    e_global = None
    for _n, m in model.named_modules():
        if (
            getattr(m, "num_local_experts", None) is not None
            and getattr(m, "num_experts_global", None) is not None
            and m.num_local_experts < m.num_experts_global
        ):
            e_global = m.num_experts_global
            for pn in ("gate_up_proj", "down_proj"):
                if hasattr(m, pn):
                    expert_param_names.add(pn)
    if e_global is None or not expert_param_names:
        return False

    expert_wrappers = [
        (wname, wrapper)
        for wname, wrapper in model.named_modules()
        if _is_param_wrapper(wrapper)
        and getattr(wrapper, "parameter_name", None) in expert_param_names
    ]
    if not expert_wrappers:
        return False

    active = model.active_adapter if hasattr(model, "active_adapter") else "default"
    if isinstance(active, (list, tuple)):
        active = active[0]

    # Prefer the ep group captured at FSDP-setup time; re-resolving from cfg at save can
    # return a stale/degenerate (size-1) mesh.
    stashed = getattr(model, "_ep_lora_group", None)
    if stashed is not None:
        ep_group = stashed

    # Replicated (attention/router) LoRA: full tensors via FSDP all-gather, canonical PEFT keys.
    sd = {
        name: (p.full_tensor() if type(p).__name__ == "DTensor" else p.data).detach()
        for name, p in model.named_parameters()
        if "lora_" in name
    }
    adapter_sd = get_peft_model_state_dict(model, state_dict=sd)

    # Expert LoRA: gather each wrapper's adapter across FSDP (dp_shard) + EP, key by module name.
    gathered = 0
    for wname, wrapper in expert_wrappers:
        # Only EP×dp_shard/cp composition physically slices the adapter to E_local (shard_expert_lora
        # sets _ep_lora_sharded), so it must be EP-all-gathered back to E_global. Pure EP keeps the
        # adapter GLOBAL (E_global, forward-sliced at runtime) — gathering it would replicate every
        # expert ep_size times into an oversized adapter, so write the FSDP-gathered tensor as-is.
        ep_sharded = getattr(wrapper, "_ep_lora_sharded", False)
        for sub, kind in (("lora_A", "A"), ("lora_B", "B")):
            for w in (mod.weight for mod in getattr(wrapper, sub, {}).values()):
                full_local = (
                    (w.full_tensor() if type(w).__name__ == "DTensor" else w.data)
                    .detach()
                    .contiguous()
                )
                full = (
                    gather_expert_lora_full(full_local, kind, e_global, ep_group)
                    if ep_sharded
                    else full_local
                )
                key = f"{wname}.{sub}.weight"
                target = (
                    key
                    if key in adapter_sd
                    else next(
                        (
                            k
                            for k in adapter_sd
                            if k.endswith(key.split("base_model.model.")[-1])
                        ),
                        None,
                    )
                )
                if target is not None:
                    adapter_sd[target] = full
                    gathered += 1
                else:
                    LOG.warning(f"EP-LoRA save: no adapter key for {key!r}.")

    if dist.get_rank() == 0:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.cpu().contiguous() for k, v in adapter_sd.items()},
            str(out / "adapter_model.safetensors"),
        )
        model.peft_config[active].save_pretrained(str(out))
        LOG.info(
            f"Saved EP-gathered LoRA adapter ({len(adapter_sd)} tensors, "
            f"{gathered} expert params gathered across EP) to {out}."
        )
    dist.barrier()
    return True


def save_fsdp2_lora_adapter(model, output_dir: str) -> bool:
    """Write a complete LoRA adapter under FSDP2 WITHOUT expert parallelism.

    The DCP ``SHARDED_STATE_DICT`` save fails ("Failed to validate global plan") on the frozen NVFP4
    base params (torchao tensor-subclass DTensors the planner can't validate). For a LoRA run we only
    need the (tiny) adapter, so gather each ``lora_`` param to a full tensor via FSDP all-gather
    (``DTensor.full_tensor``) and write ``adapter_model.safetensors`` on rank 0. ``target_parameters``
    expert LoRA lives on PEFT ParamWrappers (``lora_A``/``lora_B`` submodules) — gather those too and
    key by module name (no EP axis to gather here, unlike :func:`save_ep_lora_adapter`).

    Returns ``True`` if it handled the save (model has LoRA params), else ``False``.
    """
    from pathlib import Path

    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import save_file

    if not any("lora_" in n for n, _ in model.named_parameters()):
        return False

    active = model.active_adapter if hasattr(model, "active_adapter") else "default"
    if isinstance(active, (list, tuple)):
        active = active[0]

    # Replicated + dp-sharded LoRA: full tensors via FSDP all-gather (collective — same iteration
    # order on every rank). Canonical PEFT keys via get_peft_model_state_dict.
    sd = {
        name: (p.full_tensor() if type(p).__name__ == "DTensor" else p.data).detach()
        for name, p in model.named_parameters()
        if "lora_" in name
    }
    adapter_sd = get_peft_model_state_dict(model, state_dict=sd)

    gathered = 0
    for wname, wrapper in model.named_modules():
        if not _is_param_wrapper(wrapper):
            continue
        for sub in ("lora_A", "lora_B"):
            for w in (mod.weight for mod in getattr(wrapper, sub, {}).values()):
                full = (
                    w.full_tensor() if type(w).__name__ == "DTensor" else w.data
                ).detach()
                key = f"{wname}.{sub}.weight"
                target = (
                    key
                    if key in adapter_sd
                    else next(
                        (
                            k
                            for k in adapter_sd
                            if k.endswith(key.split("base_model.model.")[-1])
                        ),
                        None,
                    )
                )
                if target is not None:
                    adapter_sd[target] = full
                    gathered += 1

    if not dist.is_initialized() or dist.get_rank() == 0:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.cpu().contiguous() for k, v in adapter_sd.items()},
            str(out / "adapter_model.safetensors"),
        )
        model.peft_config[active].save_pretrained(str(out))
        LOG.info(
            f"Saved FSDP2-gathered LoRA adapter ({len(adapter_sd)} tensors, "
            f"{gathered} expert params) to {out}."
        )
    if dist.is_initialized():
        dist.barrier()
    return True


def gather_expert_lora_full(local: torch.Tensor, kind: str, e_global: int, ep_group):
    """Inverse of the EP LoRA slice: all-gather a local-experts LoRA tensor across the
    EP group and reassemble the full ``e_global``-expert tensor in the PEFT layout.

      * ``kind="A"`` (expert-major ``[E*r, in]``): concat gathered slices along rows.
      * ``kind="B"`` (rank-major ``[out, r*E]``): place each rank's experts into the
        ``E`` axis of ``[out, r, E]`` and flatten.
    """
    ep_size = dist.get_world_size(ep_group)
    gathered = [torch.empty_like(local) for _ in range(ep_size)]
    dist.all_gather(gathered, local.contiguous(), group=ep_group)
    if kind == "A":
        return torch.cat(gathered, dim=0)
    out_dim = local.shape[0]
    e_local = e_global // ep_size
    r = local.shape[1] // e_local
    parts = [g.reshape(out_dim, r, e_local) for g in gathered]
    return torch.cat(parts, dim=2).reshape(out_dim, r * e_global)
