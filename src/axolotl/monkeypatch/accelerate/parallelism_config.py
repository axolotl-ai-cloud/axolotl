"""ParallelismConfig monkeypatch.

Two extensions:
- Allow pure CP standalone via `ACCELERATE_ALLOW_CP_STANDALONE`.
- Add Expert Parallel (`ep`) as a first-class mesh axis inside the
  data-parallel group. Mesh order is `(ep, dp_replicate, dp_shard, cp, sp, tp)`
  so the dp axes stay contiguous (required for `_flatten("dp")`).

See `expert_parallel/README.md` for the full integration story.
"""

import os
import warnings

from accelerate import DistributedType


def _patched_post_init(self):
    _ORIG_POST_INIT(self)

    if not hasattr(self, "ep_size") or self.ep_size is None:
        self.ep_size = int(os.environ.get("PARALLELISM_CONFIG_EP_SIZE", "1") or 1)
    if self.ep_size < 1:
        raise ValueError(f"ep_size must be at least 1, got {self.ep_size}")

    # Register so `_set_size`, `_validate_accelerator`, `_get_mesh` see it.
    self._sizes["ep"] = self.ep_size


def _patched_total_size(self):
    return (
        self.dp_replicate_size
        * self.dp_shard_size
        * self.tp_size
        * self.cp_size
        * self.sp_size
        * getattr(self, "ep_size", 1)
    )


def _patched_ep_enabled(self):
    return getattr(self, "ep_size", 1) > 1


def _patched_dp_dim_names(self):
    """DP axes (different ranks see different data). EP is included — each
    EP rank pulls its own batch."""
    dims = []
    if self.ep_enabled:
        dims += ["ep"]
    if self.dp_replicate_enabled:
        dims += ["dp_replicate"]
    if self.dp_shard_enabled:
        dims += ["dp_shard"]
    return dims


def _patched_dp_shard_cp_dim_names(self):
    """Axes the outer FSDP wrap shards along (flattened into `dp_shard_cp`).
    Including `ep` makes non-expert grads reduce-scatter across the full
    world; experts are pre-wrapped on `mesh["dp_shard"]` only and skipped
    by the auto-wrap walker."""
    dims = []
    if self.ep_enabled:
        dims += ["ep"]
    if self.dp_shard_enabled:
        dims += ["dp_shard"]
    if self.cp_enabled:
        dims += ["cp"]
    return dims


def _patched_non_dp_dim_names(self):
    """Non-DP axes (TP/CP/SP). EP moved into `dp_dim_names`."""
    dims = []
    if self.tp_enabled:
        dims += ["tp"]
    if self.cp_enabled:
        dims += ["cp"]
    if self.sp_enabled:
        dims += ["sp"]
    return dims


def _patched_get_mesh(self):
    """Build (dim_names, shape) for `init_device_mesh`. Order keeps the dp
    block (ep, dp_replicate, dp_shard) contiguous so `_flatten("dp")` works.
    """
    mesh_dims = {p: self._sizes[p] for p in self.active_mesh_dims}
    mesh_order = ["ep", "dp_replicate", "dp_shard", "cp", "sp", "tp"]
    sorted_items = sorted(mesh_dims.items(), key=lambda x: mesh_order.index(x[0]))
    return tuple(zip(*sorted_items, strict=True))


def _validate_accelerator(self, accelerator):
    _warnings = set()
    if not accelerator.multi_device and self.total_size == 1:
        # No distributed setup, valid parallelism config
        return

    # We need this to ensure DDP works
    if self.total_size == 1:
        self._set_size("dp_replicate", accelerator.num_processes)

    if self.total_size != accelerator.num_processes:
        raise ValueError(
            f"ParallelismConfig total_size ({self.total_size}) does not match "
            f"num_processes ({accelerator.num_processes}). Please adjust dp_replicate_size/ "
            f"dp_shard_size/tp_size/cp_size/ep_size."
        )

    # allow parallelism config when not using fsdp if using pure context parallelism
    allow_parallelism_config = False

    if (
        self.cp_size > 1
        and self.dp_shard_size <= 1
        and os.environ.get("ACCELERATE_ALLOW_CP_STANDALONE", "false").lower() == "true"
    ):
        allow_parallelism_config = True

    # Pure EP (no FSDP/TP/CP) is valid: the plugin handles dispatch/combine
    # and DDP's _ddp_params_and_buffers_to_ignore keeps experts out of DDP.
    if (
        getattr(self, "ep_enabled", False)
        and self.dp_shard_size <= 1
        and self.tp_size <= 1
        and self.cp_size <= 1
    ):
        allow_parallelism_config = True

    if (
        self.total_size > 1
        and not allow_parallelism_config
        and not (accelerator.is_fsdp2 or accelerator.multi_device)
    ):
        raise ValueError(
            f"ParallelismConfig is only compatible DistributedType.FSDP (version 2) or DistributedType.Multi{{Device}}, but got {accelerator.distributed_type}."
        )

    for parallelism, size in self._sizes.items():
        if size == 1 and getattr(self, f"{parallelism}_handler", None) is not None:
            _warnings.add(
                f"ParallelismConfig.{parallelism}_handler is set, but {parallelism}_size is set to 1. This handler will be ignored."
            )

    if _warnings and accelerator.is_main_process:
        warnings.warn(
            "ParallelismConfig has the following warnings:\n" + "\n".join(_warnings),
            UserWarning,
            stacklevel=2,
        )


def patched_is_fsdp2(self) -> bool:
    """
    Patched version of is_fsdp2 that guards against a None fsdp_plugin.
    """
    # The new logic checks if fsdp_plugin exists before accessing its attributes
    return (
        self.distributed_type == DistributedType.FSDP
        and self.fsdp_plugin
        and self.fsdp_plugin.fsdp_version == 2
    )


# Captured in `patch_parallelism_config()` so we can chain the original
# __post_init__ before adding ep.
_ORIG_POST_INIT = None


def patch_parallelism_config():
    global _ORIG_POST_INIT
    from accelerate.accelerator import AcceleratorState, ParallelismConfig

    if _ORIG_POST_INIT is None:
        _ORIG_POST_INIT = ParallelismConfig.__post_init__

    ParallelismConfig.__post_init__ = _patched_post_init
    # `total_size` is a property on the dataclass; replace it.
    ParallelismConfig.total_size = property(_patched_total_size)
    ParallelismConfig.ep_enabled = property(_patched_ep_enabled)
    ParallelismConfig.dp_dim_names = property(_patched_dp_dim_names)
    ParallelismConfig.dp_shard_cp_dim_names = property(_patched_dp_shard_cp_dim_names)
    ParallelismConfig.non_dp_dim_names = property(_patched_non_dp_dim_names)
    ParallelismConfig._get_mesh = _patched_get_mesh
    ParallelismConfig._validate_accelerator = _validate_accelerator
    AcceleratorState.is_fsdp2 = property(patched_is_fsdp2)
    patch_prepare_data_loader_for_ep()
    patch_clip_grad_norm_for_ep()


def _ep_aware_clip_grad_norm(parameters, max_norm, norm_type=2.0):
    """`clip_grad_norm_` for params sharded across different DeviceMeshes.

    Stock `torch.nn.utils.clip_grad_norm_` stacks per-param norms, which
    DTensor rejects across meshes (experts on `dp_shard` vs non-experts on
    `dp_shard_cp`). Instead, compute the local p-norm contribution per rank,
    all-reduce the sum across the world, take the p-th root, and apply the
    clip coefficient. Supports any finite p ≥ 1 plus `inf`.
    """
    import math

    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    device = grads[0].device
    if isinstance(grads[0], DTensor):
        device = grads[0].to_local().device

    is_inf = math.isinf(norm_type)
    local_acc = torch.zeros((), device=device, dtype=torch.float32)
    for g in grads:
        local = g.to_local() if isinstance(g, DTensor) else g
        local_f32 = local.detach().to(torch.float32)
        if is_inf:
            local_acc = torch.maximum(local_acc, local_f32.abs().max())
        else:
            local_acc = local_acc + local_f32.abs().pow(norm_type).sum()

    if dist.is_available() and dist.is_initialized():
        op = dist.ReduceOp.MAX if is_inf else dist.ReduceOp.SUM
        dist.all_reduce(local_acc, op=op)

    total_norm = local_acc if is_inf else local_acc.pow(1.0 / norm_type)
    clip_coef = (float(max_norm) / (total_norm + 1e-6)).clamp(max=1.0)
    for g in grads:
        local = g.to_local() if isinstance(g, DTensor) else g
        local.detach().mul_(clip_coef.to(local.dtype))

    return total_norm.to(
        grads[0].dtype if grads[0].is_floating_point() else torch.float32
    )


def patch_clip_grad_norm_for_ep():
    """Replace `Accelerator.clip_grad_norm_` with the EP-aware version when
    the active parallelism includes both `ep` and `dp_shard` (i.e., the
    FSDP+EP composition produces multi-mesh DTensor grads).
    """
    from accelerate import Accelerator

    if getattr(Accelerator, "_AXOLOTL_EP_CLIP_PATCHED", False):
        return
    orig = Accelerator.clip_grad_norm_

    def patched_clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        pc = getattr(self, "parallelism_config", None)
        if (
            pc is not None
            and getattr(pc, "ep_enabled", False)
            and getattr(pc, "dp_shard_enabled", False)
        ):
            self.unscale_gradients()
            params = list(parameters)
            return _ep_aware_clip_grad_norm(params, max_norm, norm_type=norm_type)
        return orig(self, parameters, max_norm, norm_type=norm_type)

    Accelerator.clip_grad_norm_ = patched_clip_grad_norm_
    Accelerator._AXOLOTL_EP_CLIP_PATCHED = True


def patch_prepare_cp():
    import contextlib

    from accelerate import Accelerator

    def patched_prepare_cp(self, *args):
        if self.parallelism_config.cp_backend == "deepspeed":
            return args

        @contextlib.contextmanager
        def _noop_cp_context(
            buffers=None, buffer_seq_dims=None, no_restore_buffers=None
        ):
            yield

        self._cp_context = _noop_cp_context
        return args

    Accelerator._prepare_cp = patched_prepare_cp


def _patched_prepare_data_loader_factory(orig_fn):
    """Wrap `accelerate.data_loader.prepare_data_loader` to count the EP axis
    as a data-parallel dimension.

    Stock accelerate (line ~1155 in 1.13.0) computes
        num_processes = dp_shard * dp_replicate
        process_index = process_index // (tp * cp)
    which ignores EP. EP ranks see DIFFERENT data (each rank pulls its own
    batch), so EP belongs in the data-parallel size — same way `dp_replicate`
    does.
    """
    import torch

    def patched(*args, **kwargs):
        torch_device_mesh = kwargs.get("torch_device_mesh", None)
        if (
            torch_device_mesh is not None
            and isinstance(torch_device_mesh, torch.distributed.device_mesh.DeviceMesh)
            and "ep" in torch_device_mesh.mesh_dim_names
        ):
            from accelerate.state import PartialState
            from accelerate.utils import DistributedType

            state = PartialState()
            if state.distributed_type != DistributedType.DEEPSPEED:
                ep_size = torch_device_mesh["ep"].size()
                tp_size = (
                    torch_device_mesh["tp"].size()
                    if "tp" in torch_device_mesh.mesh_dim_names
                    else 1
                )
                cp_size = (
                    torch_device_mesh["cp"].size()
                    if "cp" in torch_device_mesh.mesh_dim_names
                    else 1
                )
                fsdp_size = (
                    torch_device_mesh["dp_shard"].size()
                    if "dp_shard" in torch_device_mesh.mesh_dim_names
                    else 1
                )
                dp_size = (
                    torch_device_mesh["dp_replicate"].size()
                    if "dp_replicate" in torch_device_mesh.mesh_dim_names
                    else 1
                )
                num_processes = fsdp_size * dp_size * ep_size
                process_index = state.process_index // (tp_size * cp_size)
                kwargs["num_processes"] = num_processes
                kwargs["process_index"] = process_index
                # Once we've supplied num_processes/process_index explicitly,
                # accelerate's internal mesh path (which would re-derive without
                # ep) is bypassed.
                kwargs["torch_device_mesh"] = None
        return orig_fn(*args, **kwargs)

    return patched


def patch_prepare_data_loader_for_ep():
    """Apply the EP-aware data-loader patch.

    Idempotent: replacing the bound function more than once is harmless because
    the wrapper closes over the *current* `prepare_data_loader`.
    """
    import accelerate as _accel
    from accelerate import data_loader as _dl

    if getattr(_dl, "_AXOLOTL_EP_PATCHED", False):
        return
    orig = _dl.prepare_data_loader
    wrapped = _patched_prepare_data_loader_factory(orig)
    _dl.prepare_data_loader = wrapped
    # accelerate.Accelerator imports prepare_data_loader at module load, so
    # we have to patch the binding it captured too.
    if hasattr(_accel, "prepare_data_loader"):
        _accel.prepare_data_loader = wrapped
    # Likewise the Accelerator module's local reference.
    from accelerate import accelerator as _acc_mod

    if hasattr(_acc_mod, "prepare_data_loader"):
        _acc_mod.prepare_data_loader = wrapped
    _dl._AXOLOTL_EP_PATCHED = True
