"""ParallelismConfig monkeypatch.

Two extensions:
- Allow pure CP standalone via `ACCELERATE_ALLOW_CP_STANDALONE`.
- Add Expert Parallel (`ep`) as a first-class mesh axis inside the
  data-parallel group. Mesh order is `(dp_replicate, dp_shard, cp, ep, sp, tp)`:
  `ep` sits innermost of the data axes so its all-to-all groups are runs of
  consecutive ranks and stay inside a node whenever `ep_size * tp_size` fits.

See `expert_parallel/README.md` for the full integration story.
"""

import os
import warnings

from accelerate import DistributedType

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Outer to inner. Consecutive ranks share a node, so the innermost axes get node-local groups:
# tp (activation all-reduces every layer) first, then ep (dispatch/combine all-to-all).
MESH_ORDER = ("dp_replicate", "dp_shard", "cp", "ep", "sp", "tp")


def ordered_mesh_dims(dims):
    return sorted(dims, key=MESH_ORDER.index)


def data_parallel_index(mesh_dim_names, coordinate, sizes):
    """Row-major index over the data axes (`dp_replicate`, `dp_shard`, `ep`) from a rank's mesh
    coordinate, so ranks that only differ on cp/tp/sp share a data shard whatever the axis order."""
    index = 0
    for name in ("dp_replicate", "dp_shard", "ep"):
        if name in mesh_dim_names:
            index = index * sizes[name] + coordinate[mesh_dim_names.index(name)]
    return index


def mesh_placement_report(shape, names, gpus_per_node):
    """Per axis: the size of its process groups and the fraction of those groups that span nodes,
    assuming rank `r` lives on node `r // gpus_per_node` (the launcher default)."""
    import torch

    ranks = torch.arange(int(torch.tensor(shape).prod())).view(*shape)
    report = {}
    for axis, name in enumerate(names):
        groups = ranks.movedim(axis, -1).reshape(-1, shape[axis]).tolist()
        cross = sum(len({r // gpus_per_node for r in g}) > 1 for g in groups)
        report[name] = (shape[axis], cross / len(groups))
    return report


def log_mesh_placement(mesh):
    gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", "0") or 0)
    names = tuple(mesh.mesh_dim_names or ())
    if gpus_per_node <= 0 or not names:
        return
    report = mesh_placement_report(tuple(mesh.shape), names, gpus_per_node)
    parts = [
        f"{name}={size}{' (crosses nodes)' if cross else ''}"
        for name, (size, cross) in report.items()
    ]
    LOG.info(
        f"parallelism mesh {names} shape={tuple(mesh.shape)}, {gpus_per_node} GPUs/node: "
        + ", ".join(parts),
        main_process_only=True,
    )


def _patched_post_init(self):
    if not hasattr(self, "ep_size") or self.ep_size is None:
        self.ep_size = int(os.environ.get("PARALLELISM_CONFIG_EP_SIZE", "1") or 1)
    if self.ep_size < 1:
        raise ValueError(f"ep_size must be at least 1, got {self.ep_size}")

    try:
        _ORIG_POST_INIT(self)
    except ValueError as exc:
        # accelerate reads dp_shard == 1 as DDP and refuses to compose it with CP under
        # dp_replicate; with EP the dense weights still FSDP-shard over (ep, cp), so the
        # layout is HSDP whose shard group is ep x cp. That check is the last statement
        # before `_sizes` is assigned, so finish the original init here.
        if "pure data parallelism" not in str(exc) or self.ep_size <= 1:
            raise
        self._sizes = {
            "dp_replicate": self.dp_replicate_size,
            "dp_shard": self.dp_shard_size,
            "tp": self.tp_size,
            "cp": self.cp_size,
            "sp": self.sp_size,
        }

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
    if self.dp_replicate_enabled:
        dims += ["dp_replicate"]
    if self.ep_enabled:
        dims += ["ep"]
    if self.dp_shard_enabled:
        dims += ["dp_shard"]
    return ordered_mesh_dims(dims)


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
    return ordered_mesh_dims(dims)


def _patched_dp_cp_dim_names(self):
    dims = list(self.dp_dim_names)
    if self.cp_enabled:
        dims += ["cp"]
    return ordered_mesh_dims(dims)


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
    """Build (dim_names, shape) for `init_device_mesh` in `MESH_ORDER`. `dp_replicate` stays
    outermost so `(dp_replicate, dp_shard_cp)` slices in ascending order; `dp` = (dp_replicate,
    dp_shard, ep) is flattened across the cp gap, which DeviceMesh supports."""
    mesh_dims = {p: self._sizes[p] for p in self.active_mesh_dims}
    sorted_items = sorted(mesh_dims.items(), key=lambda x: MESH_ORDER.index(x[0]))
    return tuple(zip(*sorted_items, strict=True))


def _patched_build_device_mesh(self, device_type):
    mesh = _ORIG_BUILD_DEVICE_MESH(self, device_type)
    if mesh is not None:
        log_mesh_placement(mesh)
    return mesh


def _validate_accelerator(self, accelerator):
    _warnings = set()
    if not accelerator.multi_device and self.total_size == 1:
        # No distributed setup, valid parallelism config
        return

    # We need this to ensure DDP works
    if self.total_size == 1:
        self._set_size("dp_replicate", accelerator.num_processes)

    # DeepSpeed manages SP process groups globally, so total_size (the local parallelism config)
    # need not equal num_processes; keep this branch in sync with accelerate's upstream validator.
    if self.sp_backend == "deepspeed" and self.sp_size > 1:
        pass
    elif self.total_size != accelerator.num_processes:
        raise ValueError(
            f"ParallelismConfig total_size ({self.total_size}) does not match "
            f"num_processes ({accelerator.num_processes}). Please adjust dp_replicate_size/ "
            f"dp_shard_size/tp_size/cp_size/sp_size/ep_size."
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
        and not (
            accelerator.is_fsdp2
            or accelerator.multi_device
            or accelerator.distributed_type == DistributedType.DEEPSPEED
        )
    ):
        raise ValueError(
            f"ParallelismConfig is only compatible with DistributedType.FSDP (version 2), DistributedType.Multi{{Device}}, or DistributedType.DEEPSPEED, but got {accelerator.distributed_type}."
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
_ORIG_BUILD_DEVICE_MESH = None


def patch_parallelism_config():
    global _ORIG_POST_INIT, _ORIG_BUILD_DEVICE_MESH
    from accelerate.accelerator import AcceleratorState, ParallelismConfig

    if _ORIG_POST_INIT is None:
        _ORIG_POST_INIT = ParallelismConfig.__post_init__
    if _ORIG_BUILD_DEVICE_MESH is None:
        _ORIG_BUILD_DEVICE_MESH = ParallelismConfig.build_device_mesh

    ParallelismConfig.__post_init__ = _patched_post_init
    # `total_size` is a property on the dataclass; replace it.
    ParallelismConfig.total_size = property(_patched_total_size)
    ParallelismConfig.ep_enabled = property(_patched_ep_enabled)
    ParallelismConfig.dp_dim_names = property(_patched_dp_dim_names)
    ParallelismConfig.dp_shard_cp_dim_names = property(_patched_dp_shard_cp_dim_names)
    ParallelismConfig.non_dp_dim_names = property(_patched_non_dp_dim_names)
    ParallelismConfig.dp_cp_dim_names = property(_patched_dp_cp_dim_names)
    ParallelismConfig._get_mesh = _patched_get_mesh
    ParallelismConfig.build_device_mesh = _patched_build_device_mesh
    ParallelismConfig._validate_accelerator = _validate_accelerator
    AcceleratorState.is_fsdp2 = property(patched_is_fsdp2)
    patch_prepare_data_loader_for_ep()


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
                names = tuple(torch_device_mesh.mesh_dim_names)
                process_index = data_parallel_index(
                    names,
                    torch_device_mesh.get_coordinate(),
                    {n: torch_device_mesh[n].size() for n in names},
                )
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
