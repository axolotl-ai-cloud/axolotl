"""Compare the MeshSpec lowerings with the layout derivations they replaced, row by row.

Run as a script (``python _mesh_layout_matrix.py OUT.json``) so accelerate's
``ParallelismConfig`` monkeypatch stays out of the test process. Each row records whether the
replaced code produced a layout accelerate accepts, the kwargs and environment both versions
emit, and whether accelerate 1.15 accepts the new ones (patched validator under FSDP, CP or EP,
as at runtime; the stock one otherwise).
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

WORLDS = (1, 2, 4, 8, 16)
DISTS = ("ddp", "fsdp", "deepspeed")
TP = (1, 2, 4)
CP = (1, 2, 4)
EP = (1, 2, 8)
DP_SHARD = (None, 1, 2, 4, 8, 16)
DP_REPLICATE = (None, 1, 2, 4)


def legacy_parallel_config_kwargs(
    world_size,
    tensor_parallel_size=1,
    context_parallel_size=1,
    dp_shard_size=None,
    dp_replicate_size=None,
    is_fsdp=False,
    expert_parallel_size=None,
    is_deepspeed=False,
):
    """``axolotl.utils.distributed._get_parallel_config_kwargs`` as of main @27bb23ce38."""
    pc_kwargs = {}
    remaining_world_size = world_size
    if expert_parallel_size and expert_parallel_size > 1:
        if remaining_world_size % expert_parallel_size != 0:
            raise ValueError("expert_parallel_size must divide world_size")
        remaining_world_size = remaining_world_size // expert_parallel_size
    if tensor_parallel_size and tensor_parallel_size > 1:
        pc_kwargs["tp_size"] = tensor_parallel_size
        remaining_world_size = remaining_world_size // tensor_parallel_size
    if context_parallel_size and context_parallel_size > 1:
        if is_deepspeed:
            pc_kwargs["sp_size"] = context_parallel_size
            pc_kwargs["sp_backend"] = "deepspeed"
        else:
            pc_kwargs["cp_size"] = context_parallel_size
        remaining_world_size = remaining_world_size // context_parallel_size
    if is_deepspeed and "sp_size" in pc_kwargs and remaining_world_size > 1:
        if dp_replicate_size and dp_replicate_size > 1:
            if dp_replicate_size != remaining_world_size:
                raise ValueError("dp_replicate_size must equal world / cp")
        pc_kwargs["dp_replicate_size"] = remaining_world_size
        return pc_kwargs
    if dp_shard_size is None and dp_replicate_size in (None, 1):
        if remaining_world_size > 1:
            pc_kwargs["dp_shard_size"] = remaining_world_size
            remaining_world_size = 1
    if dp_replicate_size and dp_replicate_size > 1:
        pc_kwargs["dp_replicate_size"] = dp_replicate_size
        remaining_world_size = remaining_world_size // dp_replicate_size
    if remaining_world_size > 1 and dp_shard_size and dp_shard_size > 1:
        if not is_fsdp:
            raise ValueError("dp_shard_size was configured without fsdp_config")
        pc_kwargs["dp_shard_size"] = dp_shard_size
        remaining_world_size = remaining_world_size // dp_shard_size
        if remaining_world_size > 1 and "dp_replicate_size" not in pc_kwargs:
            pc_kwargs["dp_replicate_size"] = remaining_world_size
            remaining_world_size = 1
    if remaining_world_size > 1:
        if "dp_shard_size" not in pc_kwargs and is_fsdp:
            pc_kwargs["dp_shard_size"] = remaining_world_size
            remaining_world_size = 1
    if remaining_world_size > 1:
        raise ValueError("parallelisms incompatible with the world size")
    return pc_kwargs


def legacy_accelerate_env(cfg, world_size):
    """``setup_parallelism_envs`` + ``setup_deepspeed_context_parallel_envs`` as of main
    @27bb23ce38, returning the variables they set."""
    env = {}
    use = False
    if cfg.tensor_parallel_size and cfg.tensor_parallel_size > 1:
        use = True
        env["PARALLELISM_CONFIG_TP_SIZE"] = str(cfg.tensor_parallel_size)
    if cfg.dp_shard_size and cfg.dp_shard_size > 1:
        use = True
        env["PARALLELISM_CONFIG_DP_SHARD_SIZE"] = str(cfg.dp_shard_size)
    if cfg.dp_replicate_size and cfg.dp_replicate_size > 1:
        use = True
        env["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] = str(cfg.dp_replicate_size)
    if cfg.context_parallel_size and cfg.context_parallel_size > 1:
        use = True
        if cfg.deepspeed:
            cp_size = cfg.context_parallel_size
            env["PARALLELISM_CONFIG_SP_SIZE"] = str(cp_size)
            env["PARALLELISM_CONFIG_SP_BACKEND"] = "deepspeed"
            env["PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE"] = "true"
            env["PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION"] = (
                cfg.attn_implementation or "sdpa"
            )
            non_dp = cp_size * (cfg.tensor_parallel_size or 1)
            if world_size > 1 and world_size % non_dp == 0 and world_size // non_dp > 1:
                env["PARALLELISM_CONFIG_DP_REPLICATE_SIZE"] = str(world_size // non_dp)
            env.pop("PARALLELISM_CONFIG_CP_SIZE", None)
        else:
            env["PARALLELISM_CONFIG_CP_SIZE"] = str(cfg.context_parallel_size)
            env["ACCELERATE_ALLOW_CP_STANDALONE"] = "true"
    if cfg.expert_parallel_size and cfg.expert_parallel_size > 1:
        env["PARALLELISM_CONFIG_EP_SIZE"] = str(cfg.expert_parallel_size)
        if cfg.dp_shard_size and cfg.dp_shard_size > 1:
            use = True
    if use:
        env["ACCELERATE_USE_PARALLELISM_CONFIG"] = "true"
    return env


def rows():
    for world, dist, tp, cp, ep, ds, dr in itertools.product(
        WORLDS, DISTS, TP, CP, EP, DP_SHARD, DP_REPLICATE
    ):
        if tp * cp > world or ep > world:
            continue
        # validation rejects these before the layout is derived
        if ep > 1 and (dist != "fsdp" or tp > 1):
            continue
        yield {
            "world": world,
            "dist": dist,
            "tp": tp,
            "cp": cp,
            "ep": ep,
            "dp_shard": ds,
            "dp_replicate": dr,
        }


def make_cfg(row):
    from axolotl.utils.dict import DictDefault

    return DictDefault(
        tensor_parallel_size=row["tp"],
        context_parallel_size=row["cp"],
        expert_parallel_size=row["ep"],
        dp_shard_size=row["dp_shard"],
        dp_replicate_size=row["dp_replicate"],
        fsdp_config={"fsdp_version": 2} if row["dist"] == "fsdp" else None,
        deepspeed="deepspeed_configs/zero2.json"
        if row["dist"] == "deepspeed"
        else None,
        attn_implementation="flash_attention_2",
    )


class _Validators:
    """Stock and Axolotl-patched ``ParallelismConfig`` behaviour, switchable per row."""

    NAMES = (
        "__post_init__",
        "total_size",
        "_validate_accelerator",
        "_get_mesh",
        "build_device_mesh",
        "dp_dim_names",
        "dp_shard_cp_dim_names",
        "non_dp_dim_names",
        "dp_cp_dim_names",
    )

    def __init__(self):
        from accelerate.parallelism_config import ParallelismConfig

        from axolotl.monkeypatch.accelerate.parallelism_config import (
            patch_parallelism_config,
        )

        self.cls = ParallelismConfig
        self.stock = {name: ParallelismConfig.__dict__[name] for name in self.NAMES}
        patch_parallelism_config()
        self.patched = {
            name: ParallelismConfig.__dict__[name]
            for name in (*self.NAMES, "ep_enabled")
        }

    def use(self, patched):
        for name, value in (self.patched if patched else self.stock).items():
            setattr(self.cls, name, value)
        if not patched and "ep_enabled" in self.cls.__dict__:
            delattr(self.cls, "ep_enabled")

    def check(self, row, env, kwargs):
        """None when accelerate accepts both the env-built and the kwargs-built config."""
        from accelerate import DistributedType

        world = row["world"]
        fsdp = row["dist"] == "fsdp"
        deepspeed = row["dist"] == "deepspeed"
        self.use(fsdp or row["cp"] > 1 or row["ep"] > 1)
        accelerator = SimpleNamespace(
            multi_device=world > 1,
            num_processes=world,
            is_fsdp2=fsdp,
            is_main_process=True,
            distributed_type=(
                DistributedType.FSDP
                if fsdp
                else DistributedType.DEEPSPEED
                if deepspeed
                else DistributedType.MULTI_GPU
            ),
        )
        with patch.dict(os.environ, env, clear=False):
            try:
                if env.get("ACCELERATE_USE_PARALLELISM_CONFIG") == "true":
                    self.cls()._validate_accelerator(accelerator)
                if kwargs:
                    total = self.cls(**kwargs).total_size
                    if total != world and not (deepspeed and row["cp"] > 1):
                        return f"kwargs total {total}"
            except ValueError as err:
                return str(err).splitlines()[0]
        return None


def _clean_env():
    for key in list(os.environ):
        if key.startswith(("PARALLELISM_CONFIG_", "ACCELERATE_")):
            del os.environ[key]


def evaluate(row, validators):
    from axolotl.core.plan.errors import PlanError
    from axolotl.utils import distributed as dist_utils, trainer as trainer_utils

    cfg = make_cfg(row)
    world = row["world"]
    result = dict(row)

    legacy_error = None
    try:
        legacy_kwargs = legacy_parallel_config_kwargs(
            world,
            row["tp"],
            row["cp"],
            row["dp_shard"],
            row["dp_replicate"],
            row["dist"] == "fsdp",
            row["ep"],
            is_deepspeed=row["dist"] == "deepspeed",
        )
    except ValueError as err:
        legacy_kwargs, legacy_error = None, f"kwargs: {err}"
    legacy_env = legacy_accelerate_env(cfg, world)
    if legacy_error is None:
        legacy_error = validators.check(row, legacy_env, legacy_kwargs)
    if legacy_error is None and row["ep"] > 1:
        # ExpertParallelPlugin._resolve_ep_group as of main @27bb23ce38
        product = (
            row["ep"] * (row["dp_replicate"] or 1) * (row["dp_shard"] or 1) * row["cp"]
        )
        filled = (
            row["ep"] == world
            or max(row["dp_shard"] or 1, row["cp"], row["dp_replicate"] or 1) > 1
        )
        if product != world or not filled:
            legacy_error = f"ep plugin product {product}"
    result.update(
        legacy_ok=legacy_error is None,
        legacy_error=legacy_error,
        legacy_kwargs=legacy_kwargs,
        legacy_env=legacy_env,
    )

    os.environ["WORLD_SIZE"] = str(world)
    _clean_env()
    built = []

    class _RecordingParallelismConfig:
        def __init__(self, **kwargs):
            built.append(kwargs)

        def build_device_mesh(self, device_type):
            return None

    try:
        mesh = dist_utils.resolve_cfg_mesh(cfg)
    except PlanError as err:
        result.update(plan_error=err.code, plan_key=err.key, plan_hint=err.hint)
        return result
    with (
        patch.object(trainer_utils, "get_world_size", lambda: world),
        patch("axolotl.monkeypatch.accelerate.tp.patch_accelerate_prepare_tp"),
        patch(
            "axolotl.monkeypatch.accelerate.parallelism_config.patch_parallelism_config"
        ),
        patch.object(dist_utils, "ParallelismConfig", _RecordingParallelismConfig),
    ):
        trainer_utils.setup_parallelism_envs(cfg)
        new_env = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("PARALLELISM_CONFIG_", "ACCELERATE_"))
        }
        _clean_env()
        dist_utils.build_parallelism_config(cfg)
    new_kwargs = built[0] if built else {}
    result.update(
        plan_error=None,
        mesh={
            "dist": mesh.dist,
            "dp_replicate": mesh.dp_replicate,
            "dp_shard": mesh.dp_shard,
            "cp": mesh.cp,
            "cp_impl": mesh.cp_impl,
            "tp": mesh.tp,
            "ep": mesh.ep,
            "efsdp": mesh.efsdp,
            "legacy_spelling": mesh.legacy_spelling,
        },
        new_kwargs=new_kwargs,
        new_kwargs_order=list(new_kwargs),
        new_env=new_env,
        new_error=validators.check(row, new_env, new_kwargs),
    )
    if result["legacy_kwargs"] is not None:
        result["legacy_kwargs_order"] = list(result["legacy_kwargs"])
    return result


def main(out_path):
    _clean_env()
    validators = _Validators()
    results = [evaluate(row, validators) for row in rows()]
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(results, fout)


if __name__ == "__main__":
    main(sys.argv[1])
