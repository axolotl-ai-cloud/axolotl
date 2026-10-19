"""Mesh axis order, node placement, and the data-parallel index derived from coordinates."""

from types import SimpleNamespace

import pytest

from axolotl.monkeypatch.accelerate.parallelism_config import (
    MESH_ORDER,
    _patched_dp_cp_dim_names,
    _patched_dp_dim_names,
    _patched_dp_shard_cp_dim_names,
    _patched_get_mesh,
    data_parallel_index,
    mesh_placement_report,
)


def _config(**sizes):
    active = [n for n in MESH_ORDER if sizes.get(n, 1) > 1]
    return SimpleNamespace(
        _sizes={n: sizes.get(n, 1) for n in MESH_ORDER},
        active_mesh_dims=active,
        dp_replicate_enabled=sizes.get("dp_replicate", 1) > 1,
        dp_shard_enabled=sizes.get("dp_shard", 1) > 1,
        ep_enabled=sizes.get("ep", 1) > 1,
        cp_enabled=sizes.get("cp", 1) > 1,
        tp_enabled=sizes.get("tp", 1) > 1,
        sp_enabled=sizes.get("sp", 1) > 1,
    )


def test_ep_sits_inside_the_data_and_context_axes():
    names, shape = _patched_get_mesh(
        _config(dp_replicate=2, dp_shard=2, cp=2, ep=2, tp=2)
    )
    assert names == ("dp_replicate", "dp_shard", "cp", "ep", "tp")
    assert shape == (2, 2, 2, 2, 2)


def test_flattened_dim_names_follow_the_mesh_order():
    cfg = _config(dp_replicate=2, dp_shard=2, cp=2, ep=2)
    cfg.dp_dim_names = _patched_dp_dim_names(cfg)
    assert cfg.dp_dim_names == ["dp_replicate", "dp_shard", "ep"]
    assert _patched_dp_shard_cp_dim_names(cfg) == ["dp_shard", "cp", "ep"]
    assert _patched_dp_cp_dim_names(cfg) == ["dp_replicate", "dp_shard", "cp", "ep"]


@pytest.mark.parametrize(
    "shape,names,local_ep,cross_shard",
    [
        ((2, 8), ("dp_shard", "ep"), True, True),
        ((4, 4), ("dp_shard", "ep"), True, True),
        ((2, 2, 4), ("dp_shard", "cp", "ep"), True, True),
        ((2, 8), ("dp_replicate", "dp_shard"), None, False),
        ((16,), ("ep",), False, None),
    ],
)
def test_ep_groups_stay_node_local_on_two_eight_gpu_nodes(
    shape, names, local_ep, cross_shard
):
    report = mesh_placement_report(shape, names, gpus_per_node=8)
    if local_ep is not None:
        assert (report["ep"][1] == 0.0) is local_ep, report
    if cross_shard is not None:
        assert (report["dp_shard"][1] > 0.0) is cross_shard, report


def test_placement_report_matches_the_old_order_regression():
    # the previous (dp_replicate, ep, dp_shard) order put every EP pair across the node boundary
    report = mesh_placement_report((2, 8), ("ep", "dp_shard"), gpus_per_node=8)
    assert report["ep"] == (2, 1.0)


def test_data_parallel_index_ignores_cp_and_tp():
    names = ("dp_shard", "cp", "ep", "tp")
    sizes = {"dp_shard": 2, "cp": 2, "ep": 2, "tp": 2}
    seen = {}
    for ds in range(2):
        for c in range(2):
            for e in range(2):
                for t in range(2):
                    idx = data_parallel_index(names, (ds, c, e, t), sizes)
                    seen.setdefault((ds, e), set()).add(idx)
    assert all(len(v) == 1 for v in seen.values()), seen
    assert {next(iter(v)) for v in seen.values()} == set(range(4))
