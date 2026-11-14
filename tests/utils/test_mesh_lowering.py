"""MeshSpec -> accelerate kwargs / environment, against the derivations they replaced."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import axolotl
from axolotl.core.plan.errors import PlanError
from axolotl.core.plan.slices.mesh import resolve_mesh
from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import (
    log_mesh_spelling,
    mesh_to_accelerate_env,
    mesh_to_parallelism_kwargs,
    resolve_cfg_mesh,
)

FSDP = {"fsdp_config": {"fsdp_version": 2}}
MATRIX_SCRIPT = Path(__file__).with_name("_mesh_layout_matrix.py")


def _lower(config, world_size, **written):
    mesh = resolve_mesh(config, world_size=world_size)
    return (
        mesh_to_parallelism_kwargs(
            mesh, dp_replicate_written=written.get("dp_replicate_written", False)
        ),
        mesh_to_accelerate_env(mesh, **written),
    )


class TestCanonicalLowerings:
    def test_8_ranks_ep8_any_spelling(self):
        for dp_shard_size in (None, 1, 8):
            kwargs, env = _lower(
                {**FSDP, "expert_parallel_size": 8, "dp_shard_size": dp_shard_size},
                8,
                dp_shard_written=(dp_shard_size or 1) > 1,
            )
            assert kwargs == {}
            assert env == {"PARALLELISM_CONFIG_EP_SIZE": "8"}

    @pytest.mark.parametrize("dp_shard_size", [None, 2, 16])
    def test_16_ranks_ep8_any_spelling(self, dp_shard_size):
        kwargs, env = _lower(
            {**FSDP, "expert_parallel_size": 8, "dp_shard_size": dp_shard_size},
            16,
            dp_shard_written=dp_shard_size is not None,
        )
        assert kwargs == {"dp_shard_size": 2}
        assert env == {
            "PARALLELISM_CONFIG_EP_SIZE": "8",
            "PARALLELISM_CONFIG_DP_SHARD_SIZE": "2",
            "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
        }

    def test_pure_fsdp_gets_accelerate_config_only_when_written(self):
        assert _lower(FSDP, 8) == ({"dp_shard_size": 8}, {})
        assert _lower({**FSDP, "dp_shard_size": 8}, 8, dp_shard_written=True) == (
            {"dp_shard_size": 8},
            {
                "PARALLELISM_CONFIG_DP_SHARD_SIZE": "8",
                "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
            },
        )

    def test_ddp_axis_name_follows_spelling(self):
        assert _lower({}, 4) == ({"dp_shard_size": 4}, {})
        assert _lower({"dp_replicate_size": 4}, 4, dp_replicate_written=True) == (
            {"dp_replicate_size": 4},
            {
                "PARALLELISM_CONFIG_DP_REPLICATE_SIZE": "4",
                "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
            },
        )

    def test_hsdp(self):
        assert _lower(
            {**FSDP, "dp_replicate_size": 2, "dp_shard_size": 4},
            8,
            dp_shard_written=True,
            dp_replicate_written=True,
        ) == (
            {"dp_replicate_size": 2, "dp_shard_size": 4},
            {
                "PARALLELISM_CONFIG_DP_SHARD_SIZE": "4",
                "PARALLELISM_CONFIG_DP_REPLICATE_SIZE": "2",
                "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
            },
        )

    def test_fsdp_context_parallel_fills_dp_shard_for_accelerate(self):
        """The replaced env path left dp_shard out here, so accelerate saw 2 of 8 ranks."""
        kwargs, env = _lower({**FSDP, "context_parallel_size": 2}, 8)
        assert kwargs == {"cp_size": 2, "dp_shard_size": 4}
        assert env == {
            "PARALLELISM_CONFIG_CP_SIZE": "2",
            "ACCELERATE_ALLOW_CP_STANDALONE": "true",
            "PARALLELISM_CONFIG_DP_SHARD_SIZE": "4",
            "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
        }

    def test_deepspeed_context_parallel(self):
        kwargs, env = _lower(
            {"deepspeed": "zero3.json", "context_parallel_size": 2},
            8,
        )
        assert kwargs == {
            "sp_size": 2,
            "sp_backend": "deepspeed",
            "dp_replicate_size": 4,
        }
        assert env == {
            "PARALLELISM_CONFIG_SP_SIZE": "2",
            "PARALLELISM_CONFIG_SP_BACKEND": "deepspeed",
            "PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE": "true",
            "PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION": "sdpa",
            "PARALLELISM_CONFIG_DP_REPLICATE_SIZE": "4",
            "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
        }

    def test_tensor_parallel(self):
        kwargs, env = _lower(
            {**FSDP, "tensor_parallel_size": 2, "dp_shard_size": 4},
            8,
            dp_shard_written=True,
        )
        assert kwargs == {"tp_size": 2, "dp_shard_size": 4}
        assert env == {
            "PARALLELISM_CONFIG_TP_SIZE": "2",
            "PARALLELISM_CONFIG_DP_SHARD_SIZE": "4",
            "ACCELERATE_USE_PARALLELISM_CONFIG": "true",
        }


class TestCfgMesh:
    def test_single_process_ignores_the_layout(self):
        cfg = DictDefault(
            {**FSDP, "dp_shard_size": 8, "tensor_parallel_size": 2},
            expert_parallel_size=8,
        )
        mesh = resolve_cfg_mesh(cfg, world_size=1)
        assert (mesh.dp_shard, mesh.tp, mesh.ep, mesh.dist) == (1, 1, 1, "fsdp2")

    def test_same_inputs_same_spec(self):
        cfg = DictDefault({**FSDP, "expert_parallel_size": 8})
        mesh = resolve_cfg_mesh(cfg, world_size=16)
        assert resolve_cfg_mesh(DictDefault(cfg), world_size=16) is mesh

    def test_world_size_defaults_to_env(self, monkeypatch):
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert resolve_cfg_mesh(DictDefault(FSDP)).dp_shard == 4

    def test_inconsistent_layout_raises(self):
        with pytest.raises(PlanError):
            resolve_cfg_mesh(
                DictDefault(FSDP, expert_parallel_size=8, dp_shard_size=8),
                world_size=16,
            )

    def test_legacy_spelling_warns_with_the_rewrite(self, caplog):
        cfg = DictDefault(FSDP, expert_parallel_size=8, dp_shard_size=2)
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            log_mesh_spelling(cfg, resolve_cfg_mesh(cfg, world_size=16))
        warnings = [r.getMessage() for r in caplog.records]
        assert len(warnings) == 1
        assert "dp_shard_size: 2 with expert_parallel_size: 8" in warnings[0]
        assert "write dp_shard_size: 16" in warnings[0]

    def test_dense_spelling_does_not_warn(self, caplog):
        cfg = DictDefault(FSDP, expert_parallel_size=8, dp_shard_size=16)
        with caplog.at_level(logging.WARNING, logger="axolotl"):
            log_mesh_spelling(cfg, resolve_cfg_mesh(cfg, world_size=16))
        assert not caplog.records


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    """Rows from ``_mesh_layout_matrix.py``, run in its own interpreter so accelerate's
    ParallelismConfig patch never reaches this process."""
    out = tmp_path_factory.mktemp("mesh") / "matrix.json"
    src = str(Path(axolotl.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            filter(None, [src, os.environ.get("PYTHONPATH")])
        ),
    }
    subprocess.run(  # nosec B603
        [sys.executable, str(MATRIX_SCRIPT), str(out)],
        check=True,
        env=env,
        timeout=600,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def _row(row):
    keys = ("world", "dist", "tp", "cp", "ep", "dp_shard", "dp_replicate")
    return {key: row[key] for key in keys}


def _stray_deepspeed_degree(row):
    """DeepSpeed CP with a dp_shard_size, or a dp_replicate_size that disagrees with the
    replicas. accelerate skips its size check under DeepSpeed SP, so these ran with a
    ParallelismConfig whose data-parallel size was wrong; they now fail at config load."""
    return (
        row["dist"] == "deepspeed"
        and row["cp"] > 1
        and row["plan_key"] in ("dp_shard_size", "dp_replicate_size")
    )


def test_layouts_that_worked_lower_identically(matrix):
    mismatched = []
    for row in matrix:
        if not row["legacy_ok"]:
            continue
        if row["plan_error"]:
            if not _stray_deepspeed_degree(row):
                mismatched.append((_row(row), row["plan_error"], row["plan_hint"]))
            continue
        if (
            list(row["new_kwargs"].items()) != list(row["legacy_kwargs"].items())
            or row["new_env"] != row["legacy_env"]
        ):
            mismatched.append(
                (
                    _row(row),
                    row["legacy_kwargs"],
                    row["new_kwargs"],
                    row["legacy_env"],
                    row["new_env"],
                )
            )
    assert not mismatched, mismatched[:10]
    identical = sum(1 for r in matrix if r["legacy_ok"] and not r["plan_error"])
    assert identical > 250


def test_layouts_that_failed_now_fail_at_load_or_work(matrix):
    outcomes = Counter()
    still_failing = []
    for row in matrix:
        if row["legacy_ok"]:
            continue
        if row["plan_error"]:
            outcomes["rejected at load"] += 1
        elif row["new_error"]:
            still_failing.append((_row(row), row["new_error"]))
        else:
            outcomes["now accepted"] += 1
    assert not still_failing, still_failing[:10]
    assert outcomes["now accepted"] > 0 and outcomes["rejected at load"] > 0


def test_accelerate_accepts_every_emitted_layout(matrix):
    rejected = [
        (_row(row), row["new_error"])
        for row in matrix
        if not row["plan_error"] and row["new_error"]
    ]
    assert not rejected, rejected[:10]


def test_canonical_rows_in_matrix(matrix):
    def find(world, ep, dp_shard):
        (row,) = [
            r
            for r in matrix
            if (r["world"], r["dist"], r["tp"], r["cp"], r["ep"])
            == (world, "fsdp", 1, 1, ep)
            and r["dp_shard"] == dp_shard
            and r["dp_replicate"] is None
        ]
        return row

    assert find(8, 8, None)["legacy_ok"]
    assert not find(8, 8, 8)["legacy_ok"] and find(8, 8, 8)["new_error"] is None
    assert not find(16, 8, None)["legacy_ok"]
    assert find(16, 8, None)["mesh"]["efsdp"] == 2
    assert find(16, 8, 2)["legacy_ok"] and find(16, 8, 2)["mesh"]["legacy_spelling"]
    assert find(16, 8, 8)["plan_error"] == "PLAN-001"
    assert not find(16, 8, 16)["legacy_ok"]
    assert find(16, 8, 16)["new_env"] == find(16, 8, 2)["new_env"]
