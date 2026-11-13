"""Mesh slice: config degrees -> MeshSpec under the dense-degree convention."""

from __future__ import annotations

import dataclasses
import typing

import pytest

from axolotl.core.plan.errors import PlanError
from axolotl.core.plan.slices.mesh import READS, resolve_mesh
from axolotl.core.plan.specs import MeshSpec

FSDP = {"fsdp_config": {"fsdp_version": 2}}
DEEPSPEED = {"deepspeed": "deepspeed_configs/zero3.json"}


def _layout(mesh: MeshSpec) -> tuple[int, ...]:
    return (mesh.dp_replicate, mesh.dp_shard, mesh.cp, mesh.tp, mesh.ep, mesh.efsdp)


def _error(config, world_size) -> PlanError:
    with pytest.raises(PlanError) as info:
        resolve_mesh(config, world_size=world_size)
    assert info.value.code == "PLAN-001"
    return info.value


class TestCanonicalExpertParallelRows:
    def test_8_ranks_ep8_unset(self):
        mesh = resolve_mesh({**FSDP, "expert_parallel_size": 8}, world_size=8)
        assert _layout(mesh) == (1, 8, 1, 1, 8, 1)
        assert mesh.expert_placement == "ep_within_dp_shard"
        assert not mesh.legacy_spelling

    def test_8_ranks_ep8_dense_spelling(self):
        mesh = resolve_mesh(
            {**FSDP, "expert_parallel_size": 8, "dp_shard_size": 8}, world_size=8
        )
        assert _layout(mesh) == (1, 8, 1, 1, 8, 1)
        assert not mesh.legacy_spelling

    def test_16_ranks_ep8_unset(self):
        mesh = resolve_mesh({**FSDP, "expert_parallel_size": 8}, world_size=16)
        assert _layout(mesh) == (1, 16, 1, 1, 8, 2)
        assert not mesh.legacy_spelling

    def test_16_ranks_ep8_legacy_spelling(self):
        mesh = resolve_mesh(
            {**FSDP, "expert_parallel_size": 8, "dp_shard_size": 2}, world_size=16
        )
        assert _layout(mesh) == (1, 16, 1, 1, 8, 2)
        assert mesh.legacy_spelling

    def test_16_ranks_ep8_fits_neither_spelling(self):
        err = _error({**FSDP, "expert_parallel_size": 8, "dp_shard_size": 8}, 16)
        assert err.key == "dp_shard_size"
        assert err.value == 8
        assert "dp_shard_size: 16 or leave it unset" in err.hint
        assert "dp_replicate_size: 2" in err.hint

    def test_16_ranks_ep8_dense_spelling(self):
        mesh = resolve_mesh(
            {**FSDP, "expert_parallel_size": 8, "dp_shard_size": 16}, world_size=16
        )
        assert _layout(mesh) == (1, 16, 1, 1, 8, 2)
        assert not mesh.legacy_spelling


class TestSpellings:
    @pytest.mark.parametrize(
        "world_size,ep,legacy,dense",
        [(4, 2, 2, 4), (8, 2, 4, 8), (16, 4, 4, 16), (32, 8, 4, 32)],
    )
    def test_both_spellings_resolve_to_one_spec(self, world_size, ep, legacy, dense):
        """Only legacy_spelling tells the two apart, so the plan hash can ignore it later."""
        base = {**FSDP, "expert_parallel_size": ep}
        old = resolve_mesh({**base, "dp_shard_size": legacy}, world_size=world_size)
        new = resolve_mesh({**base, "dp_shard_size": dense}, world_size=world_size)
        unset = resolve_mesh(base, world_size=world_size)
        assert old.legacy_spelling and not new.legacy_spelling
        assert dataclasses.replace(old, legacy_spelling=False) == new == unset

    def test_ep_dp_shard_size_one_is_the_legacy_spelling(self):
        mesh = resolve_mesh(
            {**FSDP, "expert_parallel_size": 8, "dp_shard_size": 1}, world_size=8
        )
        assert _layout(mesh) == (1, 8, 1, 1, 8, 1)
        assert mesh.legacy_spelling

    def test_without_ep_dp_shard_size_one_means_unset(self):
        mesh = resolve_mesh({**FSDP, "dp_shard_size": 1}, world_size=8)
        assert _layout(mesh) == (1, 8, 1, 1, 1, 8)

    def test_ep_must_divide_the_dense_degree(self):
        err = _error(
            {
                **FSDP,
                "expert_parallel_size": 4,
                "dp_shard_size": 2,
                "dp_replicate_size": 2,
            },
            4,
        )
        assert err.key == "expert_parallel_size"

    def test_ep_does_not_borrow_context_parallel_ranks(self):
        """Experts stay replicated across cp; ep_within_dp_shard_cp_tp is unreachable for now."""
        mesh = resolve_mesh(
            {**FSDP, "expert_parallel_size": 2, "context_parallel_size": 2},
            world_size=8,
        )
        assert _layout(mesh) == (1, 4, 2, 1, 2, 2)
        assert mesh.expert_placement == "ep_within_dp_shard"
        _error({**FSDP, "expert_parallel_size": 4, "context_parallel_size": 4}, 8)

    def test_ep_requires_fsdp(self):
        err = _error({"expert_parallel_size": 2}, 2)
        assert err.allowed == ("fsdp2",)


class TestLayouts:
    def test_single_process(self):
        mesh = resolve_mesh({}, world_size=1)
        assert mesh.dist == "single"
        assert _layout(mesh) == (1, 1, 1, 1, 1, 1)

    def test_ddp_replicates(self):
        mesh = resolve_mesh({}, world_size=4)
        assert mesh.dist == "ddp"
        assert _layout(mesh) == (4, 1, 1, 1, 1, 1)

    def test_fsdp_fills_dp_shard(self):
        mesh = resolve_mesh(FSDP, world_size=8)
        assert mesh.dist == "fsdp2"
        assert _layout(mesh) == (1, 8, 1, 1, 1, 8)

    def test_hsdp(self):
        mesh = resolve_mesh(
            {**FSDP, "dp_replicate_size": 2, "dp_shard_size": 4}, world_size=8
        )
        assert _layout(mesh) == (2, 4, 1, 1, 1, 4)

    def test_hsdp_fills_dp_shard(self):
        mesh = resolve_mesh({**FSDP, "dp_replicate_size": 2}, world_size=8)
        assert _layout(mesh) == (2, 4, 1, 1, 1, 4)

    def test_dp_replicate_is_never_filled(self):
        err = _error({**FSDP, "dp_shard_size": 4}, 8)
        assert "dp_shard_size: 8 or leave it unset" in err.hint
        assert "dp_replicate_size: 2" in err.hint

    def test_context_parallel_under_fsdp_is_ring(self):
        mesh = resolve_mesh({**FSDP, "context_parallel_size": 2}, world_size=8)
        assert mesh.cp_impl == "ring"
        assert _layout(mesh) == (1, 4, 2, 1, 1, 4)

    def test_context_parallel_under_deepspeed_is_ulysses_and_replicates(self):
        mesh = resolve_mesh({**DEEPSPEED, "context_parallel_size": 2}, world_size=8)
        assert mesh.dist == "deepspeed"
        assert mesh.cp_impl == "ulysses"
        assert _layout(mesh) == (4, 1, 2, 1, 1, 1)

    def test_deepspeed_rejects_dp_shard_size(self):
        err = _error({**DEEPSPEED, "context_parallel_size": 2, "dp_shard_size": 4}, 8)
        assert err.key == "dp_shard_size"

    def test_deepspeed_rejects_inconsistent_replicate(self):
        err = _error(
            {**DEEPSPEED, "context_parallel_size": 2, "dp_replicate_size": 2}, 8
        )
        assert err.key == "dp_replicate_size"

    def test_tensor_parallel(self):
        mesh = resolve_mesh(
            {**FSDP, "tensor_parallel_size": 2, "dp_shard_size": 4}, world_size=8
        )
        assert _layout(mesh) == (1, 4, 1, 2, 1, 4)

    def test_fsdp_tp_cp(self):
        mesh = resolve_mesh(
            {
                **FSDP,
                "tensor_parallel_size": 2,
                "context_parallel_size": 2,
                "dp_shard_size": 2,
            },
            world_size=8,
        )
        assert _layout(mesh) == (1, 2, 2, 2, 1, 2)

    @pytest.mark.parametrize(
        "config",
        [
            {"tensor_parallel_size": 2},
            {**FSDP, "tensor_parallel_size": 2, "dp_replicate_size": 4},
            {**DEEPSPEED, "tensor_parallel_size": 2, "context_parallel_size": 2},
        ],
    )
    def test_tp_beside_pure_replicas_is_rejected(self, config):
        """accelerate's ParallelismConfig refuses TP/CP next to dp_replicate with dp_shard 1."""
        assert _error(config, 8).key == "dp_replicate_size"

    def test_tp_cp_must_divide_world(self):
        _error({**FSDP, "tensor_parallel_size": 4, "context_parallel_size": 4}, 8)

    @pytest.mark.parametrize("value", [0, -2, 2.0, True, "2"])
    def test_degree_must_be_positive_int(self, value):
        assert _error({**FSDP, "dp_shard_size": value}, 8).key == "dp_shard_size"


class TestSpec:
    def test_frozen_and_slotted(self):
        mesh = resolve_mesh(FSDP, world_size=2)
        assert not hasattr(mesh, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            mesh.dp_shard = 4  # type: ignore[misc]

    def test_field_names(self):
        assert [f.name for f in dataclasses.fields(MeshSpec)] == [
            "world_size",
            "world_size_source",
            "device_kind",
            "dist",
            "dp_replicate",
            "dp_shard",
            "cp",
            "cp_impl",
            "tp",
            "ep",
            "pp",
            "efsdp",
            "expert_placement",
            "legacy_spelling",
        ]
        placement = typing.get_type_hints(MeshSpec)["expert_placement"]
        assert typing.get_args(placement) == (
            "none",
            "ep_within_dp_shard",
            "ep_within_dp_shard_cp_tp",
        )

    def test_reads_only_declared_keys(self):
        seen = set()

        class Recording(dict):
            def get(self, key, default=None):
                seen.add(key)
                return super().get(key, default)

        resolve_mesh(Recording(FSDP, expert_parallel_size=2), world_size=4)
        assert seen <= READS

    def test_does_not_mutate_input(self):
        config = {**FSDP, "expert_parallel_size": 8, "dp_shard_size": 2}
        snapshot = {**config, "fsdp_config": dict(config["fsdp_config"])}
        resolve_mesh(config, world_size=16)
        assert config == snapshot
