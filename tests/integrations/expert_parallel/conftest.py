"""Fixtures shared by the expert-parallel integration tests."""

import pytest


@pytest.fixture
def fake_ep_sharder(monkeypatch):
    """Run the real ``shard_expert_weights`` as one rank of a faked two-rank EP group.

    The collective scatter is replaced by a CPU slice; everything else the sharder
    does to the module (counts, offsets, flags, DDP ignores) is the production code.
    """
    from axolotl.integrations.expert_parallel import shard

    def run(model, rank, world_size=2):
        monkeypatch.setattr(shard.dist, "get_world_size", lambda group=None: world_size)
        monkeypatch.setattr(shard.dist, "get_rank", lambda group=None: rank)
        monkeypatch.setattr(
            shard.dist,
            "all_gather_object",
            lambda ranks, value: ranks.__setitem__(
                slice(None), list(range(world_size))
            ),
        )

        def scatter_on_cpu(module, name, count, ranks):
            shard._replace_with_slice(
                module, name, ranks[rank] * count, (ranks[rank] + 1) * count
            )

        monkeypatch.setattr(shard, "_scatter_expert_from_rank0", scatter_on_cpu)
        return shard.shard_expert_weights(model, object())

    return run
