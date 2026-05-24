"""Mode C DDP bypass: ``_maybe_bypass_ddp_for_mode_c`` must flip ``distributed_type=NO``.

ProTrain Mode C (ZeRO-3 sharded CPU offload) on multi-rank non-NVLink rigs is
unreachable when Accelerate wraps the model in DDP, because DDP's bucketed
all-reduce double-syncs gradients on top of ProTrain's per-chunk
``reduce_scatter``. The plugin runs ``_maybe_bypass_ddp_for_mode_c`` in
``post_trainer_create`` to flip ``accelerator.state.distributed_type`` to
``DistributedType.NO`` BEFORE ``accelerator.prepare()`` runs, leaving
cross-rank grad sync to ProTrain's own collectives. These tests assert the
bypass fires exactly when it should and not otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axolotl.integrations.protrain.plugin import _maybe_bypass_ddp_for_mode_c


def _fake_accelerator(distributed_type):
    """Build the minimal accelerator-state shape the bypass touches."""
    state = SimpleNamespace(
        distributed_type=distributed_type,
        _shared_state={"distributed_type": distributed_type},
    )
    return SimpleNamespace(state=state)


def _fake_wrapped(zero3_shard: bool):
    chunk_manager = SimpleNamespace(zero3_shard=zero3_shard)
    return SimpleNamespace(chunk_manager=chunk_manager)


def _patch_dist(*, world_size: int = 4, initialized: bool = True):
    """Patch ``torch.distributed`` so the helper sees a configured PG."""
    import torch.distributed as dist

    return [
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=initialized),
        patch.object(dist, "get_world_size", return_value=world_size),
    ]


def _enter_all(stack):
    for p in stack:
        p.start()


def _exit_all(stack):
    for p in stack:
        p.stop()


def test_bypass_fires_for_zero3_shard_plus_multirank() -> None:
    """Mode C + world_size>1 -> distributed_type flipped to NO."""
    from accelerate.utils import DistributedType

    accelerator = _fake_accelerator(DistributedType.MULTI_GPU)
    trainer = SimpleNamespace(accelerator=accelerator)
    wrapped = _fake_wrapped(zero3_shard=True)

    patches = _patch_dist(world_size=4, initialized=True)
    _enter_all(patches)
    try:
        # PartialState imported lazily inside the helper; patch where it's looked up.
        with patch("accelerate.PartialState") as mock_ps:
            instance = SimpleNamespace(distributed_type=DistributedType.MULTI_GPU)
            mock_ps.return_value = instance
            fired = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)
    finally:
        _exit_all(patches)

    assert fired is True
    assert accelerator.state.distributed_type == DistributedType.NO
    assert accelerator.state._shared_state["distributed_type"] == DistributedType.NO
    assert instance.distributed_type == DistributedType.NO


def test_bypass_does_not_fire_for_single_rank() -> None:
    """Mode C + world_size=1 -> no DDP would be installed, override is a no-op."""
    from accelerate.utils import DistributedType

    accelerator = _fake_accelerator(DistributedType.NO)
    trainer = SimpleNamespace(accelerator=accelerator)
    wrapped = _fake_wrapped(zero3_shard=True)

    patches = _patch_dist(world_size=1, initialized=True)
    _enter_all(patches)
    try:
        fired = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)
    finally:
        _exit_all(patches)

    assert fired is False
    assert accelerator.state.distributed_type == DistributedType.NO


def test_bypass_does_not_fire_when_zero3_shard_false() -> None:
    """Modes A / B (zero3_shard=False) need DDP for replicated sync — must not bypass."""
    from accelerate.utils import DistributedType

    accelerator = _fake_accelerator(DistributedType.MULTI_GPU)
    trainer = SimpleNamespace(accelerator=accelerator)
    wrapped = _fake_wrapped(zero3_shard=False)

    patches = _patch_dist(world_size=4, initialized=True)
    _enter_all(patches)
    try:
        fired = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)
    finally:
        _exit_all(patches)

    assert fired is False
    assert accelerator.state.distributed_type == DistributedType.MULTI_GPU


def test_bypass_does_not_fire_when_dist_not_initialised() -> None:
    """Without an initialised PG ``get_world_size`` is undefined — must short-circuit."""
    from accelerate.utils import DistributedType

    accelerator = _fake_accelerator(DistributedType.MULTI_GPU)
    trainer = SimpleNamespace(accelerator=accelerator)
    wrapped = _fake_wrapped(zero3_shard=True)

    patches = _patch_dist(world_size=4, initialized=False)
    _enter_all(patches)
    try:
        fired = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)
    finally:
        _exit_all(patches)

    assert fired is False
    assert accelerator.state.distributed_type == DistributedType.MULTI_GPU


def test_bypass_is_idempotent_when_already_NO() -> None:
    """If a prior call (or test harness) already pinned NO, the bypass is a no-op."""
    from accelerate.utils import DistributedType

    accelerator = _fake_accelerator(DistributedType.NO)
    trainer = SimpleNamespace(accelerator=accelerator)
    wrapped = _fake_wrapped(zero3_shard=True)

    patches = _patch_dist(world_size=4, initialized=True)
    _enter_all(patches)
    try:
        fired = _maybe_bypass_ddp_for_mode_c(trainer, wrapped)
    finally:
        _exit_all(patches)

    assert fired is False
    assert accelerator.state.distributed_type == DistributedType.NO


def test_validator_does_not_raise_on_modec_multirank() -> None:
    """``_reject_ddp_with_zero3_shard`` is now an info-log; the bypass owns the fix."""
    import os

    from axolotl.integrations.protrain.args import ProTrainArgs

    cfg = {
        "protrain_auto_memory": True,
        "plugins": ["axolotl.integrations.protrain.ProTrainPlugin"],
        "base_model": "HuggingFaceTB/SmolLM2-135M",
        "protrain_auto_mode": False,
        "protrain_zero3_shard": True,
    }
    prior = os.environ.get("WORLD_SIZE")
    os.environ["WORLD_SIZE"] = "4"
    try:
        # Must not raise; the bypass mixin will fire at post_trainer_create.
        ProTrainArgs.model_validate(cfg)
    finally:
        if prior is None:
            os.environ.pop("WORLD_SIZE", None)
        else:
            os.environ["WORLD_SIZE"] = prior


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
