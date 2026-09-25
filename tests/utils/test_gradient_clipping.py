"""CPU-shard regression coverage for FSDP2 gradient clipping."""

from types import SimpleNamespace

import pytest
import torch

from axolotl.utils import gradient_clipping


def test_clips_mixed_plain_and_local_shard_gradients_on_cpu(monkeypatch):
    class LocalShard:
        def __init__(self, local):
            self.local = local
            self.placements = (gradient_clipping.Shard(0),)

        def to_local(self):
            return self.local

    monkeypatch.setattr(gradient_clipping, "DTensor", LocalShard)
    local = torch.tensor([3.0, 4.0])
    sharded = SimpleNamespace(grad=LocalShard(local))
    plain = SimpleNamespace(grad=torch.tensor([12.0]))

    norm = gradient_clipping.clip_grad_norm_local_shards_(
        [sharded, plain], max_norm=6.5
    )

    assert norm.device.type == "cpu"
    torch.testing.assert_close(norm, torch.tensor(13.0))
    torch.testing.assert_close(local, torch.tensor([1.5, 2.0]))
    torch.testing.assert_close(plain.grad, torch.tensor([6.0]))


def test_replicated_dtensor_is_not_reduced_again(monkeypatch):
    class Replicate:
        pass

    class LocalShard:
        def __init__(self, local):
            self.local = local
            self.placements = (Replicate(),)

        def to_local(self):
            return self.local

    monkeypatch.setattr(gradient_clipping, "DTensor", LocalShard)
    monkeypatch.setattr(gradient_clipping, "Shard", type("Shard", (), {}))
    monkeypatch.setattr(gradient_clipping, "Partial", type("Partial", (), {}))
    monkeypatch.setattr(gradient_clipping.dist, "is_available", lambda: True)
    monkeypatch.setattr(gradient_clipping.dist, "is_initialized", lambda: True)

    replica = SimpleNamespace(grad=LocalShard(torch.tensor([3.0])))
    plain = SimpleNamespace(grad=torch.tensor([4.0]))

    norm = gradient_clipping.clip_grad_norm_local_shards_([replica, plain], 2.5)

    torch.testing.assert_close(norm, torch.tensor(5.0))
    torch.testing.assert_close(replica.grad.local, torch.tensor([1.5]))
    torch.testing.assert_close(plain.grad, torch.tensor([2.0]))


def test_sharded_dtensor_reduces_only_its_mesh_dimension(monkeypatch):
    class Shard:
        pass

    class Mesh:
        def get_group(self, axis):
            assert axis == 0
            return "shard-group"

    class LocalShard:
        def __init__(self, local):
            self.local = local
            self.placements = (Shard(),)
            self.device_mesh = Mesh()

        def to_local(self):
            return self.local

    calls = []
    monkeypatch.setattr(gradient_clipping, "DTensor", LocalShard)
    monkeypatch.setattr(gradient_clipping, "Shard", Shard)
    monkeypatch.setattr(gradient_clipping, "Partial", type("Partial", (), {}))
    monkeypatch.setattr(gradient_clipping.dist, "is_available", lambda: True)
    monkeypatch.setattr(gradient_clipping.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gradient_clipping.dist, "get_backend", lambda group: "gloo")

    def all_reduce(value, op, group):
        calls.append(group)
        value.mul_(2)

    monkeypatch.setattr(gradient_clipping.dist, "all_reduce", all_reduce)
    sharded = SimpleNamespace(grad=LocalShard(torch.tensor([3.0])))

    norm = gradient_clipping.clip_grad_norm_local_shards_([sharded], 3.0)

    torch.testing.assert_close(norm, torch.sqrt(torch.tensor(18.0)))
    torch.testing.assert_close(sharded.grad.local, torch.full((1,), 3.0 / 2**0.5))
    assert calls == ["shard-group"]


def test_rejects_partial_before_mutating_gradients(monkeypatch):
    class Partial:
        pass

    class LocalShard:
        placements = (Partial(),)

        def __init__(self, local):
            self.local = local

        def to_local(self):
            return self.local

    monkeypatch.setattr(gradient_clipping, "DTensor", LocalShard)
    monkeypatch.setattr(gradient_clipping, "Partial", Partial)
    local = torch.tensor([3.0])
    parameter = SimpleNamespace(grad=LocalShard(local))

    with pytest.raises(NotImplementedError, match="Partial placements"):
        gradient_clipping.clip_grad_norm_local_shards_([parameter], 1.0)

    torch.testing.assert_close(local, torch.tensor([3.0]))


def test_inf_norm_handles_an_empty_shard(monkeypatch):
    class Shard:
        pass

    class Mesh:
        def get_group(self, axis):
            assert axis == 0
            return "shard-group"

    class LocalShard:
        placements = (Shard(),)
        device_mesh = Mesh()

        def __init__(self, local):
            self.local = local

        def to_local(self):
            return self.local

    monkeypatch.setattr(gradient_clipping, "DTensor", LocalShard)
    monkeypatch.setattr(gradient_clipping, "Shard", Shard)
    monkeypatch.setattr(gradient_clipping.dist, "is_available", lambda: True)
    monkeypatch.setattr(gradient_clipping.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(gradient_clipping.dist, "get_backend", lambda group: "gloo")
    monkeypatch.setattr(
        gradient_clipping.dist, "all_reduce", lambda value, op, group: None
    )
    empty = SimpleNamespace(grad=LocalShard(torch.empty(0)))

    norm = gradient_clipping.clip_grad_norm_local_shards_([empty], 1.0, float("inf"))

    torch.testing.assert_close(norm, torch.tensor(0.0))


def test_rejects_zero_norm_type():
    parameter = SimpleNamespace(grad=torch.tensor([1.0]))

    with pytest.raises(ValueError, match="positive or inf"):
        gradient_clipping.clip_grad_norm_local_shards_([parameter], 1.0, 0)


def test_bfloat16_uses_full_precision_clip_coefficient():
    gradient = torch.tensor([1.75, -3.5], dtype=torch.bfloat16)
    parameter = SimpleNamespace(grad=gradient)
    before = gradient.float().clone()
    norm = before.norm()
    max_norm = norm / 3

    gradient_clipping.clip_grad_norm_local_shards_([parameter], max_norm.item())

    expected = (before * (max_norm / (norm + 1e-6))).to(torch.bfloat16)
    torch.testing.assert_close(gradient, expected)
