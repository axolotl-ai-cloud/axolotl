"""accelerate's FSDP2 x TP path imports ``ReplicateParallel`` from transformers, which 5.17
removed; the shim must let a module holding replicated DTensor params run on plain inputs."""

import os

import pytest
import torch
import torch.distributed as dist

from axolotl.monkeypatch.accelerate.tp import patch_accelerate_prepare_tp


@pytest.fixture
def one_rank_mesh():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29873")
    dist.init_process_group("gloo", rank=0, world_size=1)
    from torch.distributed.device_mesh import init_device_mesh

    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("tp",))
    finally:
        dist.destroy_process_group()


def test_replicated_param_module_runs_on_plain_inputs(one_rank_mesh):
    from torch.distributed.tensor import DTensor, Replicate

    patch_accelerate_prepare_tp()
    from transformers.integrations.tensor_parallel import ReplicateParallel

    norm = torch.nn.LayerNorm(4)
    x = torch.ones(2, 4) * torch.arange(4)
    ref = norm(x).sum()

    replicated = DTensor.from_local(
        norm.weight.data.clone(), device_mesh=one_rank_mesh, placements=[Replicate()]
    )
    # accelerate calls this once per parameter of the module; transformers marks every
    # module _is_hooked, so that flag must not stop the wrapper
    norm._is_hooked = True
    ReplicateParallel().prepare_module_tp(norm, one_rank_mesh)
    ReplicateParallel().prepare_module_tp(norm, one_rank_mesh)
    norm.weight = torch.nn.Parameter(replicated)

    out = norm(x).sum()
    assert not isinstance(out, DTensor)
    assert torch.allclose(out, ref)
    out.backward()
    assert isinstance(norm.weight, DTensor), "param must be restored after forward"
    assert isinstance(norm.weight.grad, DTensor)


def test_patch_is_idempotent_and_skips_existing_class():
    import transformers.integrations.tensor_parallel as compat

    patch_accelerate_prepare_tp()
    installed = compat.ReplicateParallel
    assert patch_accelerate_prepare_tp() is False
    assert compat.ReplicateParallel is installed


def test_replicated_grad_all_reduce_accepts_dtensor_grads(one_rank_mesh):
    from torch.distributed.tensor import DTensor, Replicate
    from transformers.distributed.tensor_parallel import ReplicatedWithGradAllReduce

    patch_accelerate_prepare_tp()
    from transformers.integrations.tensor_parallel import ReplicateParallel

    x = (torch.ones(2, 4) * torch.arange(4)).requires_grad_()
    plain = torch.nn.LayerNorm(4)
    plain(x).sum().backward()

    norm = torch.nn.LayerNorm(4)
    ReplicatedWithGradAllReduce().install_forward(norm, one_rank_mesh)
    ReplicateParallel().prepare_module_tp(norm, one_rank_mesh)
    norm.weight = torch.nn.Parameter(
        DTensor.from_local(
            norm.weight.data.clone(),
            device_mesh=one_rank_mesh,
            placements=[Replicate()],
        )
    )
    # the hook must reduce the local shard in place instead of dispatching c10d on a DTensor
    norm(x).sum().backward()
    assert isinstance(norm.weight.grad, DTensor)
    assert torch.allclose(norm.weight.grad.to_local(), plain.weight.grad)


def test_replicated_grad_all_reduce_sums_each_backward_once(one_rank_mesh, monkeypatch):
    import torch.distributed as dist
    from transformers.distributed.tensor_parallel import ReplicatedWithGradAllReduce

    patch_accelerate_prepare_tp()

    # stand in for a 2-rank sum of identical partials
    def fake_all_reduce(tensor, *args, **kwargs):
        tensor.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    x = (torch.ones(2, 4) * torch.arange(4)).requires_grad_()
    plain = torch.nn.LayerNorm(4)
    plain(x).sum().backward()

    norm = torch.nn.LayerNorm(4)
    ReplicatedWithGradAllReduce().install_forward(norm, one_rank_mesh)
    for _ in range(2):
        norm(x).sum().backward()
    # accumulated over two micro-steps: 2 * (2 * g), not 2 * (2 * g + g)
    assert torch.allclose(norm.weight.grad, 4 * plain.weight.grad)


def test_replicate_plain_params_for_tp(one_rank_mesh):
    from torch.distributed.tensor import DTensor

    from axolotl.monkeypatch.accelerate.tp import replicate_plain_params_for_tp

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    x = torch.ones(2, 4)
    ref = model(x)
    assert replicate_plain_params_for_tp(model, one_rank_mesh) == 4
    assert all(isinstance(p, DTensor) for p in model.parameters())
    out = model(x)
    assert not isinstance(out, DTensor)
    assert torch.allclose(out, ref)
    out.sum().backward()
    # a foreach optimizer step must see a uniform parameter list
    torch.optim.SGD(model.parameters(), lr=0.1, foreach=True).step()


def test_num_items_in_batch_is_not_divided_by_tp_size(monkeypatch):
    from types import SimpleNamespace

    import transformers

    from axolotl.core.trainers.base import AxolotlTrainer

    # transformers already divided the true count (10) by tp_size; patch the class
    # `super()` resolves to (the session may hold more than one Trainer class object)
    base = next(
        c for c in AxolotlTrainer.__mro__[1:] if "_get_num_items_in_batch" in vars(c)
    )
    assert issubclass(base, transformers.Trainer) or base.__name__ == "Trainer"
    monkeypatch.setattr(
        base,
        "_get_num_items_in_batch",
        lambda self, batch_samples, device: torch.tensor(5),
    )

    def trainer(tp_size, average):
        t = AxolotlTrainer.__new__(AxolotlTrainer)
        t.accelerator = SimpleNamespace(
            parallelism_config=SimpleNamespace(tp_size=tp_size)
        )
        t.args = SimpleNamespace(average_tokens_across_devices=average)
        return t

    assert int(trainer(2, False)._get_num_items_in_batch([], "cpu")) == 10
    assert int(trainer(1, False)._get_num_items_in_batch([], "cpu")) == 5
    assert int(trainer(2, True)._get_num_items_in_batch([], "cpu")) == 5
