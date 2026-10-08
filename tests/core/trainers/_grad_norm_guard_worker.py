"""Gloo workers for per-tensor ratio clipping under FSDP2, HSDP/TP and EP, and its resume.

Run under ``torch.distributed.run`` with the mode as the only argument; each mode prints
``GRAD_NORM_GUARD_<MODE>_PASS`` on rank 0.
"""

import math
import os
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import (
    DTensor,
    Partial,
    Replicate,
    Shard,
    distribute_tensor,
)

from axolotl.core.trainers.mixins.grad_norm_guard import (
    GRAD_NORM_EMA_KEY,
    GradNormGuardMixin,
)

RATIO, BETA, STEPS, SPIKE_STEP = 1.5, 0.8, 6, 4


class _Base:
    def _get_grad_norm(self, model, grad_norm=None):
        return grad_norm

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.loader(checkpoint)


class _Trainer(GradNormGuardMixin, _Base):
    def __init__(self, optimizer=None, *, ep_mesh=None):
        self.args = SimpleNamespace(
            step_outlier_grad_norm_zscore=None,
            step_outlier_loss_zscore=None,
            grad_clip_norm_ratio=RATIO,
            grad_clip_norm_ratio_beta=BETA,
        )
        self.state = SimpleNamespace(global_step=0)
        self.optimizer = optimizer
        self._ep_mesh = ep_mesh

    def _expert_parallel_enabled(self):
        return self._ep_mesh is not None

    def _global_mesh(self):
        return self._ep_mesh


class _Reference:
    """Single-process OLMo ratio clipping on whole tensors, keyed by name."""

    def __init__(self):
        self.ema: dict[str, torch.Tensor] = {}

    def step(self, grads: dict[str, torch.Tensor]):
        coefs, out = {}, {}
        for name, grad in grads.items():
            norm = grad.double().norm()
            ema = self.ema.get(name, norm)
            coef = torch.clamp(RATIO * ema / (norm + 1e-6), max=1.0)
            out[name] = grad.double() * coef
            self.ema[name] = ema + (norm * coef - ema) * (1 - BETA)
            coefs[name] = coef
        return out, coefs


def _full_grads(shapes: dict[str, tuple], step: int, spike: str | None = None):
    generator = torch.Generator().manual_seed(1000 + step)
    grads = {
        name: torch.randn(shape, generator=generator) * (1.0 + 0.1 * index)
        for index, (name, shape) in enumerate(shapes.items())
    }
    if step == SPIKE_STEP and spike is not None:
        grads[spike] = grads[spike] * 50.0
    return grads


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _full(tensor):
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _close(actual, expected, what):
    torch.testing.assert_close(
        actual.double(), expected.double(), rtol=1e-5, atol=1e-6, msg=what
    )


def _averages(trainer, params):
    return trainer._grad_clip_averages(params)


def _accelerated(optimizer):
    from accelerate import Accelerator

    return Accelerator(cpu=True).prepare_optimizer(optimizer)


def _fsdp_model(mesh):
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    # uneven rows, and a 1-row bias that leaves some ranks with an empty shard
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5), torch.nn.Linear(5, 3), torch.nn.Linear(3, 1)
    )
    for layer in model:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def _set_grads(named, grads):
    for name, param in named.items():
        param.grad = distribute_tensor(
            grads[name], param.device_mesh, param.placements, src_data_rank=None
        )


def fsdp2():
    world = dist.get_world_size()
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp_shard",))
    model = _fsdp_model(mesh)
    named = dict(model.named_parameters())
    shapes = {name: tuple(param.shape) for name, param in named.items()}
    optimizer = _accelerated(torch.optim.AdamW(model.parameters(), lr=1e-3))
    trainer = _Trainer(optimizer)
    reference = _Reference()
    for step in range(STEPS):
        grads = _full_grads(shapes, step, spike="1.weight")
        _set_grads(named, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        expected, coefs = reference.step(grads)
        for name, param in named.items():
            _close(_full(param.grad), expected[name], f"step {step} {name} grad")
        _close(
            _averages(trainer, list(named.values())),
            torch.stack([reference.ema[name] for name in named]),
            f"step {step} averages",
        )
        if step == SPIKE_STEP:
            assert coefs["1.weight"] < 0.1, coefs
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # after the first optimizer step the averages live in the optimizer state
        for param in named.values():
            if step > 0:
                value = optimizer.optimizer.state[param][GRAD_NORM_EMA_KEY]
                assert value.dtype == torch.float32 and value.dim() == 0


def hsdp_tp():
    mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp_shard", "tp"))
    layouts = {
        "fsdp_tp_colwise": (Shard(0), Shard(0)),
        "fsdp_tp_rowwise": (Shard(0), Shard(1)),
        "hsdp": (Replicate(), Shard(0)),
        "tp_only": (Replicate(), Shard(1)),
        "fsdp_tp_replicated": (Shard(0), Replicate()),
        "replicated": (Replicate(), Replicate()),
    }
    shapes = {name: (5, 3) for name in layouts}
    shapes["plain"] = (2,)
    params = {}
    for name, placements in layouts.items():
        params[name] = torch.nn.Parameter(
            distribute_tensor(torch.zeros(shapes[name]), mesh, placements)
        )
    params["plain"] = torch.nn.Parameter(torch.zeros(2))
    model = torch.nn.Module()
    for name, param in params.items():
        model.register_parameter(name, param)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = _Trainer(optimizer)
    reference = _Reference()
    for step in range(STEPS):
        grads = _full_grads(shapes, step, spike="fsdp_tp_rowwise")
        for name, param in params.items():
            param.grad = (
                distribute_tensor(
                    grads[name], mesh, param.placements, src_data_rank=None
                )
                if isinstance(param, DTensor)
                else grads[name].clone()
            )
        trainer._get_grad_norm(model, torch.tensor(1.0))
        expected, _ = reference.step(grads)
        for name, param in params.items():
            _close(_full(param.grad), expected[name], f"step {step} {name} grad")
        _close(
            _averages(trainer, list(params.values())),
            torch.stack([reference.ema[name] for name in params]),
            f"step {step} averages",
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    partial = torch.nn.Parameter(
        distribute_tensor(torch.zeros(2), mesh, (Replicate(), Replicate()))
    )
    partial.grad = DTensor.from_local(
        torch.ones(2), mesh, (Replicate(), Partial()), run_check=False
    )
    holder = torch.nn.Module()
    holder.p = partial
    try:
        trainer._get_grad_norm(holder, torch.tensor(1.0))
    except NotImplementedError as error:
        assert "Partial" in str(error)
    else:
        raise AssertionError("Partial gradients must be rejected")


def _dtensor(local, mesh, placements, shape):
    stride = torch.empty(shape).stride()
    return DTensor.from_local(
        local, mesh, placements, run_check=False, shape=torch.Size(shape), stride=stride
    )


def ep():
    rank = dist.get_rank()
    mesh = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("ep", "dp"))
    ep_rank, dp_rank = mesh.get_coordinate()
    dp_mesh = mesh["dp"]
    all_mesh = mesh._flatten("all")
    experts_global, experts_local = 4, 2

    experts = torch.nn.Module()
    experts.num_local_experts = experts_local
    experts.num_experts_global = experts_global
    expert_shape = (experts_local, 3, 2)
    experts.weight = torch.nn.Parameter(
        _dtensor(torch.zeros(1, 3, 2), dp_mesh, (Shard(0),), expert_shape)
    )
    lora = torch.nn.Module()
    lora._ep_lora_sharded = True
    lora.weight = torch.nn.Parameter(torch.zeros(experts_local, 2))
    model = torch.nn.Module()
    model.experts = experts
    model.lora = lora
    model.dense = torch.nn.Parameter(
        _dtensor(torch.zeros(2, 3), all_mesh, (Shard(0),), (8, 3))
    )
    model.replica = torch.nn.Parameter(
        _dtensor(torch.zeros(3), all_mesh, (Replicate(),), (3,))
    )
    model.norm = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    trainer = _Trainer(optimizer, ep_mesh=mesh)

    # whole-model gradients: experts (4, 3, 2) and lora (4, 2) cover every expert
    shapes = {
        "experts": (experts_global, 3, 2),
        "lora": (experts_global, 2),
        "dense": (8, 3),
        "replica": (3,),
        "norm": (3,),
    }
    mine = slice(ep_rank * experts_local, (ep_rank + 1) * experts_local)
    # this rank's EP slice is one tensor; the reference sees per-slice tensors
    reference = _Reference()
    named = {
        "experts": experts.weight,
        "lora": lora.weight,
        "dense": model.dense,
        "replica": model.replica,
        "norm": model.norm,
    }
    for step in range(STEPS):
        grads = _full_grads(shapes, step)
        if step == SPIKE_STEP:
            # spike only the second EP rank's experts
            grads["experts"][experts_local:] *= 50.0
        experts.weight.grad = _dtensor(
            grads["experts"][mine][dp_rank : dp_rank + 1].clone(),
            dp_mesh,
            (Shard(0),),
            expert_shape,
        )
        lora.weight.grad = grads["lora"][mine].clone()
        model.dense.grad = _dtensor(
            grads["dense"][2 * rank : 2 * rank + 2].clone(),
            all_mesh,
            (Shard(0),),
            (8, 3),
        )
        model.replica.grad = _dtensor(
            grads["replica"].clone(), all_mesh, (Replicate(),), (3,)
        )
        model.norm.grad = grads["norm"].clone()
        trainer._get_grad_norm(model, torch.tensor(1.0))

        sliced = dict(grads)
        sliced["experts"] = grads["experts"][mine]
        sliced["lora"] = grads["lora"][mine]
        expected, coefs = reference.step(sliced)
        for name, param in named.items():
            _close(_full(param.grad), expected[name], f"step {step} {name} grad")
        _close(
            _averages(trainer, list(named.values())),
            torch.stack([reference.ema[name] for name in named]),
            f"step {step} averages",
        )
        if step == SPIKE_STEP:
            # no cross-expert mixing: only the spiking EP rank clips its experts
            assert (coefs["experts"] < 1.0) == (ep_rank == 1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(
        gathered, float(optimizer.state[experts.weight][GRAD_NORM_EMA_KEY])
    )
    assert gathered[0] == gathered[1] and gathered[2] == gathered[3]
    assert gathered[0] != gathered[2], gathered

    # an EP tensor sharded over a mesh axis that spans ep would mix experts
    bad = torch.nn.Module()
    bad.num_local_experts, bad.num_experts_global = 1, 4
    bad.weight = torch.nn.Parameter(
        _dtensor(torch.zeros(1), all_mesh, (Shard(0),), (4,))
    )
    bad.weight.grad = _dtensor(torch.ones(1), all_mesh, (Shard(0),), (4,))
    bad_trainer = _Trainer(torch.optim.AdamW(bad.parameters()), ep_mesh=mesh)
    try:
        bad_trainer._get_grad_norm(bad, torch.tensor(1.0))
    except NotImplementedError as error:
        assert "mix experts" in str(error)
    else:
        raise AssertionError("an EP tensor sharded across ep must be rejected")


def _fsdp_plugin(full: bool):
    from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

    return SimpleNamespace(
        fsdp_version=2,
        state_dict_type=StateDictType.FULL_STATE_DICT
        if full
        else StateDictType.SHARDED_STATE_DICT,
        state_dict_config=SimpleNamespace(offload_to_cpu=False, rank0_only=False),
        optim_state_dict_config=SimpleNamespace(rank0_only=False),
    )


def _resume_run(mesh, plugin, directory, *, ratio_before=True):
    from accelerate.utils.fsdp_utils import load_fsdp_optimizer, save_fsdp_optimizer

    accelerator = SimpleNamespace(
        process_index=dist.get_rank(), wait_for_everyone=dist.barrier
    )
    model = _fsdp_model(mesh)
    named = dict(model.named_parameters())
    shapes = {name: tuple(param.shape) for name, param in named.items()}
    optimizer = _accelerated(torch.optim.AdamW(model.parameters(), lr=1e-2))
    trainer = _Trainer(optimizer)
    if not ratio_before:
        trainer.args.grad_clip_norm_ratio = None
    half = STEPS // 2
    for step in range(half):
        _set_grads(named, _full_grads(shapes, step, spike="0.weight"))
        trainer._get_grad_norm(model, torch.tensor(1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    save_fsdp_optimizer(plugin, accelerator, optimizer, model, directory)
    dist.barrier()

    resumed_model = _fsdp_model(mesh)
    resumed_model.load_state_dict(model.state_dict())
    resumed_named = dict(resumed_model.named_parameters())
    resumed_optimizer = _accelerated(
        torch.optim.AdamW(resumed_model.parameters(), lr=1e-2)
    )
    resumed = _Trainer(resumed_optimizer)
    resumed.loader = lambda checkpoint: load_fsdp_optimizer(
        plugin, accelerator, resumed_optimizer, resumed_model, checkpoint
    )
    resumed._load_optimizer_and_scheduler(directory)
    if ratio_before:
        _close(
            _averages(resumed, list(resumed_named.values())),
            _averages(trainer, list(named.values())),
            "restored averages",
        )
    else:
        assert torch.isnan(_averages(resumed, list(resumed_named.values()))).all()
        trainer.args.grad_clip_norm_ratio = RATIO

    # the original and the resumed run now make identical decisions
    for step in range(half, STEPS):
        grads = _full_grads(shapes, step, spike="0.weight")
        _set_grads(named, grads)
        _set_grads(resumed_named, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        resumed._get_grad_norm(resumed_model, torch.tensor(1.0))
        for name in named:
            _close(
                _full(resumed_named[name].grad),
                _full(named[name].grad),
                f"step {step} {name}",
            )
        _close(
            _averages(resumed, list(resumed_named.values())),
            _averages(trainer, list(named.values())),
            f"step {step} averages",
        )
        assert int(resumed._grad_clip_last_clipped) == int(
            trainer._grad_clip_last_clipped
        )
        if ratio_before and step == SPIKE_STEP:
            assert int(trainer._grad_clip_last_clipped) == 1
        for opt in (optimizer, resumed_optimizer):
            opt.step()
            opt.zero_grad(set_to_none=True)
    for param in resumed_named.values():
        value = resumed_optimizer.optimizer.state[param][GRAD_NORM_EMA_KEY]
        assert value.dtype == torch.float32 and value.dim() == 0


def resume():
    mesh = init_device_mesh(
        "cpu", (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    for full in (False, True):
        directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(directory)
        _resume_run(mesh, _fsdp_plugin(full), directory[0])
    # a sharded checkpoint written before ratio clipping was enabled still loads
    directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(directory)
    _resume_run(mesh, _fsdp_plugin(False), directory[0], ratio_before=False)


MODES = {"fsdp2": fsdp2, "hsdp_tp": hsdp_tp, "ep": ep, "resume": resume}


def main():
    mode = sys.argv[1]
    os.environ.setdefault("ACCELERATE_USE_CPU", "true")
    dist.init_process_group("gloo")
    torch.manual_seed(0)
    MODES[mode]()
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"GRAD_NORM_GUARD_{mode.upper()}_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    assert not math.isnan(RATIO)
    main()
