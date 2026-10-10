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
# "cuda" runs the same checks on NCCL (tests/e2e/multigpu/solo); default CPU + gloo
DEVICE = os.environ.get("GRAD_NORM_GUARD_DEVICE", "cpu")


class _Base:
    def _get_grad_norm(self, model, grad_norm=None):
        return grad_norm

    def _load_optimizer_and_scheduler(self, checkpoint):
        self.loader(checkpoint)


class _Trainer(GradNormGuardMixin, _Base):
    def __init__(self, optimizer=None, *, ep_mesh=None, pure_ep=False):
        self.args = SimpleNamespace(
            step_outlier_grad_norm_zscore=None,
            step_outlier_loss_zscore=None,
            grad_clip_norm_ratio=RATIO,
            grad_clip_norm_ratio_beta=BETA,
        )
        self.state = SimpleNamespace(global_step=0)
        self.optimizer = optimizer
        self._ep_mesh = ep_mesh
        # pure EP: no global mesh; the experts' own root mesh carries the ep axis
        self._pure_ep = pure_ep

    def _expert_parallel_enabled(self):
        return self._ep_mesh is not None

    def _global_mesh(self):
        return None if self._pure_ep else self._ep_mesh


class _Reference:
    """Single-process OLMo ratio clipping, keyed by name; per expert for ``experts`` names.

    An expert tensor is clipped per slice along dim 0, with one average per expert.
    """

    def __init__(self, experts=()):
        self.ema: dict[str, torch.Tensor] = {}
        self.experts = set(experts)

    def step(self, grads: dict[str, torch.Tensor]):
        coefs, out = {}, {}
        for name, grad in grads.items():
            grad = grad.double()
            if name in self.experts:
                norm = grad.reshape(grad.shape[0], -1).norm(dim=1)
            else:
                norm = grad.norm()
            ema = self.ema.get(name, norm)
            coef = torch.clamp(RATIO * ema / (norm + 1e-6), max=1.0)
            shape = (-1,) + (1,) * (grad.dim() - 1) if name in self.experts else ()
            out[name] = grad * (coef.view(shape) if shape else coef)
            self.ema[name] = ema + (norm * coef - ema) * (1 - BETA)
            coefs[name] = coef
        return out, coefs

    def averages(self, names, experts_slice=None):
        values = []
        for name in names:
            value = self.ema[name].reshape(-1)
            if name in self.experts and experts_slice is not None:
                value = value[experts_slice]
            values.append(value)
        return torch.cat(values)


def _full_grads(shapes: dict[str, tuple], step: int, spike: str | None = None):
    generator = torch.Generator(device="cpu").manual_seed(1000 + step)
    grads = {
        name: (
            torch.randn(shape, generator=generator, device="cpu") * (1.0 + 0.1 * index)
        ).to(DEVICE)
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
        actual.double().cpu(), expected.double().cpu(), rtol=1e-5, atol=1e-6, msg=what
    )


def _averages(trainer, params):
    return trainer._grad_clip_averages(params)


def _accelerated(optimizer):
    from accelerate import Accelerator

    return Accelerator(cpu=DEVICE == "cpu").prepare_optimizer(optimizer)


class _Experts(torch.nn.Module):
    """A fused experts module as ``@use_experts_implementation`` lays it out."""

    def __init__(self, num_experts, *, num_global=None, offset=0):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(num_experts, 6, 3))
        self.down_proj = torch.nn.Parameter(torch.zeros(num_experts, 3, 4))
        if num_global is not None and num_global > num_experts:
            self.num_local_experts = num_experts
            self.num_experts_global = num_global
            self.local_expert_offset = offset
            self._is_expert_parallel = True


EXPERTS = ("experts.gate_up_proj", "experts.down_proj")


def _fsdp_model(mesh):
    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = torch.nn.Module()
    # uneven rows, and a 1-row bias that leaves some ranks with an empty shard
    model.a = torch.nn.Linear(4, 5)
    model.b = torch.nn.Linear(5, 3)
    model.c = torch.nn.Linear(3, 1)
    model.experts = _Experts(3)  # 3 experts over 2 or 4 ranks: uneven expert shards
    for layer in (model.a, model.b, model.c, model.experts):
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
    mesh = init_device_mesh(DEVICE, (world,), mesh_dim_names=("dp_shard",))
    model = _fsdp_model(mesh)
    named = dict(model.named_parameters())
    shapes = {name: tuple(param.shape) for name, param in named.items()}
    optimizer = _accelerated(
        torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    )
    trainer = _Trainer(optimizer)
    reference = _Reference(EXPERTS)
    for step in range(STEPS):
        grads = _full_grads(shapes, step, spike="b.weight")
        if step == SPIKE_STEP:
            grads["experts.gate_up_proj"][1] *= 50.0
        _set_grads(named, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        expected, coefs = reference.step(grads)
        for name, param in named.items():
            _close(_full(param.grad), expected[name], f"step {step} {name} grad")
        _close(
            _averages(trainer, list(named.values())),
            reference.averages(named),
            f"step {step} averages",
        )
        if step == SPIKE_STEP:
            assert coefs["b.weight"] < 0.1, coefs
            # only the spiking expert is clipped hard
            experts = coefs["experts.gate_up_proj"]
            assert experts[1] < 0.1 and experts[0] > 0.5 and experts[2] > 0.5, experts
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # after the first optimizer step the averages live in the optimizer state
        for name, param in named.items():
            if step > 0:
                value = optimizer.optimizer.state[param][GRAD_NORM_EMA_KEY]
                shape = (3,) if name in EXPERTS else ()
                assert value.dtype == torch.float32 and value.shape == shape


def hsdp_tp():
    mesh = init_device_mesh(DEVICE, (2, 2), mesh_dim_names=("dp_shard", "tp"))
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
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


NUM_EXPERTS = 4
EP_SHAPES = {
    "experts.gate_up_proj": (NUM_EXPERTS, 6, 3),
    "experts.down_proj": (NUM_EXPERTS, 3, 4),
    "fallback": (4,),
    "dense": (8, 3),
    "replica": (3,),
    "norm": (3,),
}


def _ep_model(ep_size):
    """This rank's slice of an EP model on a ``(dp, ep)`` mesh, ``ep`` innermost.

    The experts FSDP-shard on ``dp`` (gate_up_proj on the expert dim, down_proj on
    dim 1); ``fallback`` is an EP-local tensor with no expert layout; the dense
    tensors shard over all ranks, replicate, or stay plain.
    """
    world = dist.get_world_size()
    mesh = DeviceMesh(
        DEVICE,
        torch.arange(world).reshape(world // ep_size, ep_size),
        mesh_dim_names=("dp", "ep"),
    )
    dp_rank, ep_rank = mesh.get_coordinate()
    dp_mesh = mesh["dp"]
    all_mesh = mesh._flatten("all")
    local = NUM_EXPERTS // ep_size
    experts = _Experts(local, num_global=NUM_EXPERTS, offset=ep_rank * local)
    experts.gate_up_proj = torch.nn.Parameter(
        distribute_tensor(torch.zeros(local, 6, 3), dp_mesh, (Shard(0),))
    )
    experts.down_proj = torch.nn.Parameter(
        distribute_tensor(torch.zeros(local, 3, 4), dp_mesh, (Shard(1),))
    )
    lora = torch.nn.Module()
    lora._ep_lora_sharded = True
    lora.weight = torch.nn.Parameter(torch.zeros(4 // ep_size))
    model = torch.nn.Module()
    model.experts = experts
    model.lora = lora
    model.dense = torch.nn.Parameter(
        distribute_tensor(torch.zeros(8, 3), all_mesh, (Shard(0),))
    )
    model.replica = torch.nn.Parameter(
        distribute_tensor(torch.zeros(3), all_mesh, (Replicate(),))
    )
    model.norm = torch.nn.Parameter(torch.zeros(3))
    named = {
        "experts.gate_up_proj": experts.gate_up_proj,
        "experts.down_proj": experts.down_proj,
        "fallback": lora.weight,
        "dense": model.dense,
        "replica": model.replica,
        "norm": model.norm,
    }
    mine = slice(ep_rank * local, (ep_rank + 1) * local)
    return mesh, model, named, mine, ep_rank


def _ep_grads(step, mine, ep_size, spike_expert=None):
    grads = _full_grads(EP_SHAPES, step)
    if step == SPIKE_STEP and spike_expert is not None:
        grads["experts.gate_up_proj"][spike_expert] *= 50.0
    fallback_piece = slice(mine.start * 4 // NUM_EXPERTS, mine.stop * 4 // NUM_EXPERTS)
    pieces = {
        "experts.gate_up_proj": mine,
        "experts.down_proj": mine,
        "fallback": fallback_piece,
    }
    return grads, pieces


def _set_ep_grads(named, pieces, grads):
    for name, param in named.items():
        value = grads[name][pieces.get(name, slice(None))].clone()
        param.grad = (
            distribute_tensor(
                value, param.device_mesh, param.placements, src_data_rank=None
            )
            if isinstance(param, DTensor)
            else value
        )


def _ep_run(ep_size, pure_ep=False):
    mesh, model, named, mine, ep_rank = _ep_model(ep_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
    trainer = _Trainer(
        optimizer, ep_mesh=mesh if ep_size > 1 else None, pure_ep=pure_ep
    )
    reference = _Reference(EXPERTS)
    for step in range(STEPS):
        grads, pieces = _ep_grads(step, mine, ep_size, spike_expert=3)
        _set_ep_grads(named, pieces, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        expected, coefs = reference.step(grads)
        for name, param in named.items():
            want = expected[name][pieces.get(name, slice(None))]
            _close(_full(param.grad), want, f"ep {ep_size} step {step} {name} grad")
        _close(
            _averages(trainer, list(named.values())),
            reference.averages(named, mine),
            f"ep {ep_size} step {step} averages",
        )
        if step == SPIKE_STEP:
            experts = coefs["experts.gate_up_proj"]
            assert experts[3] < 0.1 and (experts[:3] > 0.5).all(), experts
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    stored = optimizer.state[named["experts.gate_up_proj"]][GRAD_NORM_EMA_KEY]
    if ep_size > 1:
        assert isinstance(stored, DTensor) and stored.shape == (NUM_EXPERTS,), stored
        _close(stored.full_tensor(), reference.ema["experts.gate_up_proj"], "full")
    else:
        assert not isinstance(stored, DTensor) and stored.shape == (NUM_EXPERTS,)


def ep():
    # ep 1, 2 and 4 on the same gradients all match the per-expert reference
    for ep_size in (1, 2, 4):
        _ep_run(ep_size)
    _ep_run(4, pure_ep=True)

    # an expert tensor sharded over a mesh axis that spans ep would mix experts
    mesh = DeviceMesh(
        DEVICE, torch.arange(4).reshape(2, 2), mesh_dim_names=("dp", "ep")
    )
    bad = torch.nn.Module()
    bad.experts = _Experts(1, num_global=2)
    all_mesh = mesh._flatten("all_bad")
    bad.experts.gate_up_proj = torch.nn.Parameter(
        _dtensor(torch.zeros(1, 6, 3), all_mesh, (Shard(1),), (1, 24, 3))
    )
    bad.experts.gate_up_proj.grad = _dtensor(
        torch.ones(1, 6, 3), all_mesh, (Shard(1),), (1, 24, 3)
    )
    bad.experts.down_proj = torch.nn.Parameter(torch.zeros(1, 3, 4))
    bad_trainer = _Trainer(
        torch.optim.AdamW(bad.parameters(), foreach=False), ep_mesh=mesh
    )
    try:
        bad_trainer._get_grad_norm(bad, torch.tensor(1.0))
    except NotImplementedError as error:
        assert "mix experts" in str(error)
    else:
        raise AssertionError("an EP expert tensor sharded across ep must be rejected")


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
    optimizer = _accelerated(
        torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)
    )
    trainer = _Trainer(optimizer)
    if not ratio_before:
        trainer.args.grad_clip_norm_ratio = None
    half = STEPS // 2
    for step in range(half):
        _set_grads(named, _full_grads(shapes, step, spike="a.weight"))
        trainer._get_grad_norm(model, torch.tensor(1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    save_fsdp_optimizer(plugin, accelerator, optimizer, model, directory)
    dist.barrier()

    resumed_model = _fsdp_model(mesh)
    resumed_model.load_state_dict(model.state_dict())
    resumed_named = dict(resumed_model.named_parameters())
    resumed_optimizer = _accelerated(
        torch.optim.AdamW(resumed_model.parameters(), lr=1e-2, foreach=False)
    )
    resumed = _Trainer(resumed_optimizer)
    resumed.model = resumed_model
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
        grads = _full_grads(shapes, step, spike="a.weight")
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
            assert int(trainer._grad_clip_last_clipped) >= 1
        for opt in (optimizer, resumed_optimizer):
            opt.step()
            opt.zero_grad(set_to_none=True)
    for name, param in resumed_named.items():
        value = resumed_optimizer.optimizer.state[param][GRAD_NORM_EMA_KEY]
        shape = (3,) if name in EXPERTS else ()
        assert value.dtype == torch.float32 and value.shape == shape


def resume():
    mesh = init_device_mesh(
        DEVICE, (dist.get_world_size(),), mesh_dim_names=("dp_shard",)
    )
    for full in (False, True):
        directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(directory)
        _resume_run(mesh, _fsdp_plugin(full), directory[0])
    # a sharded checkpoint written before ratio clipping was enabled still loads
    directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(directory)
    _resume_run(mesh, _fsdp_plugin(False), directory[0], ratio_before=False)


def _ep_resume_run(ep_size, plugin, directory, pure_ep=False):
    from accelerate.utils.fsdp_utils import load_fsdp_optimizer, save_fsdp_optimizer

    accelerator = SimpleNamespace(
        process_index=dist.get_rank(), wait_for_everyone=dist.barrier
    )
    mesh, model, named, mine, _ = _ep_model(ep_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)
    trainer = _Trainer(optimizer, ep_mesh=mesh, pure_ep=pure_ep)
    half = STEPS // 2
    for step in range(half):
        grads, pieces = _ep_grads(step, mine, ep_size)
        _set_ep_grads(named, pieces, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    save_fsdp_optimizer(plugin, accelerator, optimizer, model, directory)
    dist.barrier()

    _, resumed_model, resumed_named, _, _ = _ep_model(ep_size)
    with torch.no_grad():
        for name, param in resumed_named.items():
            _local(param).copy_(_local(named[name]))
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=1e-2, foreach=False
    )
    resumed = _Trainer(resumed_optimizer, ep_mesh=mesh, pure_ep=pure_ep)
    resumed.model = resumed_model
    resumed.loader = lambda checkpoint: load_fsdp_optimizer(
        plugin, accelerator, resumed_optimizer, resumed_model, checkpoint
    )
    resumed._load_optimizer_and_scheduler(directory)
    # every EP rank gets its own experts' averages back
    _close(
        _averages(resumed, list(resumed_named.values())),
        _averages(trainer, list(named.values())),
        f"ep {ep_size} restored averages",
    )
    for step in range(half, STEPS):
        grads, pieces = _ep_grads(step, mine, ep_size, spike_expert=2)
        _set_ep_grads(named, pieces, grads)
        _set_ep_grads(resumed_named, pieces, grads)
        trainer._get_grad_norm(model, torch.tensor(1.0))
        resumed._get_grad_norm(resumed_model, torch.tensor(1.0))
        for name in named:
            _close(
                _full(resumed_named[name].grad),
                _full(named[name].grad),
                f"ep {ep_size} step {step} {name}",
            )
        _close(
            _averages(resumed, list(resumed_named.values())),
            _averages(trainer, list(named.values())),
            f"ep {ep_size} step {step} averages",
        )
        for opt in (optimizer, resumed_optimizer):
            opt.step()
            opt.zero_grad(set_to_none=True)


def ep_resume():
    from accelerate import PartialState

    PartialState(cpu=DEVICE == "cpu")
    for ep_size in (2, 4):
        for full in (False, True):
            directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
            dist.broadcast_object_list(directory)
            _ep_resume_run(ep_size, _fsdp_plugin(full), directory[0])
    directory = [tempfile.mkdtemp() if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(directory)
    _ep_resume_run(4, _fsdp_plugin(True), directory[0], pure_ep=True)


MODES = {
    "fsdp2": fsdp2,
    "hsdp_tp": hsdp_tp,
    "ep": ep,
    "resume": resume,
    "ep_resume": ep_resume,
}


def main():
    mode = sys.argv[1]
    if DEVICE == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.set_default_device("cuda")
        dist.init_process_group("nccl")
    else:
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
