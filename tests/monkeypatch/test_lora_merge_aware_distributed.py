"""Merge-aware LoRA under DDP and FSDP2: gradients match one process, and the FSDP2 vLLM
sync serves exactly the merge-aware forward's merged weights without touching the base."""

import types
from contextlib import nullcontext

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

WORLD_SIZE = 2
ROUNDS = 3


class _Block(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=True)

    def forward(self, x):
        return x + self.o(torch.tanh(self.q(x)))


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(_Block() for _ in range(2))
        self.head = nn.Linear(64, 8)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _model():
    from peft import LoraConfig, get_peft_model

    from axolotl.monkeypatch.lora_merge_aware import install_lora_merge_aware_linears
    from axolotl.utils.lora_precision import upcast_lora_parameters

    torch.manual_seed(0)
    model = get_peft_model(
        _Net().to(torch.bfloat16),
        LoraConfig(r=4, lora_alpha=8, target_modules=["q", "o"]),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(0, 1e-2)
    upcast_lora_parameters(model)
    assert install_lora_merge_aware_linears(model)[0] == 4
    return model


def _batches():
    torch.manual_seed(1)
    return [torch.randn(2, 64).to(torch.bfloat16) for _ in range(WORLD_SIZE)]


def _step(model, optimizer, inputs):
    optimizer.zero_grad()
    loss = model(inputs).float().pow(2).mean()
    loss.backward()
    return loss


def _reference_step(model, optimizer):
    optimizer.zero_grad()
    loss = sum(model(x).float().pow(2).mean() for x in _batches()) / WORLD_SIZE
    loss.backward()


def _lora_grads(model):
    from torch.distributed.tensor import DTensor

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        grads[name] = grad.full_tensor() if isinstance(grad, DTensor) else grad
    return grads


def _colocated_vllm(model, loaded):
    from trl.generation.vllm_generation import VLLMGeneration

    generation = object.__new__(VLLMGeneration)
    generation.model = model
    generation.mode = "colocate"
    generation.enable_sleep_mode = False
    generation.accelerator = types.SimpleNamespace(
        device=torch.device("cpu"), is_main_process=dist.get_rank() == 0
    )
    generation._dist = types.SimpleNamespace(
        is_fsdp=True, fsdp_version=2, gather_params=lambda params: nullcontext()
    )
    engine_model = types.SimpleNamespace(
        load_weights=lambda weights: loaded.update(
            (name, param.detach().clone()) for name, param in weights
        )
    )
    worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=engine_model)
    )
    generation.llm = types.SimpleNamespace(
        llm_engine=types.SimpleNamespace(
            model_executor=types.SimpleNamespace(driver_worker=worker)
        ),
        reset_prefix_cache=lambda: None,
    )
    return generation


def _ddp_worker(rank, rendezvous):
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        reference = _model()
        _reference_step(reference, torch.optim.SGD(reference.parameters(), lr=0.1))
        model = DistributedDataParallel(_model())
        _step(model, torch.optim.SGD(model.parameters(), lr=0.1), _batches()[rank])
        expected = _lora_grads(reference)
        got = {n.removeprefix("module."): g for n, g in _lora_grads(model).items()}
        assert got.keys() == expected.keys() and expected
        for name, grad in got.items():
            assert torch.equal(grad, expected[name]), name
    finally:
        dist.destroy_process_group()


def _fsdp2_worker(rank, rendezvous):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor
    from trl.generation.vllm_generation import VLLMGeneration

    from axolotl.monkeypatch.lora_merge_aware import merge_aware_vllm_sync
    from axolotl.utils.lora_merge_precision import merged_lora_weight
    from axolotl.utils.lora_precision import lora_fsdp2_precision_policy

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=WORLD_SIZE
    )
    try:
        reference = _model()
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        model = _model()
        base = {
            n: p.detach().clone()
            for n, p in model.named_parameters()
            if "lora_" not in n
        }
        mesh = init_device_mesh("cpu", (WORLD_SIZE,))
        policy = lora_fsdp2_precision_policy(
            MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        )
        for block in model.base_model.model.layers:
            fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loaded: dict[str, torch.Tensor] = {}
        generation = _colocated_vllm(model, loaded)
        merge_aware_vllm_sync(generation)
        x = torch.randn(3, 64).to(torch.bfloat16)
        for round_ in range(ROUNDS):
            _reference_step(reference, reference_optimizer)
            _step(model, optimizer, _batches()[rank])
            expected = _lora_grads(reference)
            for name, grad in _lora_grads(model).items():
                assert torch.equal(grad, expected[name]), (round_, name)
            reference_optimizer.step()
            optimizer.step()

            loaded.clear()
            VLLMGeneration.sync_weights(generation)
            for layer in range(2):
                for proj in ("q", "o"):
                    module = getattr(reference.base_model.model.layers[layer], proj)
                    served = loaded[f"layers.{layer}.{proj}.weight"]
                    assert torch.equal(
                        served,
                        merged_lora_weight(
                            module.base_layer.weight,
                            module.lora_A["default"].weight,
                            module.lora_B["default"].weight,
                            module.scaling["default"],
                            torch.bfloat16,
                        ),
                    ), (round_, layer, proj)
                    with torch.no_grad():
                        assert torch.equal(
                            F.linear(x, served, module.base_layer.bias), module(x)
                        ), (round_, layer, proj)
            assert "head.weight" in loaded and not any("lora" in n for n in loaded)
            for name, param in model.named_parameters():
                if name in base:
                    full = param.full_tensor() if isinstance(param, DTensor) else param
                    assert torch.equal(full, base[name]), (round_, name)
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed_cpu
@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")
@pytest.mark.parametrize("worker", [_ddp_worker, _fsdp2_worker], ids=["ddp", "fsdp2"])
def test_merge_aware_matches_single_process(tmp_path, worker):
    torch.multiprocessing.spawn(
        worker, args=(str(tmp_path / "rendezvous"),), nprocs=WORLD_SIZE
    )
