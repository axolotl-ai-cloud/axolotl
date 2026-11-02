# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Expert parallelism (EP=2, torch all-to-all backend) on Nemotron-Labs-TwoTower (2 GPUs).

Both towers' MoE layers are native ``NemotronHExperts`` (non-gated relu², ``up_proj`` /
``down_proj``). Two ranks run the ``ExpertParallelPlugin`` hooks on the tiny
ScatterMoE-test config, so each rank holds half of every experts module of BOTH towers
and runs its local experts through ScatterMoE. Each rank also builds the same model
unsharded on its own GPU as the single-GPU reference and compares:

* forward logits and loss of its own batch;
* gradients of every trainable parameter (denoiser LoRA, expert LoRA through
  ``target_parameters``, trainable tokens) against the reference gradient of the mean
  of both ranks' losses, after averaging replicated grads like DDP and gathering the
  EP-sharded expert LoRA;
* the frozen context tower: no trainable parameter and no gradient under EP;
* the routing of its own tokens, router by router, against the reference;
* denoiser gradient checkpointing through the plugin's selective-checkpoint policy:
  backward never re-runs the router ``topk`` or the split host copy, and the gradients
  equal an uncheckpointed EP step's on the same routing;
* unequal per-rank document counts (the context tower runs once per prompt, and every
  MoE call is a collective): the ranks pad their context-tower calls to the same count.

bf16 top-6 near-ties flip a few tokens' routing between any two expert kernels (see
``test_scattermoe_twotower._same_routing``), so the parity checks replay the reference
routing; the natural routing is compared separately. Tolerances are the ScatterMoE
TwoTower tests'.

The process-group timeout turns an all-to-all desync into an error instead of a hang.
The ranks use NCCL; ``AXOLOTL_TEST_EP_DIST_BACKEND=gloo`` runs the same collectives
through gloo on the CUDA tensors instead (the torch EP backend supports both), for hosts
where NCCL cannot initialise, e.g. an NVML library that mismatches the loaded driver
(also launch those with ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False``).
"""

from __future__ import annotations

import collections
import contextlib
import os
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = [
    pytest.mark.gpu,
    # a cold Triton autotune of the scattermoe LoRA kernels takes minutes
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="needs two CUDA GPUs",
    ),
]

WORLD = 2
PG_TIMEOUT_S = 90
SPAWN_TIMEOUT_S = 2400
# per rank: (prompt, canvas) per packed document. Rank 1 has a prompt-less document,
# so its context tower runs twice to rank 0's three times.
DOCUMENTS = {
    0: ((70, 32), (45, 32), (90, 32)),
    1: ((0, 32), (60, 32), (30, 16)),
}
TOKEN_IDS = [3, 5, 6, 200]
LOGITS_TOLERANCE = {torch.bfloat16: 5e-2, torch.float32: 1e-2}


def _progress(rank, message):
    import sys
    import time

    print(
        f"[rank {rank} {time.strftime('%H:%M:%S')}] {message}",
        file=sys.stderr,
        flush=True,
    )


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(
        b.float().abs().max().item(), 1e-8
    )


def _batch(rank, device):
    generator = torch.Generator().manual_seed(100 + rank)
    ids, targets, positions, canvas = [], [], [], []
    for prompt, width in DOCUMENTS[rank]:
        tokens = torch.randint(10, 257, (prompt + width,), generator=generator)
        targets.append(tokens.clone())
        noisy = torch.rand(width, generator=generator) < 0.5
        tokens[prompt:][noisy] = 3
        ids.append(tokens)
        positions.append(torch.arange(prompt + width))
        canvas.append(torch.cat([torch.zeros(prompt), torch.ones(width)]).bool())
    return tuple(
        torch.cat(part)[None].to(device) for part in (ids, targets, positions, canvas)
    )


def _lora_config(model_config):
    from peft import LoraConfig

    from axolotl.model_support.nemotron_twotower import (
        scope_target_modules,
        scope_target_parameters,
        scope_trainable_token_indices,
    )

    return LoraConfig(
        r=8,
        lora_alpha=8,
        use_rslora=True,
        lora_dropout=0.0,
        target_modules=scope_target_modules(
            ["q_proj", "o_proj", "in_proj", "out_proj", "up_proj", "down_proj"]
        ),
        target_parameters=scope_target_parameters(
            ["mixer.experts.up_proj", "mixer.experts.down_proj"], model_config
        ),
        trainable_token_indices=scope_trainable_token_indices(
            {"embed_tokens": TOKEN_IDS, "diffusion_head": TOKEN_IDS}
        ),
    )


def _peft(base):
    from peft import get_peft_model

    device = base.lm_head.weight.device
    return get_peft_model(base, _lora_config(base.config)).to(device)


@contextlib.contextmanager
def _routing(model, record, replay=None):
    """Record every router call's top-k into ``record[name]`` (one tensor per call).

    With ``replay`` (a record from another run), calls that run also made are given
    its choice, the weights recomputed from the live router logits as
    ``NemotronHTopkRouter`` does; later calls (EP padding passes) keep their own.
    """
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHTopkRouter

    calls: dict[str, int] = {}

    def hook(name, module, _inputs, output):
        router_logits, _, indices = output
        call = calls.get(name, 0)
        calls[name] = call + 1
        if replay is not None and call < len(replay.get(name, ())):
            indices = replay[name][call].to(indices.device)
            weights = router_logits.sigmoid().gather(1, indices)
            if module.norm_topk_prob:
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            output = (router_logits, weights * module.routed_scaling_factor, indices)
        record.setdefault(name, []).append(indices.detach().clone())
        return output

    handles = [
        module.register_forward_hook(lambda *a, name=name: hook(name, *a))
        for name, module in model.named_modules()
        if isinstance(module, NemotronHTopkRouter)
    ]
    assert handles
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _routing_diff(got, ref):
    """Tokens whose top-k set differs, over the calls both runs made, per tower."""
    out = {}
    for name, calls in ref.items():
        tower = "context" if name.startswith("context_tower.") else "denoiser"
        layer = int(name.split(".layers.")[1].split(".")[0])
        stats = out.setdefault(tower, {"tokens": 0, "mismatched": 0, "first_layer": {}})
        for call, (mine, theirs) in enumerate(zip(got[name], calls, strict=False)):
            assert mine.shape == theirs.shape, (name, call, mine.shape, theirs.shape)
            differ = (
                (mine.sort(-1).values != theirs.to(mine.device).sort(-1).values)
                .any(-1)
                .sum()
                .item()
            )
            stats["tokens"] += mine.shape[0]
            stats["mismatched"] += differ
            first = stats["first_layer"].setdefault(layer, 0)
            stats["first_layer"][layer] = first + differ
    for stats in out.values():
        first = min(stats["first_layer"])
        stats["first_layer"] = stats["first_layer"][first]
    return out


def _loss(peft_model, batch):
    ids, targets, positions, canvas = batch
    out = peft_model(ids, position_ids=positions, axolotl_canvas_mask=canvas)
    logits = out.logits[canvas]
    return logits, torch.nn.functional.cross_entropy(logits.float(), targets[canvas])


def _trainable(peft_model):
    return {n: p for n, p in peft_model.named_parameters() if p.requires_grad}


def _memory(model):
    torch.cuda.synchronize()
    return {
        "param_mib": sum(p.numel() * p.element_size() for p in model.parameters())
        / 2**20,
        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def _reference(dtype, rank):
    """Single-GPU (no EP) ScatterMoE run over both ranks' batches on this rank's GPU."""
    from tests.e2e.kernels.test_scattermoe_twotower import _model, _register

    _register()
    torch.cuda.reset_peak_memory_stats()
    base = _model(dtype)
    peft_model = _peft(base)
    torch.manual_seed(7)
    with torch.no_grad():
        # non-zero B so every LoRA delta (experts included) is live in the forward
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.02)
            elif "trainable_tokens_delta" in name:
                parameter.normal_(0, 0.02)
    adapter = {n: p.detach().clone() for n, p in _trainable(peft_model).items()}
    base.set_experts_implementation("scattermoe")
    peft_model.train()
    out = {"logits": {}, "loss": {}, "routing": {}}
    for batch_rank in range(WORLD):
        record: dict = {}
        with _routing(base, record):
            logits, loss = _loss(peft_model, _batch(batch_rank, "cuda"))
        (loss / WORLD).backward()
        out["logits"][batch_rank] = logits.detach().float()
        out["loss"][batch_rank] = loss.item()
        out["routing"][batch_rank] = record
    out["grads"] = {
        n: p.grad.detach().float().clone() for n, p in _trainable(peft_model).items()
    }
    out["adapter"] = adapter
    out["memory"] = _memory(peft_model)
    del peft_model, base
    torch.cuda.empty_cache()
    return out


def _build_ep(dtype, rank, adapter, *, gradient_checkpointing):
    """The plugin's hooks in loader order: pre_model_load, build, post_model_build
    (shard), PEFT, post_model_load; then the reference adapter cut to local experts."""
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin
    from axolotl.integrations.expert_parallel.shard import (
        _detect_experts_modules,
        ep_adapter_load_local_shard,
        shard_expert_lora,
    )
    from axolotl.monkeypatch.selective_checkpointing import clear_registered_saves
    from axolotl.utils.dict import DictDefault

    from tests.e2e.kernels.test_scattermoe_twotower import _model

    clear_registered_saves()
    cfg = DictDefault(
        expert_parallel_size=WORLD,
        expert_parallel_backend="torch",
        use_scattermoe=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    plugin = ExpertParallelPlugin()
    plugin.pre_model_load(cfg)
    assert cfg.experts_implementation == "expert_parallel"
    torch.cuda.reset_peak_memory_stats()
    base = _model(dtype)
    base.set_experts_implementation(cfg.experts_implementation)
    plugin.post_model_build(cfg, base)
    experts = dict(_detect_experts_modules(base))
    num_experts = base.config.n_routed_experts
    sharded = {
        name: (module.num_local_experts, module.local_expert_offset)
        for name, module in experts.items()
    }
    peft_model = _peft(base)
    with torch.no_grad():
        for name, parameter in _trainable(peft_model).items():
            source = adapter[name]
            if source.shape != parameter.shape:
                source = ep_adapter_load_local_shard(
                    source,
                    0 if ".lora_A." in name else 1,
                    num_experts,
                    rank,
                    WORLD,
                    (),
                    1,
                    0,
                )
            parameter.copy_(source)
    # what the FSDP2 prepare path does for pure EP: flag the (already local) expert
    # LoRA for the gathered save and scale its grads by 1/ep
    shard_expert_lora(peft_model, WORLD)
    plugin.post_model_load(cfg, peft_model)
    if gradient_checkpointing:
        peft_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        assert base.gradient_checkpointing
    peft_model.train()
    return peft_model, base, sharded


def _synced_grads(peft_model, reference_grads):
    """DDP-averaged replicated grads; EP-sharded expert-LoRA grads gathered to E_global."""
    from axolotl.integrations.expert_parallel.shard import gather_expert_lora_full

    num_experts = peft_model.base_model.model.config.n_routed_experts
    grads = {}
    for name, parameter in _trainable(peft_model).items():
        grad = parameter.grad
        assert grad is not None, name
        grad = grad.detach().float().clone()
        if grad.shape != reference_grads[name].shape:
            grad = gather_expert_lora_full(
                grad,
                "A" if ".lora_A." in name else "B",
                num_experts,
                dist.group.WORLD,
            )
        else:
            dist.all_reduce(grad)
            grad /= WORLD
        grads[name] = grad
    return grads


def _grad_report(grads, reference):
    report = {}
    for name, ref in reference.items():
        diff = (grads[name] - ref).abs().max().item()
        report[name] = (diff, diff / (ref.abs().max().item() + 1e-8))
    return report


class _OpCounter(TorchDispatchMode):
    WATCHED = {
        "aten::topk",
        "axolotl::ep_to_host",
        "axolotl::ep_all_to_all_single",
        "axolotl::ep_all_to_all_single_equal",
    }

    def __init__(self):
        super().__init__()
        self.counts = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func.name().split(".")[0]
        if name in self.WATCHED:
            self.counts[name] += 1
        return func(*args, **(kwargs or {}))


def _ep_checks(dtype, rank):
    from axolotl.monkeypatch.selective_checkpointing import clear_registered_saves

    _progress(rank, f"{dtype}: reference")
    reference = _reference(dtype, rank)
    _progress(rank, f"{dtype}: EP build")
    batch = _batch(rank, "cuda")
    out = {"reference_memory": reference["memory"], "reference_loss": {}}
    out["reference_loss"] = reference["loss"][rank]

    peft_model, base, sharded = _build_ep(
        dtype, rank, reference["adapter"], gradient_checkpointing=False
    )
    out["sharded"] = sharded
    out["experts_implementation"] = base.config._experts_implementation

    # natural routing, plain EP step
    natural: dict = {}
    with _routing(base, natural):
        logits, loss = _loss(peft_model, batch)
    loss.backward()
    out["routing"] = _routing_diff(natural, reference["routing"][rank])
    out["context_calls"] = len(
        next(v for k, v in natural.items() if k.startswith("context_tower."))
    )
    out["natural"] = {
        "logits_rel": _rel(logits, reference["logits"][rank]),
        "loss": loss.item(),
    }
    out["context_trainable"] = [
        n
        for n, p in peft_model.named_parameters()
        if p.requires_grad and "context_tower" in n
    ]
    out["context_grads"] = [
        n
        for n, p in peft_model.named_parameters()
        if p.grad is not None and "context_tower" in n
    ]

    _progress(rank, f"{dtype}: EP replayed-routing step")
    # parity on the reference routing
    peft_model.zero_grad(set_to_none=True)
    replayed: dict = {}
    with _routing(base, replayed, replay=reference["routing"][rank]):
        logits, loss = _loss(peft_model, batch)
    loss.backward()
    grads = _synced_grads(peft_model, reference["grads"])
    out["parity"] = {
        "logits_rel": _rel(logits, reference["logits"][rank]),
        "logits_abs": (logits.float() - reference["logits"][rank]).abs().max().item(),
        "loss": loss.item(),
        "grads": _grad_report(grads, reference["grads"]),
        "grad_names": sorted(grads),
        "reference_grad_names": sorted(reference["grads"]),
        "expert_lora": sorted(n for n in grads if ".experts." in n),
        "token_adapters": sorted(n for n in grads if "trainable_tokens" in n),
    }
    out["ep_memory"] = _memory(peft_model)
    del peft_model, base
    torch.cuda.empty_cache()

    _progress(rank, f"{dtype}: EP gradient-checkpointing step")
    # denoiser gradient checkpointing under the plugin's selective-checkpoint policy
    peft_model, base, _ = _build_ep(
        dtype, rank, reference["adapter"], gradient_checkpointing=True
    )
    forward, backward = _OpCounter(), _OpCounter()
    checkpointed_routing: dict = {}
    with forward, _routing(base, checkpointed_routing):
        logits, loss = _loss(peft_model, batch)
    with backward:
        loss.backward()
    checkpointed_loss = loss.item()
    checkpointed = _synced_grads(peft_model, reference["grads"])
    checkpointed_memory = _memory(peft_model)
    del peft_model, base
    torch.cuda.empty_cache()

    _progress(rank, f"{dtype}: EP step on the checkpointed routing")
    # The combine's index_add_ sums in atomic order, so two bf16 forwards differ in the
    # last bits and flip near-tied top-6 choices: compare against the same routing.
    peft_model, base, _ = _build_ep(
        dtype, rank, reference["adapter"], gradient_checkpointing=False
    )
    with _routing(base, {}, replay=checkpointed_routing):
        _, loss = _loss(peft_model, batch)
    loss.backward()
    out["checkpointing"] = {
        "forward": dict(forward.counts),
        "backward": dict(backward.counts),
        "loss_diff": abs(checkpointed_loss - loss.item()),
        "grads_vs_plain_ep": _grad_report(
            checkpointed, _synced_grads(peft_model, reference["grads"])
        ),
        "memory": checkpointed_memory,
    }
    del peft_model, base
    torch.cuda.empty_cache()
    clear_registered_saves()
    _progress(rank, f"{dtype}: done")
    return out


def _worker(rank, world_size, port, queue):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.cuda.set_device(rank)
    backend = os.environ.get("AXOLOTL_TEST_EP_DIST_BACKEND", "nccl")
    dist.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=PG_TIMEOUT_S),
        device_id=torch.device("cuda", rank) if backend == "nccl" else None,
    )
    try:
        results = {}
        for dtype in (torch.float32, torch.bfloat16):
            try:
                results[str(dtype)] = _ep_checks(dtype, rank)
            except Exception:  # pylint: disable=broad-exception-caught
                results[str(dtype)] = traceback.format_exc()
        queue.put((rank, results))
    except Exception:  # pylint: disable=broad-exception-caught
        queue.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ep2_results():
    from tests.integrations.expert_parallel.test_torch_backend import _run_spawned

    results = _run_spawned(_worker, world_size=WORLD, timeout=SPAWN_TIMEOUT_S)
    dump = os.environ.get("AXOLOTL_TEST_EP_RESULTS_JSON")
    if dump:
        import json

        with open(dump, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=1, default=str)
    return results


DTYPES = [torch.float32, torch.bfloat16]


def _section(results, dtype):
    out = {rank: res[str(dtype)] for rank, res in results.items()}
    errors = {rank: res for rank, res in out.items() if isinstance(res, str)}
    assert not errors, "\n".join(f"rank {r}:\n{e}" for r, e in errors.items())
    return out


@pytest.mark.parametrize("dtype", DTYPES)
def test_experts_of_both_towers_are_sharded(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        assert res["experts_implementation"] == "expert_parallel"
        sharded = res["sharded"]
        towers = {name.split(".")[0] for name in sharded}
        assert towers == {"context_tower", "denoiser_tower"}, sharded
        # tiny config: 3 MoE layers per tower, 16 experts -> 8 per rank
        assert len(sharded) == 6, sharded
        assert all(count == (8, 8 * rank) for count in sharded.values()), sharded


@pytest.mark.parametrize("dtype", DTYPES)
def test_forward_matches_single_gpu(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        parity = res["parity"]
        assert parity["logits_rel"] < LOGITS_TOLERANCE[dtype], (rank, parity)
        loss_rel = abs(parity["loss"] - res["reference_loss"]) / res["reference_loss"]
        assert loss_rel < LOGITS_TOLERANCE[dtype], (rank, parity["loss"], loss_rel)


@pytest.mark.parametrize("dtype", DTYPES)
def test_gradients_match_single_gpu(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        parity = res["parity"]
        assert parity["grad_names"] == parity["reference_grad_names"]
        assert parity["expert_lora"], "no expert LoRA gradient"
        assert len(parity["token_adapters"]) == 2, parity["token_adapters"]
        for name, (diff, rel) in parity["grads"].items():
            assert not (rel > 0.1 and diff > 1e-2), (rank, name, diff, rel)


@pytest.mark.parametrize("dtype", DTYPES)
def test_context_tower_frozen_without_grads(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        assert not res["context_trainable"], (rank, res["context_trainable"])
        assert not res["context_grads"], (rank, res["context_grads"])
        assert not any("context_tower" in n for n in res["parity"]["grad_names"])


@pytest.mark.parametrize("dtype", DTYPES)
def test_context_tower_calls_padded_to_the_same_count(ep2_results, dtype):
    calls = {
        rank: res["context_calls"] for rank, res in _section(ep2_results, dtype).items()
    }
    prompts = {
        rank: sum(prompt > 0 for prompt, _ in DOCUMENTS[rank]) for rank in DOCUMENTS
    }
    assert prompts[0] != prompts[1]
    assert calls == {rank: max(prompts.values()) for rank in calls}, calls


@pytest.mark.parametrize("dtype", DTYPES)
def test_routing_matches_single_gpu(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        for tower, stats in res["routing"].items():
            # the first MoE layer of each tower sees identical inputs
            assert stats["first_layer"] == 0, (rank, tower, stats)
            if dtype is torch.float32:
                assert stats["mismatched"] == 0, (rank, tower, stats)
            else:
                # bf16 near-ties only: a handful of tokens downstream
                assert stats["mismatched"] <= 0.05 * stats["tokens"], (
                    rank,
                    tower,
                    stats,
                )


@pytest.mark.parametrize("dtype", DTYPES)
def test_gradient_checkpointing_saves_routing(ep2_results, dtype):
    for rank, res in _section(ep2_results, dtype).items():
        ckpt = res["checkpointing"]
        # recompute reuses the forward's top-k and split host copies
        assert "aten::topk" not in ckpt["backward"], (rank, ckpt)
        assert "axolotl::ep_to_host" not in ckpt["backward"], (rank, ckpt)
        # ...but does re-run the dispatch (expert_parallel_save_dispatch is off)
        assert ckpt["backward"].get("axolotl::ep_all_to_all_single", 0) > 0, ckpt
        loss = res["natural"]["loss"]
        if dtype is torch.float32:
            # the recompute replays the forward exactly
            assert ckpt["loss_diff"] <= 1e-6 * max(1.0, loss), ckpt
            for name, (diff, rel) in ckpt["grads_vs_plain_ep"].items():
                assert not (rel > 1e-2 and diff > 1e-4), (rank, name, diff, rel)
        else:
            # same routing, but bf16 atomics still differ in the last bits: hold
            # them to the parity tolerances
            assert ckpt["loss_diff"] <= LOGITS_TOLERANCE[dtype] * loss, ckpt
            for name, (diff, rel) in ckpt["grads_vs_plain_ep"].items():
                assert not (rel > 0.1 and diff > 1e-2), (rank, name, diff, rel)
