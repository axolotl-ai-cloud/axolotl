"""Tests for the torch all-to-all expert-parallel backend (CPU, gloo)."""

import copy
import os
import queue as queue_mod
import socket
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from axolotl.integrations.expert_parallel import torch_dispatch as TD
from axolotl.integrations.expert_parallel.torch_dispatch import (
    build_send_layout,
    combine,
    compute_send_counts,
    dispatch,
)

WORLD = 2
E, K, H, INTER, T = 8, 2, 16, 32, 12
E_LOCAL = E // WORLD


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _init_gloo(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _run_spawned(target, world_size=WORLD, timeout=180):
    """Spawn ``target(rank, world_size, port, q)`` on gloo; return per-rank results by rank.

    Workers report ``(rank, result)``; an exception inside a worker is reported as
    ``(rank, traceback_str)`` so a failure surfaces as an assertion, not a hang.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=target, args=(r, world_size, port, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.monotonic() + timeout
    while len(results) < world_size and time.monotonic() < deadline:
        try:
            rank, res = q.get(timeout=5)
            results[rank] = res
        except queue_mod.Empty:
            if all(not p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=20)
        if p.is_alive():
            p.kill()
    errors = {r: res for r, res in results.items() if isinstance(res, str)}
    assert not errors, "\n".join(f"rank {r}:\n{e}" for r, e in errors.items())
    exitcodes = [p.exitcode for p in procs]
    assert len(results) == world_size, (
        f"only {len(results)}/{world_size} workers reported; exitcodes={exitcodes}"
    )
    assert all(code == 0 for code in exitcodes), f"worker exitcodes={exitcodes}"
    return results


def _ep2_worker(rank, world_size, port, q):
    """All 2-rank checks in one spawn: process start-up (torch/transformers/peft imports)
    dominates the runtime. Each section's failure is reported as its traceback string."""
    _init_gloo(rank, world_size, port)
    try:
        out = {}
        for name, fn in (
            ("custom_op", _custom_op_checks),
            ("parity", _parity_checks),
            ("sac", _sac_checks),
            ("chunked", _chunked_checks),
        ):
            try:
                out[name] = fn(rank, world_size)
            except Exception:  # pylint: disable=broad-exception-caught
                out[name] = traceback.format_exc()
        q.put((rank, out))
    except Exception:  # pylint: disable=broad-exception-caught
        q.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


@pytest.fixture(scope="module")
def ep2_results():
    return _run_spawned(_ep2_worker)


def _section(results, name):
    errors = {r: res[name] for r, res in results.items() if isinstance(res[name], str)}
    assert not errors, "\n".join(f"rank {r}:\n{e}" for r, e in errors.items())
    return {r: res[name] for r, res in results.items()}


# --------------------------------------------------------------------------- #
# Send layout (single process, P simulated)
# --------------------------------------------------------------------------- #


def _reference_layout(idx, num_ranks, num_local):
    """Brute-force DeepEP-contract layout: rank-major, token-ascending, one row per
    (token, destination rank), local ids for that rank's slots and -1 elsewhere."""
    token_ids, ranks, local_idx = [], [], []
    for r in range(num_ranks):
        for t, row in enumerate(idx.tolist()):
            owned = [e >= 0 and e // num_local == r for e in row]
            if not any(owned):
                continue
            token_ids.append(t)
            ranks.append(r)
            local_idx.append(
                [
                    e - r * num_local if o else -1
                    for e, o in zip(row, owned, strict=True)
                ]
            )
    return token_ids, ranks, local_idx


def _layout(idx, w, num_ranks, num_local):
    dest, is_in_rank, send_counts = compute_send_counts(idx, num_ranks, num_local)
    tok, rank, send_idx, send_w = build_send_layout(
        idx, w, dest, is_in_rank, int(send_counts.sum()), num_local
    )
    return send_counts, tok, rank, send_idx, send_w


class TestSendLayout:
    def _check_against_reference(self, idx, num_ranks, num_local):
        w = torch.rand(idx.shape, dtype=torch.float32)
        counts, tok, rank, send_idx, send_w = _layout(idx, w, num_ranks, num_local)
        ref_tok, ref_rank, ref_idx = _reference_layout(idx, num_ranks, num_local)
        assert tok.tolist() == ref_tok
        assert rank.tolist() == ref_rank
        assert send_idx.tolist() == ref_idx
        assert counts.tolist() == [ref_rank.count(r) for r in range(num_ranks)]
        expected_w = torch.where(
            send_idx >= 0, w[tok], torch.zeros_like(send_idx, dtype=w.dtype)
        )
        assert torch.equal(send_w, expected_w)
        assert send_idx.dtype == torch.int64 and send_w.dtype == torch.float32
        return counts, tok, rank, send_idx, send_w

    @pytest.mark.parametrize(
        "num_ranks,num_experts,top_k", [(2, 8, 2), (4, 16, 4), (8, 8, 3)]
    )
    def test_random_routing_matches_reference(self, num_ranks, num_experts, top_k):
        g = torch.Generator().manual_seed(num_ranks * 100 + top_k)
        idx = torch.stack(
            [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(64)]
        )
        idx[torch.rand(idx.shape, generator=g) < 0.1] = -1
        self._check_against_reference(idx, num_ranks, num_experts // num_ranks)

    def test_dedup_one_row_per_destination_rank(self):
        # experts 0,1 on rank 0 (sent once); 2 on rank 0 and 5 on rank 1 (sent to both)
        idx = torch.tensor([[0, 1], [2, 5]])
        counts, tok, rank, send_idx, _ = self._check_against_reference(idx, 2, 4)
        assert counts.tolist() == [2, 1]
        assert tok.tolist() == [0, 1, 1]
        assert rank.tolist() == [0, 0, 1]
        assert send_idx.tolist() == [[0, 1], [2, -1], [-1, 1]]

    def test_rank_major_token_ascending_order(self):
        idx = torch.tensor([[6, 0], [1, 7], [4, 5], [3, 2]])
        _, tok, rank, _, _ = self._check_against_reference(idx, 2, 4)
        assert rank.tolist() == [0, 0, 0, 1, 1, 1]
        assert tok.tolist() == [0, 1, 3, 0, 1, 2]

    def test_local_id_remap(self):
        idx = torch.tensor([[7, 4], [5, 6]])
        _, _, rank, send_idx, _ = self._check_against_reference(idx, 2, 4)
        assert rank.tolist() == [1, 1]
        assert send_idx.tolist() == [[3, 0], [1, 2]]

    def test_unrouted_rows_are_dropped(self):
        idx = torch.tensor([[-1, -1], [1, -1], [-1, -1], [-1, 6]])
        counts, tok, _, send_idx, send_w = self._check_against_reference(idx, 2, 4)
        assert counts.tolist() == [1, 1]
        assert tok.tolist() == [1, 3]
        assert send_idx.tolist() == [[1, -1], [-1, 2]]
        assert (send_w[send_idx < 0] == 0).all()

    def test_all_tokens_to_one_rank(self):
        idx = torch.tensor([[0, 1], [2, 3], [3, 0]])
        counts, *_ = self._check_against_reference(idx, 2, 4)
        assert counts.tolist() == [3, 0]

    def test_zero_tokens_to_some_ranks(self):
        idx = torch.tensor([[0, 7], [6, 1]])
        counts, *_ = self._check_against_reference(idx, 4, 2)
        assert counts.tolist() == [2, 0, 0, 2]

    def test_top_k_larger_than_local_experts(self):
        # K=3 > E_local=2: every token owns slots on several ranks, at most E_local each
        idx = torch.tensor([[0, 1, 2], [3, 1, 0], [2, 3, -1]])
        _, _, _, send_idx, _ = self._check_against_reference(idx, 2, 2)
        assert ((send_idx >= 0).sum(dim=1) <= 2).all()
        assert send_idx.tolist() == [
            [0, 1, -1],
            [-1, 1, 0],
            [-1, -1, 0],
            [1, -1, -1],
            [0, 1, -1],
        ]

    def test_no_routed_tokens(self):
        idx = torch.full((3, 2), -1)
        counts, tok, _, send_idx, _ = self._check_against_reference(idx, 2, 4)
        assert counts.tolist() == [0, 0]
        assert tok.numel() == 0 and send_idx.shape == (0, 2)


class TestSingleRankPath:
    def test_dispatch_combine_are_collective_free(self):
        x = torch.randn(4, H)
        idx = torch.tensor([[0, 3], [-1, 2], [-1, -1], [1, 1]])
        w = torch.rand(4, 2)
        recv_x, recv_idx, recv_w, handle = dispatch(
            x, idx, w, num_local_experts=E, group=None
        )
        assert recv_x is x and recv_idx is idx
        assert torch.equal(recv_w, torch.where(idx >= 0, w, torch.zeros_like(w)))
        out = torch.randn(4, H)
        assert combine(out, handle) is out
        assert combine(out, handle, dtype=torch.bfloat16).dtype == torch.bfloat16

    def test_sharded_experts_without_group_raise(self):
        from axolotl.integrations.expert_parallel.experts_fn import _ep_forward

        experts = _build_experts()
        _shard_experts(experts, rank=0, world_size=2)
        TD.set_ep_group(None)
        idx = torch.tensor([[0, 5]])
        with pytest.raises(RuntimeError, match="EP group is unset"):
            _ep_forward(
                experts,
                torch.randn(1, H),
                idx,
                torch.rand(1, 2),
                kernel_name="eager",
                backend="torch",
            )


# --------------------------------------------------------------------------- #
# Custom ops
# --------------------------------------------------------------------------- #


class TestCustomOps:
    def test_registered_under_axolotl_namespace(self):
        assert TD.OPS_REGISTERED
        schema = str(torch.ops.axolotl.ep_all_to_all_single.default._schema)
        assert schema == (
            "axolotl::ep_all_to_all_single(Tensor x, Tensor output_splits, "
            "Tensor input_splits, str group_name) -> Tensor"
        )
        assert (
            torch.ops.axolotl.ep_all_to_all_single_equal.default.namespace == "axolotl"
        )

    def test_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        with FakeTensorMode(shape_env=ShapeEnv()):
            splits = torch.tensor([2, 3])
            y = torch.ops.axolotl.ep_all_to_all_single(
                torch.empty(5, H), splits, splits, "unused"
            )
            z = torch.ops.axolotl.ep_all_to_all_single_equal(
                torch.empty(4, 3, dtype=torch.int64), "unused"
            )
        assert isinstance(y.shape[0], torch.SymInt)
        assert y.shape[1] == H and y.dtype == torch.float32
        assert tuple(z.shape) == (4, 3) and z.dtype == torch.int64

    def test_forward_and_backward_two_ranks(self, ep2_results):
        for rank, res in _section(ep2_results, "custom_op").items():
            for name, ok in res.items():
                assert ok, f"rank {rank}: {name}"


# rows each rank sends to each destination; rank 1 sends nothing to itself
_SEND = [[1, 3], [2, 0]]


def _custom_op_checks(rank, world_size):
    group = dist.group.WORLD
    send = torch.tensor(_SEND[rank], dtype=torch.int64)
    recv = torch.tensor([_SEND[s][rank] for s in range(world_size)], dtype=torch.int64)

    def expected_rows():
        rows = []
        for src in range(world_size):
            start = sum(_SEND[src][:rank])
            rows += [src * 100 + i for i in range(start, start + _SEND[src][rank])]
        return torch.tensor(rows, dtype=torch.float32)

    dest_of_row = torch.repeat_interleave(torch.arange(world_size), send)
    out = {}
    for registered in (True, False):
        TD.OPS_REGISTERED = registered
        tag = "op" if registered else "autograd.Function"
        try:
            x = rank * 100 + torch.arange(int(send.sum()), dtype=torch.float32)
            x = x.unsqueeze(-1).expand(-1, 3).contiguous().requires_grad_(True)
            y = TD.all_to_all_single(x, recv, send, group)
            out[f"{tag} uneven forward"] = torch.equal(y[:, 0], expected_rows())
            # dL/dy = y * (rank + 1): x.grad must be x scaled by (its destination + 1)
            (0.5 * (y**2) * (rank + 1)).sum().backward()
            out[f"{tag} uneven backward = reverse a2a"] = torch.equal(
                x.grad, x.detach() * (dest_of_row + 1).unsqueeze(-1)
            )

            with torch.no_grad():
                ids = TD.all_to_all_single(
                    torch.arange(int(send.sum()), dtype=torch.int64) + rank * 100,
                    recv,
                    send,
                    group,
                )
            out[f"{tag} uneven int64"] = ids.dtype == torch.int64 and torch.equal(
                ids.float(), expected_rows()
            )

            xe = (
                rank * 10 + torch.arange(world_size, dtype=torch.float32)
            ).requires_grad_(True)
            ye = TD.all_to_all_single_equal(xe, group)
            out[f"{tag} equal forward"] = ye.tolist() == [
                s * 10 + rank for s in range(world_size)
            ]
            (ye * (rank + 1)).sum().backward()
            out[f"{tag} equal backward"] = xe.grad.tolist() == [
                d + 1 for d in range(world_size)
            ]
        finally:
            TD.OPS_REGISTERED = True
    return out


# --------------------------------------------------------------------------- #
# End-to-end parity: EP=2 vs single-rank reference
# --------------------------------------------------------------------------- #


def _qwen3moe_config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    return Qwen3MoeConfig(
        hidden_size=H,
        moe_intermediate_size=INTER,
        num_experts=E,
        num_experts_per_tok=K,
        norm_topk_prob=True,
    )


def _build_experts():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeExperts

    torch.manual_seed(0)
    experts = Qwen3MoeExperts(_qwen3moe_config())
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.2)
        experts.down_proj.normal_(0, 0.2)
    return experts


def _build_block():
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

    torch.manual_seed(0)
    block = Qwen3MoeSparseMoeBlock(_qwen3moe_config())
    with torch.no_grad():
        block.experts.gate_up_proj.normal_(0, 0.2)
        block.experts.down_proj.normal_(0, 0.2)
        block.gate.weight.normal_(0, 1.0)
    return block


def _shard_experts(experts, rank, world_size):
    """What ``shard_expert_weights`` leaves behind, without its CUDA scatter."""
    e_local = experts.gate_up_proj.shape[0] // world_size
    s = slice(rank * e_local, (rank + 1) * e_local)
    experts.gate_up_proj = torch.nn.Parameter(experts.gate_up_proj.detach()[s].clone())
    experts.down_proj = torch.nn.Parameter(experts.down_proj.detach()[s].clone())
    experts.num_experts_global = experts.num_experts
    experts.num_local_experts = e_local
    experts.local_expert_offset = rank * e_local
    experts.num_experts = e_local
    return s


def _routing(rank, scenario, num_tokens=T):
    g = torch.Generator().manual_seed(100 + rank)
    x = torch.randn(num_tokens, H, generator=g)
    w = torch.rand(num_tokens, K, generator=g)
    if scenario == "mixed":
        idx = torch.stack(
            [torch.randperm(E, generator=g)[:K] for _ in range(num_tokens)]
        )
        idx[0] = torch.tensor([5, 6])  # both on rank 1: sent once
        idx[1] = torch.tensor([1, 2])  # both on rank 0: sent once
        idx[2] = torch.tensor([3, 4])  # one expert per rank
        idx[3] = torch.tensor([-1, -1])  # unrouted: not dispatched, zero output
        idx[4] = torch.tensor([7, -1])
    elif scenario == "rank1_receives_zero":
        idx = torch.stack(
            [torch.randperm(E_LOCAL, generator=g)[:K] for _ in range(num_tokens)]
        )
    elif scenario == "random":
        idx = torch.stack(
            [torch.randperm(E, generator=g)[:K] for _ in range(num_tokens)]
        )
    else:
        raise ValueError(scenario)
    return x, idx, w


def _max_diff(a, b):
    if a is None:
        a = torch.zeros_like(b)
    return (a - b).abs().max().item() if a.numel() else 0.0


def _parity_checks(rank, world_size):
    """Returns ``{case: {metric: max_abs_diff}}``.

    Compared per rank: forward output and grads w.r.t. x, the routing weights and the
    router, all against the full-expert reference run on the SAME rank's tokens. Expert
    weights: each rank's local slice collects grads from every rank's tokens, so its
    raw grad equals the SUM over ranks of the reference grad for those experts; with the
    plugin's 1/ep post-accumulate hook it equals the ep-mean (what DDP produces for the
    reference model).
    """
    from axolotl.integrations.expert_parallel.experts_fn import (
        _ep_forward,
        register_all,
    )
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    register_all()
    TD.set_ep_group(dist.group.WORLD)
    results = {}

    full = _build_experts()
    for scenario in ("mixed", "rank1_receives_zero"):
        for kernel in ("eager", "grouped_mm"):
            x, idx, w = _routing(rank, scenario)
            gout = torch.randn(T, H, generator=torch.Generator().manual_seed(7 + rank))

            ref = copy.deepcopy(full)
            xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
            yr = _ep_forward(ref, xr, idx, wr, kernel_name=kernel, backend="torch")
            (yr * gout).sum().backward()

            ep = copy.deepcopy(full)
            s = _shard_experts(ep, rank, world_size)
            xe, we = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
            ye = _ep_forward(ep, xe, idx, we, kernel_name=kernel, backend="torch")
            (ye * gout).sum().backward()

            ref_gu, ref_dn = ref.gate_up_proj.grad.clone(), ref.down_proj.grad.clone()
            dist.all_reduce(ref_gu)
            dist.all_reduce(ref_dn)
            metrics = {
                "fwd": _max_diff(ye, yr),
                "dx": _max_diff(xe.grad, xr.grad),
                "dw": _max_diff(we.grad, wr.grad),
                "d_gate_up": _max_diff(ep.gate_up_proj.grad, ref_gu[s]),
                "d_down": _max_diff(ep.down_proj.grad, ref_dn[s]),
            }
            if scenario == "mixed":
                metrics["unrouted_row_nonzero"] = ye[3].abs().max().item()
            results[f"{scenario}/{kernel}"] = metrics

    # full MoE block through transformers' experts dispatch: real router (in-place
    # normalised topk), `torch_ep_eager` registered name, 1/ep grad-scale hook
    block = _build_block()
    ref = copy.deepcopy(block)
    ref.experts.config._experts_implementation = "eager"
    ep = copy.deepcopy(block)
    ep.experts.config = copy.copy(ep.experts.config)
    ep.experts.config._experts_implementation = "torch_ep_eager"
    s = _shard_experts(ep.experts, rank, world_size)
    ExpertParallelPlugin._register_expert_grad_scale(ep, world_size)

    x = torch.randn(1, T, H, generator=torch.Generator().manual_seed(200 + rank))
    xr, xe = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    yr, ye = ref(xr), ep(xe)
    gout = torch.randn(yr.shape, generator=torch.Generator().manual_seed(9 + rank))
    (yr * gout).sum().backward()
    (ye * gout).sum().backward()
    ref_gu, ref_dn = (
        ref.experts.gate_up_proj.grad.clone(),
        ref.experts.down_proj.grad.clone(),
    )
    dist.all_reduce(ref_gu, op=dist.ReduceOp.AVG)
    dist.all_reduce(ref_dn, op=dist.ReduceOp.AVG)
    results["block/torch_ep_eager"] = {
        "fwd": _max_diff(ye, yr),
        "dx": _max_diff(xe.grad, xr.grad),
        "d_router": _max_diff(ep.gate.weight.grad, ref.gate.weight.grad),
        "d_gate_up_mean": _max_diff(ep.experts.gate_up_proj.grad, ref_gu[s]),
        "d_down_mean": _max_diff(ep.experts.down_proj.grad, ref_dn[s]),
    }

    # frozen inputs and routing weights, trainable experts, rank 1 receives nothing: its
    # local output has no grad, yet it must still join the combine backward all-to-all
    x, idx, w = _routing(rank, "rank1_receives_zero")
    gout = torch.randn(T, H, generator=torch.Generator().manual_seed(11 + rank))
    ref = copy.deepcopy(full)
    yr = _ep_forward(ref, x, idx, w, kernel_name="eager", backend="torch")
    (yr * gout).sum().backward()
    ep = copy.deepcopy(full)
    s = _shard_experts(ep, rank, world_size)
    ye = _ep_forward(ep, x, idx, w, kernel_name="eager", backend="torch")
    ye_requires_grad = ye.requires_grad
    (ye * gout).sum().backward()
    ref_gu, ref_dn = ref.gate_up_proj.grad.clone(), ref.down_proj.grad.clone()
    dist.all_reduce(ref_gu)
    dist.all_reduce(ref_dn)
    results["frozen_inputs/eager"] = {
        "fwd": _max_diff(ye, yr),
        "no_grad_output": float(not ye_requires_grad),
        "d_gate_up": _max_diff(ep.gate_up_proj.grad, ref_gu[s]),
        "d_down": _max_diff(ep.down_proj.grad, ref_dn[s]),
    }

    # activations stay in their own dtype (no DeepEP-style bf16 round trip)
    x, idx, w = _routing(rank, "mixed")
    ep16 = copy.deepcopy(full)
    _shard_experts(ep16, rank, world_size)
    y16 = _ep_forward(
        ep16.to(torch.float64), x.double(), idx, w, kernel_name="eager", backend="torch"
    )
    results["dtype"] = {"dtype_changed": float(y16.dtype != torch.float64)}
    return results


class TestTorchEPParity:
    @pytest.mark.parametrize(
        "case",
        [
            "mixed/eager",
            "mixed/grouped_mm",
            "rank1_receives_zero/eager",
            "rank1_receives_zero/grouped_mm",
            "block/torch_ep_eager",
            "frozen_inputs/eager",
            "dtype",
        ],
    )
    def test_matches_single_rank_reference(self, ep2_results, case):
        for rank, res in _section(ep2_results, "parity").items():
            for metric, diff in res[case].items():
                assert diff <= 1e-5, f"rank {rank} {case} {metric}={diff}"


# --------------------------------------------------------------------------- #
# Selective checkpointing: recompute replays the forward's routing
# --------------------------------------------------------------------------- #


def _sac_checks(rank, world_size):
    import collections
    from types import SimpleNamespace

    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.checkpoint import checkpoint

    import axolotl.monkeypatch.selective_checkpointing as sac
    from axolotl.integrations.expert_parallel.experts_fn import register_all
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    watched = {
        "aten::topk",
        "axolotl::ep_all_to_all_single",
        "axolotl::ep_all_to_all_single_equal",
    }

    class _Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = func.name().split(".")[0]
            if name in watched:
                self.counts[name] += 1
            return func(*args, **kwargs)

    register_all()
    TD.set_ep_group(dist.group.WORLD)
    block = _build_block()
    block.experts.config._experts_implementation = "torch_ep_eager"
    _shard_experts(block.experts, rank, world_size)
    x = torch.randn(1, T, H, generator=torch.Generator().manual_seed(300 + rank))

    def run(context_fn):
        xi = x.clone().requires_grad_(True)
        block.zero_grad(set_to_none=True)
        fwd, bwd = _Count(), _Count()
        with fwd:
            if context_fn is None:
                y = block(xi)
            else:
                y = checkpoint(block, xi, use_reentrant=False, context_fn=context_fn)
        with bwd:
            (y.float() ** 2).sum().backward()
        grads = [xi.grad] + [p.grad for p in block.parameters()]
        return dict(fwd.counts), dict(bwd.counts), grads

    _, _, ref = run(None)
    out = {}
    for save_dispatch in (True, False):
        sac.clear_registered_saves()
        ExpertParallelPlugin._register_checkpoint_saves(
            SimpleNamespace(expert_parallel_save_dispatch=save_dispatch)
        )
        fwd, bwd, grads = run(sac.build_sac_context_fn(save=[]))
        out[f"save_dispatch={save_dispatch}"] = {
            "fwd": fwd,
            "bwd": bwd,
            "grad_diff": max(_max_diff(a, b) for a, b in zip(grads, ref, strict=True)),
        }
    sac.clear_registered_saves()
    return out


class TestTorchEPSelectiveCheckpointing:
    def test_recompute_reuses_routing_and_dispatch(self, ep2_results):
        for rank, res in _section(ep2_results, "sac").items():
            saved = res["save_dispatch=True"]
            unsaved = res["save_dispatch=False"]
            for label, r in res.items():
                assert r["grad_diff"] <= 1e-6, (rank, label, r)
                # routing runs once, in forward only (on CPU `.cpu()` of the split
                # counts is a no-op, so the host-copy save is covered by the policy tests)
                assert r["fwd"].get("aten::topk") == 1, (rank, label, r)
                assert "aten::topk" not in r["bwd"], (rank, label, r)
            # saved: backward issues only the three gradient all-to-alls (combine, recv_w,
            # recv_x); unsaved: recompute re-issues the forward's collectives first
            assert saved["bwd"].get("axolotl::ep_all_to_all_single") == 3, (rank, saved)
            assert "axolotl::ep_all_to_all_single_equal" not in saved["bwd"], (
                rank,
                saved,
            )
            assert unsaved["bwd"].get("axolotl::ep_all_to_all_single", 0) > 3, (
                rank,
                unsaved,
            )


# --------------------------------------------------------------------------- #
# Chunked, overlapped dispatch (expert_parallel_dispatch_chunks > 1)
# --------------------------------------------------------------------------- #


def _chunked_checks(rank, world_size):
    """Chunked vs unchunked EP on the same shards: ``{case: {metric: max_abs_diff}}`` plus
    per-forward collective / host-copy counts and SAC recompute parity."""
    import collections
    from types import SimpleNamespace

    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils.checkpoint import checkpoint

    import axolotl.monkeypatch.selective_checkpointing as sac
    from axolotl.integrations.expert_parallel import experts_fn
    from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

    TD.set_ep_group(dist.group.WORLD)
    full = _build_experts()
    shard = copy.deepcopy(full)
    _shard_experts(shard, rank, world_size)

    def run(chunks, x, idx, w, kernel, grad_inputs=True):
        experts_fn.set_dispatch_chunks(chunks)
        try:
            ep = copy.deepcopy(shard)
            xe = x.clone().requires_grad_(grad_inputs)
            we = w.clone().requires_grad_(grad_inputs)
            y = experts_fn._ep_forward(
                ep, xe, idx, we, kernel_name=kernel, backend="torch"
            )
            gout = torch.randn(
                y.shape, generator=torch.Generator().manual_seed(17 + rank)
            )
            (y * gout).sum().backward()
            return y, xe.grad, we.grad, ep.gate_up_proj.grad, ep.down_proj.grad
        finally:
            experts_fn.set_dispatch_chunks(1)

    names = ("fwd", "dx", "dw", "d_gate_up", "d_down")
    results = {}

    def diffs(got, ref, skip=()):
        # a rank that receives no rows has no expert grads in either run
        return {
            n: 0.0 if a is None and b is None else _max_diff(a, b)
            for n, a, b in zip(names, got, ref, strict=True)
            if n not in skip
        }

    # 13 tokens: not divisible by 2 or 3
    for scenario in ("mixed", "rank1_receives_zero"):
        for kernel in ("eager", "grouped_mm"):
            x, idx, w = _routing(rank, scenario, num_tokens=13)
            ref = run(1, x, idx, w, kernel)
            for chunks in (2, 3):
                got = run(chunks, x, idx, w, kernel)
                results[f"{scenario}/{kernel}/chunks={chunks}"] = diffs(got, ref)

    # rank 0 has 2 tokens (two empty chunks at chunks=3), rank 1 has 7
    x, idx, w = _routing(rank, "random", num_tokens=2 if rank == 0 else 7)
    ref = run(1, x, idx, w, "eager")
    got = run(3, x, idx, w, "eager")
    results["empty_chunks/eager"] = diffs(got, ref)

    # frozen inputs and routing weights, rank 1 receives nothing in any chunk
    x, idx, w = _routing(rank, "rank1_receives_zero", num_tokens=13)
    ref = run(1, x, idx, w, "eager", grad_inputs=False)
    got = run(3, x, idx, w, "eager", grad_inputs=False)
    results["frozen_inputs/eager"] = diffs(got, ref, skip=("dx", "dw"))

    # the kernel must receive waited plain tensors: an opaque (Triton/CuTe) kernel does not
    # go through the AsyncCollectiveTensor dispatch that would otherwise trigger the wait
    import torch.distributed._functional_collectives as funcol

    seen = []

    def opaque_kernel(recv_x, recv_idx, recv_w):
        seen.extend(type(t) for t in (recv_x, recv_idx, recv_w))
        return recv_x * recv_w.sum(dim=-1, keepdim=True).to(recv_x.dtype)

    x, idx, w = _routing(rank, "mixed", num_tokens=13)
    TD.dispatch_chunked_forward(
        x,
        idx,
        w,
        opaque_kernel,
        num_local_experts=E_LOCAL,
        group=dist.group.WORLD,
        chunks=3,
    )
    results["kernel_input_types"] = {
        "num": len(seen),
        "async": sum(issubclass(t, funcol.AsyncCollectiveTensor) for t in seen),
    }

    # one count exchange and one device->host copy per forward, whatever the chunk count
    counts = collections.Counter()
    real_equal, real_cpu = TD.all_to_all_single_equal, torch.Tensor.cpu

    def counting_equal(*args, **kwargs):
        counts["equal_a2a"] += 1
        return real_equal(*args, **kwargs)

    def counting_cpu(self, *args, **kwargs):
        counts["cpu"] += 1
        return real_cpu(self, *args, **kwargs)

    x, idx, w = _routing(rank, "mixed", num_tokens=13)
    TD.all_to_all_single_equal = counting_equal
    torch.Tensor.cpu = counting_cpu
    experts_fn.set_dispatch_chunks(3)
    try:
        experts_fn._ep_forward(
            copy.deepcopy(shard), x, idx, w, kernel_name="eager", backend="torch"
        )
    finally:
        TD.all_to_all_single_equal = real_equal
        torch.Tensor.cpu = real_cpu
        experts_fn.set_dispatch_chunks(1)
    results["sync_counts"] = dict(counts)

    # SAC recompute replays the chunked forward's routing and (optionally) its collectives
    watched = {"aten::topk", "_c10d_functional::all_to_all_single"}

    class _Count(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.counts = collections.Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func.name().split(".")[0]
            if name in watched:
                self.counts[name] += 1
            return func(*args, **(kwargs or {}))

    experts_fn.register_all()
    block = _build_block()
    block.experts.config._experts_implementation = "torch_ep_eager"
    _shard_experts(block.experts, rank, world_size)
    xb = torch.randn(1, 13, H, generator=torch.Generator().manual_seed(400 + rank))

    def run_block(chunks, context_fn):
        experts_fn.set_dispatch_chunks(chunks)
        try:
            xi = xb.clone().requires_grad_(True)
            block.zero_grad(set_to_none=True)
            bwd = _Count()
            if context_fn is None:
                y = block(xi)
            else:
                y = checkpoint(block, xi, use_reentrant=False, context_fn=context_fn)
            with bwd:
                (y.float() ** 2).sum().backward()
            grads = [xi.grad] + [p.grad for p in block.parameters()]
            return dict(bwd.counts), grads
        finally:
            experts_fn.set_dispatch_chunks(1)

    _, ref_grads = run_block(3, None)
    for save_dispatch in (True, False):
        sac.clear_registered_saves()
        ExpertParallelPlugin._register_checkpoint_saves(
            SimpleNamespace(expert_parallel_save_dispatch=save_dispatch)
        )
        bwd, grads = run_block(3, sac.build_sac_context_fn(save=[]))
        results[f"sac/save_dispatch={save_dispatch}"] = {
            "bwd": bwd,
            "grad_diff": max(
                _max_diff(a, b) for a, b in zip(grads, ref_grads, strict=True)
            ),
        }
    sac.clear_registered_saves()
    return results


class TestTorchEPChunkedDispatch:
    @pytest.mark.parametrize(
        "case",
        [
            f"{scenario}/{kernel}/chunks={chunks}"
            for scenario in ("mixed", "rank1_receives_zero")
            for kernel in ("eager", "grouped_mm")
            for chunks in (2, 3)
        ]
        + ["empty_chunks/eager", "frozen_inputs/eager"],
    )
    def test_matches_unchunked(self, ep2_results, case):
        for rank, res in _section(ep2_results, "chunked").items():
            for metric, diff in res[case].items():
                assert diff <= 1e-5, f"rank {rank} {case} {metric}={diff}"

    def test_kernel_receives_waited_tensors(self, ep2_results):
        for rank, res in _section(ep2_results, "chunked").items():
            assert res["kernel_input_types"] == {"num": 9, "async": 0}, (rank, res)

    def test_one_count_exchange_and_host_copy_per_forward(self, ep2_results):
        for rank, res in _section(ep2_results, "chunked").items():
            assert res["sync_counts"] == {"equal_a2a": 1, "cpu": 1}, (rank, res)

    def test_selective_checkpointing_recompute(self, ep2_results):
        for rank, res in _section(ep2_results, "chunked").items():
            saved = res["sac/save_dispatch=True"]
            unsaved = res["sac/save_dispatch=False"]
            for r in (saved, unsaved):
                assert r["grad_diff"] <= 1e-6, (rank, r)
                assert "aten::topk" not in r["bwd"], (rank, r)
            # saved: 3 chunks x 3 gradient all-to-alls (combine, recv_w, recv_x)
            a2a = "_c10d_functional::all_to_all_single"
            assert saved["bwd"].get(a2a) == 9, (rank, saved)
            assert unsaved["bwd"].get(a2a, 0) > 9, (rank, unsaved)

    def test_single_rank_ignores_chunks(self):
        from axolotl.integrations.expert_parallel import experts_fn

        experts = _build_experts()
        x, idx, w = _routing(0, "mixed", num_tokens=5)
        ref = experts_fn._ep_forward(
            experts, x, idx, w, kernel_name="eager", backend="torch"
        )
        experts_fn.set_dispatch_chunks(3)
        try:
            got = experts_fn._ep_forward(
                experts, x, idx, w, kernel_name="eager", backend="torch"
            )
        finally:
            experts_fn.set_dispatch_chunks(1)
        assert torch.equal(got, ref)


class TestChunkBounds:
    @pytest.mark.parametrize("num_tokens,chunks", [(13, 3), (12, 2), (2, 3), (0, 2)])
    def test_cover_tokens_contiguously(self, num_tokens, chunks):
        bounds = TD.chunk_bounds(num_tokens, chunks)
        assert len(bounds) == chunks
        assert bounds[0][0] == 0 and bounds[-1][1] == num_tokens
        assert all(a[1] == b[0] for a, b in zip(bounds, bounds[1:], strict=False))
