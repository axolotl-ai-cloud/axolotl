"""GLM-5.2 DSA + context-parallel (CP>1) correctness under the ringmaster refactor.

GLM-DSA owns its own CP attention (compressed-KV all-gather + per-rank
``q_offset=rank*s_local``) and requires the sequence sharded CONTIGUOUSLY
(rank r owns ``[r*L, (r+1)*L)``). The ringmaster plugin only shards the batch for
it, so correctness hinges on ringmaster picking a contiguous shard layout.

Three levels of protection:
  * ``test_glm_dsa_guard_predicate`` — fast, no deps: the plugin's guard fires for
    ``use_glm_dsa_kernels``.
  * ``test_glm_dsa_configs_shard_contiguously`` — ringmaster auto_select yields a
    contiguous layout for realistic GLM configs (only odd/adversarial CP sizes
    would zigzag).
  * ``test_glm_dsa_cp_matches_single_gpu`` — end-to-end on a gloo group: ringmaster's
    real ``shard_batch`` feeding GLM's real ``cp_mla_attn`` reproduces single-GPU
    attention exactly under contiguous sharding, and a zigzag shard does NOT.
"""

from __future__ import annotations

import os
import socket
import sys
import types

import pytest
import torch.multiprocessing as mp

from axolotl.integrations.context_parallel import ContextParallelPlugin

# ---- GLM MLA dims (real KV_LORA_RANK=512, rope 64 -> 576) with tiny B/H/S ----
_KV_LORA_RANK = 512
_DQK = 576
_B, _H, _S, _T = 1, 4, 32, 8
_SCALE = 1.0 / (_DQK**0.5)


def test_glm_dsa_guard_predicate():
    """The contiguous-shard guard fires iff use_glm_dsa_kernels is set."""
    assert ContextParallelPlugin._glm_dsa_requires_contiguous(
        types.SimpleNamespace(use_glm_dsa_kernels=True)
    )
    assert not ContextParallelPlugin._glm_dsa_requires_contiguous(
        types.SimpleNamespace(use_glm_dsa_kernels=False)
    )
    assert not ContextParallelPlugin._glm_dsa_requires_contiguous(
        types.SimpleNamespace()
    )


def test_glm_dsa_configs_shard_contiguously():
    """For realistic GLM-5.2 configs (128 heads, even CP sizes) ringmaster resolves a
    contiguous shard layout — matching the layout GLM-DSA hard-requires. Only odd CP
    sizes coprime with the head count would zigzag (not a real GLM config)."""
    usp = pytest.importorskip("ringmaster.strategies.usp")
    cfg_mod = pytest.importorskip("ringmaster.config")
    AUTO, Backend = cfg_mod.AUTO, cfg_mod.Backend

    def shard_layout(cp_size, num_kv_heads, node):
        u, r, _ = usp.auto_select(
            total=cp_size,
            requested_backend=Backend.AUTO,
            requested_ulysses=AUTO,
            requested_ring=AUTO,
            num_kv_heads=num_kv_heads,
            intra_node_size=node,
        )
        # mirrors CPRuntime.shard_load_balance: zigzag only for pure ring + head_tail
        return "head_tail" if (u == 1 and r > 1) else "contiguous"

    for cp, kv, node in [
        (2, 128, 8),
        (4, 128, 8),
        (8, 128, 8),
        (16, 128, 8),
        (16, 128, 4),
        (32, 128, 8),
        (6, 128, 8),
        (4, 96, 8),
        (8, 64, 8),
    ]:
        assert shard_layout(cp, kv, node) == "contiguous", (cp, kv, node)

    # adversarial: odd CP size coprime with the (power-of-two) head count -> pure ring
    assert shard_layout(3, 128, 8) == "head_tail"


def _install_axolotl_shim():
    """Fake axolotl namespace so glm_dsa's relative imports resolve without running
    axolotl/__init__. No-op in a healthy env where real axolotl is already imported;
    only bites when the package import chain is unavailable."""
    base = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src", "axolotl")
    )
    for name, path in [
        ("axolotl", base),
        ("axolotl.integrations", base + "/integrations"),
        ("axolotl.integrations.kernels", base + "/integrations/kernels"),
        ("axolotl.integrations.kernels.libs", base + "/integrations/kernels/libs"),
    ]:
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m


def _make_global(dtype):
    """Deterministic global (single-GPU) synthetic MLA inputs; identical on every rank."""
    import torch

    g = torch.Generator().manual_seed(0)
    q_abs = torch.randn(_B, _H, _S, _DQK, generator=g, dtype=dtype)
    k_shared = torch.randn(_B, _S, _DQK, generator=g, dtype=dtype)
    topk = torch.empty(_B, _S, _T, dtype=torch.long)
    for i in range(_S):
        topk[:, i, :] = torch.randint(0, i + 1, (_B, _T), generator=g)
    return q_abs, k_shared, topk.to(torch.int32)


def _gloo_worker(rank, world, load_balance, port, q):
    """One CP rank: contiguous/zigzag-shard via ringmaster, run GLM's real cp_mla_attn,
    all-gather, and (rank 0) report max abs error vs single-GPU. Module-level so
    ``mp.get_context('spawn')`` can pickle it."""
    os.environ["GLM_DSA_DISABLE_GATHER"] = "1"  # force pure-torch dense path (CPU)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    _install_axolotl_shim()

    import torch
    import torch.distributed as dist
    from ringmaster.shard import shard_batch

    from axolotl.integrations.kernels.libs.glm_dsa.context_parallel import cp_mla_attn
    from axolotl.integrations.kernels.libs.glm_dsa.dispatch import (
        dense_masked_out_latent,
    )

    dtype = torch.float64
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        group = dist.group.WORLD
        q_abs, k_shared, topk = _make_global(dtype)

        # ringmaster's REAL shard_batch tells us which global positions this rank owns
        probe = {"input_ids": torch.arange(_S).unsqueeze(0).expand(_B, _S).contiguous()}
        sharded, _ = shard_batch(
            probe, cp_rank=rank, cp_size=world, load_balance=load_balance
        )
        idx = sharded["input_ids"][0].tolist()
        idx_t = torch.tensor(idx, dtype=torch.long)

        q_loc = q_abs[:, :, idx_t, :].contiguous()
        k_loc = k_shared[:, idx_t, :].contiguous()
        tk_loc = topk[:, idx_t, :].contiguous()

        out_loc = cp_mla_attn(q_loc, k_loc, tk_loc, _SCALE, group=group)  # real GLM CP

        gathered = [torch.empty_like(out_loc) for _ in range(world)]
        dist.all_gather(gathered, out_loc.contiguous())
        idx_all = [None] * world
        dist.all_gather_object(idx_all, idx)

        err = None
        if rank == 0:
            # reassemble to global order via owned indices, so ordering can never be
            # the cause of a mismatch — only the attention math (q_offset) can.
            recon = torch.zeros(_B, _H, _S, _KV_LORA_RANK, dtype=dtype)
            for r in range(world):
                recon[:, :, torch.tensor(idx_all[r]), :] = gathered[r]
            full = dense_masked_out_latent(q_abs, k_shared, topk, _SCALE, q_offset=0)
            err = (recon - full).abs().max().item()
        q.put((rank, err))
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_gloo(load_balance, world=4):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _find_free_port()
    procs = [
        ctx.Process(target=_gloo_worker, args=(r, world, load_balance, port, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=180) for _ in range(world)]
    for p in procs:
        p.join(timeout=20)
        assert p.exitcode == 0, f"worker exited with {p.exitcode}"
    return next(err for rank, err in results if rank == 0)


@pytest.mark.slow
def test_glm_dsa_cp_matches_single_gpu():
    """End-to-end: contiguous shard reproduces single-GPU attention to machine
    precision; zigzag (wrong layout) does not."""
    pytest.importorskip("ringmaster")
    pytest.importorskip("triton")  # glm_dsa attention module imports triton at load

    err_contig = _run_gloo("contiguous")
    assert err_contig < 1e-9, f"contiguous CP diverged from single-GPU: {err_contig}"

    err_zig = _run_gloo("head_tail")
    assert err_zig > 1e-6, (
        f"zigzag shard unexpectedly matched ({err_zig}); the contiguous requirement "
        "would then be untested"
    )
