"""Multi-rank sharded LoRA gather must restore full param.data shape on the compute stream, avoiding the ToCopyBackward [0] shape mismatch."""

from __future__ import annotations

import os
import sys

import pytest

pytestmark = pytest.mark.gpu


# ---------------------------------------------------------------------------
# mp.spawn worker bodies (must be top-level so the spawn fork can pickle them)
# ---------------------------------------------------------------------------


def _build_tiny_lora_model_cpu():
    """Tiny CPU LoRA-wrapped Linear stack; bf16 base + fp32 lora factors reproduces the mixed-dtype region split."""
    import torch
    from torch import nn

    torch.manual_seed(13)

    class _LoraWrappedLinear(nn.Module):
        """Mimics PEFT's LoRA-wrapped Linear so chunk-manager offload sees lora_A/lora_B as separate slots in the same chunk."""

        def __init__(self, in_dim: int, out_dim: int, r: int) -> None:
            super().__init__()
            self.base_layer = nn.Linear(in_dim, out_dim, bias=False).to(torch.bfloat16)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(in_dim, r, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(r, out_dim, bias=False)})
            # Mirror PEFT's autocast_adapter_dtype default: upcast LoRA
            # factor weights to fp32 even when the base is bf16. This
            # produces the mixed-dtype regions in materialize_offload.
            self.lora_A["default"].weight.data = self.lora_A["default"].weight.data.to(
                torch.float32
            )
            self.lora_B["default"].weight.data = self.lora_B["default"].weight.data.to(
                torch.float32
            )

        def forward(self, x):  # noqa: D401 — small forward
            base = self.base_layer(x)
            lora_out = self.lora_B["default"](
                self.lora_A["default"](x.to(torch.float32))
            )
            return base + lora_out.to(base.dtype)

    block = _LoraWrappedLinear(in_dim=8, out_dim=8, r=2)
    model = nn.Module()
    model.h = nn.ModuleList([block])  # type: ignore[attr-defined]
    return model


def _worker_sharded_lora_gather_rebinds(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """2-rank gloo: after sharded gather, every LoRA factor param.data must have its full shape back, not the [0] placeholder."""
    import contextlib
    import os as _os

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.types import BlockId, ChunkId, ParamId

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29605")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-sharded-lora",
        rank=rank,
        world_size=world_size,
    )

    try:
        model = _build_tiny_lora_model_cpu()

        # Layout: one block, all params in one chunk (large S_chunk).
        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14  # 16 KB — fits the tiny model
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        # Snapshot pre-offload shapes so the rebind invariant can be asserted post-gather.
        pre_shapes = {str(name): tuple(p.shape) for name, p in model.named_parameters()}
        pre_data = {
            str(name): p.detach().clone().cpu() for name, p in model.named_parameters()
        }

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )

        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            if "gloo" in str(exc).lower():
                _os.makedirs(tmpdir, exist_ok=True)
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # Post-offload invariant: every offloaded LoRA param.data is
        # the [0] empty placeholder. This is what the autograd source-
        # shape derivation would record if the cast op recorded against
        # this state — the bug the rebind is designed to prevent.
        for name, p in model.named_parameters():
            if name in {"h.0.base_layer.weight"}:
                continue  # base weight may or may not be offloaded
            assert tuple(p.shape) == (0,), (
                f"rank {rank}: post-materialize_offload, '{name}' should "
                f"be the [0] empty placeholder, got shape {tuple(p.shape)}"
            )

        # Sharded gather collective: after this, every LoRA factor's param.data must reflect its real shape so autograd records the correct source-shape.
        try:
            mgr.gather(ChunkId(0))
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "nccl" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        # Every LoRA-factor param.data must hold its real shape after the sharded gather; pins the multi-GPU failure mode at unit scope.
        for name, p in model.named_parameters():
            assert tuple(p.shape) == pre_shapes[str(name)], (
                f"rank {rank}: post-gather, '{name}' shape "
                f"{tuple(p.shape)} != pre-offload {pre_shapes[str(name)]}; "
                "the sharded gather did not restore the real shape, so "
                "any autograd source-shape derivation against this state "
                "would record [0] and backward would fail with "
                "'ToCopyBackward0 ... shape compatible with [0]'."
            )

        # Gathered bytes must match the pre-offload snapshot; ensures the routing did not perturb the byte layout.
        for name, p in model.named_parameters():
            snap = pre_data[str(name)]
            assert torch.allclose(p.data.cpu().float(), snap.float(), atol=0.0), (
                f"rank {rank}: post-gather '{name}' bytes diverge from "
                "pre-offload snapshot."
            )

        mgr.uninstall()
        host.close()

    finally:
        with contextlib.suppress(Exception):
            dist.barrier()
        dist.destroy_process_group()


def _worker_sharded_lora_ensure_chunks_resident(
    rank: int, world_size: int, tmpdir: str
) -> None:
    """2-rank gloo: Scheduler.ensure_chunks_resident must restore LoRA-factor shape on the compute stream (no prefetch-stream hop)."""
    import contextlib
    import os as _os

    import torch
    import torch.distributed as dist

    from axolotl.integrations.protrain.chunk.buffer_pool import BufferPool
    from axolotl.integrations.protrain.chunk.layout import build_layout
    from axolotl.integrations.protrain.chunk.manager import ChunkManager
    from axolotl.integrations.protrain.chunk.pinned_alloc import PinnedHostMemory
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler
    from axolotl.integrations.protrain.types import (
        BlockId,
        BlockMode,
        ChunkId,
        ParamId,
    )

    _os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    _os.environ.setdefault("MASTER_PORT", "29607")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{tmpdir}/rendezvous-sharded-lora-ecr",
        rank=rank,
        world_size=world_size,
    )

    try:
        model = _build_tiny_lora_model_cpu()

        block_spans: dict = {}
        for name, _p in model.named_parameters():
            block_spans.setdefault(BlockId(0), []).append(ParamId(name))  # type: ignore[index]
        exec_order = [ParamId(n) for n, _ in model.named_parameters()]
        S_chunk = 1 << 14
        layout = build_layout(model, exec_order, S_chunk, block_spans)

        pre_shapes = {str(name): tuple(p.shape) for name, p in model.named_parameters()}

        host = PinnedHostMemory(n_buffer=1, S_chunk=layout.S_chunk)
        pool = BufferPool(
            n_buffer=1,
            S_chunk=layout.S_chunk,
            pinned_host=host,
            device=torch.device("cpu"),
        )
        mgr = ChunkManager(
            model=model,
            layout=layout,
            n_persist=0,
            buffer_pool=pool,
            cpu_optim=None,
            gpu_optim=None,
            device=torch.device("cpu"),
            world_size=world_size,
            rank=rank,
            zero3_shard=True,
        )

        try:
            mgr.materialize_offload()
        except RuntimeError as exc:
            if "gloo" in str(exc).lower():
                _os.makedirs(tmpdir, exist_ok=True)
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-unsupported: {exc}\n")
                return
            raise

        # Build a Scheduler. The block_map is needed by Scheduler's
        # constructor; for this test we only exercise
        # ``ensure_chunks_resident`` which doesn't actually consult
        # block-mode keys, so OFFLOAD-everywhere is fine.
        block_map = {BlockId(0): BlockMode.OFFLOAD}
        # effective_h2d_bps / effective_d2h_bps are telemetry-only here; ensure_chunks_resident does not consult them.
        scheduler = Scheduler(
            chunk_manager=mgr,
            block_map=block_map,
            layout=layout,
            effective_h2d_bps=1.0e10,
            effective_d2h_bps=1.0e10,
        )

        # ensure_chunks_resident routes synchronously through the chunk manager so the rebind is inline.
        try:
            scheduler.ensure_chunks_resident([ChunkId(0)])
        except RuntimeError as exc:
            if "not implemented" in str(exc).lower() or "nccl" in str(exc).lower():
                with open(_os.path.join(tmpdir, f"rank{rank}.skip"), "w") as f:
                    f.write(f"gloo-collective-unsupported: {exc}\n")
                return
            raise

        # The container-hook contract: after ensure_chunks_resident
        # returns, every LoRA factor param has its real shape and the
        # autograd source-shape derivation step (the
        # ``ToCopyBackward0`` source-shape recorder in the multi-GPU
        # failure mode) reads the correct shape.
        for name, p in model.named_parameters():
            assert tuple(p.shape) == pre_shapes[str(name)], (
                f"rank {rank}: after ensure_chunks_resident, '{name}' "
                f"shape {tuple(p.shape)} != pre-offload "
                f"{pre_shapes[str(name)]}. The Scheduler did not synchronously "
                "rebind the LoRA factor's param.data — autograd would "
                "record [0] as the source shape and backward fails."
            )

        # Second call must hit the _active_chunks fast path without behavior change (idempotency contract).
        scheduler.ensure_chunks_resident([ChunkId(0)])
        for name, p in model.named_parameters():
            assert tuple(p.shape) == pre_shapes[str(name)], (
                f"rank {rank}: idempotent ensure_chunks_resident "
                f"second call broke param '{name}' shape: "
                f"{tuple(p.shape)} != {pre_shapes[str(name)]}"
            )

        mgr.uninstall()
        host.close()

    finally:
        with contextlib.suppress(Exception):
            dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Skip-detection helper (mirrors test_chunk_manager_distributed.py pattern)
# ---------------------------------------------------------------------------


def _check_skip_files(tmpdir: str, world_size: int) -> None:
    """If any worker dropped a ``rank{N}.skip`` file, surface as pytest.skip."""
    for r in range(world_size):
        skip_path = os.path.join(tmpdir, f"rank{r}.skip")
        if os.path.exists(skip_path):
            with open(skip_path) as f:
                pytest.skip(f"sharded-lora gloo worker skipped: {f.read().strip()}")


# ---------------------------------------------------------------------------
# Test bodies (parent-process spawners)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sharded_lora_gather_rebinds_param_data_2rank(tmp_path) -> None:
    """Sharded gather across 2 ranks must restore every LoRA factor's full shape, not the [0] placeholder."""
    import torch.multiprocessing as mp

    if sys.platform != "linux":
        pytest.skip("mp.spawn / gloo round-trip is linux-only in CI")

    world_size = 2
    mp.spawn(
        _worker_sharded_lora_gather_rebinds,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    _check_skip_files(str(tmp_path), world_size)


@pytest.mark.slow
def test_sharded_lora_ensure_chunks_resident_2rank(tmp_path) -> None:
    """Same sharded gather invariant driven via Scheduler.ensure_chunks_resident; routing must be synchronous on the compute stream."""
    import torch.multiprocessing as mp

    if sys.platform != "linux":
        pytest.skip("mp.spawn / gloo round-trip is linux-only in CI")

    world_size = 2
    mp.spawn(
        _worker_sharded_lora_ensure_chunks_resident,
        args=(world_size, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    _check_skip_files(str(tmp_path), world_size)
