"""Time rank-0 assembly of one gathered FULL_STATE_DICT tensor: index (meshgrid) vs slice views.

Mirrors an EP=8 expert LoRA tensor of Beam-1.0-501B (5120 x 16384 fp32, 8 owner chunks) for both
expert-major rows (kind A) and interleaved experts (kind B). Single process; no collectives.

    python scripts/benchmark_fsdp2_assembly.py [--threads 1] [--repeats 3]
"""

import argparse
import time

import torch

from axolotl.monkeypatch.accelerate.fsdp2_checkpoint import (
    _boxes,
    _coordinates,
    _index,
    _write,
)


def chunks(kind, rows, columns, experts):
    local = 1
    shape = [rows, columns]
    if kind == "A":
        local_shape = [rows // experts, columns]
    else:
        local_shape = [rows, columns // experts]
    infos = []
    for expert in range(experts):
        owner = dict(offset=expert, local=local, total=experts, kind=kind)
        info = dict(
            shape=shape,
            local_shape=local_shape,
            offset=[0, 0],
            owner=owner,
            axes=None,
        )
        infos.append((expert, info))
    tensors = [torch.randn(local_shape) for _ in infos]
    return shape, infos, tensors


def index_assembly(shape, infos, tensors):
    full = torch.empty(shape)
    covered = torch.zeros(shape, dtype=torch.bool)
    for (_, info), tensor in zip(infos, tensors, strict=True):
        indices = _index(_coordinates(info))
        full[indices] = tensor
        covered[indices] = True
    assert covered.all()
    return full


def slice_assembly(shape, infos, tensors):
    full = torch.empty(shape)
    view, boxes = _boxes(infos)
    view = full.view(view)
    for (starts, lengths), tensor in zip(boxes, tensors, strict=True):
        destination = view
        for dim, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            destination = destination.narrow(dim, start, length)
        _write(destination, 0, tensor.reshape(-1))
    return full


def timed(function, *args, repeats):
    best, result = float("inf"), None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function(*args)
        best = min(best, time.perf_counter() - start)
    return best, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5120)
    parser.add_argument("--columns", type=int, default=16384)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for kind in ("A", "B"):
        shape, infos, tensors = chunks(kind, args.rows, args.columns, args.experts)
        old, expected = timed(
            index_assembly, shape, infos, tensors, repeats=args.repeats
        )
        new, actual = timed(slice_assembly, shape, infos, tensors, repeats=args.repeats)
        assert torch.equal(expected.view(torch.int32), actual.view(torch.int32))
        print(
            f"kind {kind} {shape} x{args.experts} chunks, {args.threads} thread(s): "
            f"index {old:.3f}s  slice {new:.3f}s  ({old / new:.0f}x)"
        )


if __name__ == "__main__":
    main()
