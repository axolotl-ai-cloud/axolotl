"""Pinned output readiness, serialization, and background-write ownership."""

import gc
import weakref
from contextlib import nullcontext
from threading import Event

import pytest
import safetensors.torch
import torch

from axolotl.cli.utils import lora_merge


def copy_outputs(packed, scale):
    with lora_merge._ExpertOutputTransfer() as transfer:
        return transfer.copy(packed, scale)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_expert_output_copy_is_ready_and_serializable(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    packed = torch.arange(3 * 64 * 16).remainder(256).to(torch.uint8).reshape(3, 64, 16)
    scale = (
        torch.arange(3 * 64 * 2).remainder(16).to(torch.float8_e4m3fn).reshape(3, 64, 2)
    )
    if device == "cuda":
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            inputs = (packed.cuda(), scale.cuda())
            torch.cuda._sleep(10_000_000)  # pylint: disable=protected-access
            outputs = copy_outputs(*inputs)
    else:
        outputs = copy_outputs(packed, scale)
    for output, expected in zip(outputs, (packed, scale), strict=True):
        assert output.device.type == "cpu"
        assert output.is_pinned() == (device == "cuda")
        assert output.dtype == expected.dtype and output.shape == expected.shape
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
    tensors = {
        f"experts.{expert}.{name}": output[expert]
        for name, output in zip(("weight", "weight_scale"), outputs, strict=True)
        for expert in range(3)
    }
    path = tmp_path / "experts.safetensors"
    safetensors.torch.save_file(tensors, path)
    loaded = safetensors.torch.load_file(path)
    for key, tensor in tensors.items():
        assert torch.equal(loaded[key].view(torch.uint8), tensor.view(torch.uint8))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("write_fails", [False, True])
def test_write_queue_owns_output_views_until_write_finishes(
    tmp_path, monkeypatch, device, write_fails
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    entered, release = Event(), Event()
    original_write = lora_merge._write_shard

    def blocked_write(*args):
        entered.set()
        assert release.wait(timeout=20)
        if write_fails:
            raise OSError("write failed")
        return original_write(*args)

    monkeypatch.setattr(lora_merge, "_write_shard", blocked_write)
    packed, scale = copy_outputs(
        torch.arange(128, device=device).to(torch.uint8).reshape(2, 8, 8),
        torch.ones((2, 8, 2), device=device).to(torch.float8_e4m3fn),
    )
    first = packed[0]
    second = packed[1]
    refs = [weakref.ref(tensor) for tensor in (first, second)]
    path = tmp_path / "queued.safetensors"
    queue = lora_merge._ShardWriteQueue()
    try:
        queue.submit({"first": first, "second": second}, path, {}, True)
        assert entered.wait(timeout=20)
        del packed, scale, first, second
        assert all(ref() is not None for ref in refs)
        # Allocation while a write is blocked must not recycle its live storage.
        other, _ = copy_outputs(
            torch.full((2, 8, 8), 255, device=device, dtype=torch.uint8),
            torch.ones((2, 8, 2), device=device).to(torch.float8_e4m3fn),
        )
        other.fill_(255)
        release.set()
        if write_fails:
            with pytest.raises(OSError, match="write failed"):
                queue.drain()
        else:
            queue.drain()
            loaded = safetensors.torch.load_file(path)
            assert torch.equal(
                loaded["first"], torch.arange(64).to(torch.uint8).reshape(8, 8)
            )
            assert torch.equal(
                loaded["second"], torch.arange(64, 128).to(torch.uint8).reshape(8, 8)
            )
        assert queue._pending is None
        queue._pool.submit(lambda: None).result()
        gc.collect()
        assert all(ref() is None for ref in refs)
    finally:
        release.set()
        queue.close()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("fails", [False, True])
def test_consume_drains_output_transfers_on_return_and_error(
    monkeypatch, device, fails
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    writer = lora_merge._Nvfp4ExpertMergeWriter({}, {}, 1, device)
    outputs, sources = [], []

    def consume(_):
        for value in range(5):
            packed = torch.full((2, 32, 16), value, device=device, dtype=torch.uint8)
            scale = torch.full((2, 32, 2), value, device=device).to(torch.float8_e4m3fn)
            sources.extend((weakref.ref(packed), weakref.ref(scale)))
            if device == "cuda":
                torch.cuda._sleep(10_000_000)  # pylint: disable=protected-access
            outputs.append(writer._copy_outputs_to_cpu(packed, scale))
            assert len(writer._output_transfer.pending) <= 2
        if fails:
            raise ValueError("merge failed after copying outputs")
        return {}, {}, 0

    monkeypatch.setattr(writer, "_consume", consume)
    stream = torch.cuda.Stream() if device == "cuda" else None
    for _ in range(2):
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            if fails:
                with pytest.raises(ValueError, match="merge failed"):
                    writer.consume({})
            else:
                assert writer.consume({}) == ({}, {}, 0)
        assert not writer._output_transfer.pending
        for value, pair in enumerate(outputs):
            for output in pair:
                assert output.device.type == "cpu"
                assert output.is_pinned() == (device == "cuda")
                assert torch.all(output.float() == value)
        gc.collect()
        if device == "cuda":
            assert all(ref() is None for ref in sources)
        outputs.clear()
        sources.clear()


@pytest.mark.gpu
@pytest.mark.parametrize("failed_allocation", [1, 2])
def test_pinned_allocation_failure_falls_back_and_recovers(
    tmp_path, monkeypatch, caplog, failed_allocation
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    packed = torch.arange(128, device="cuda").to(torch.uint8).reshape(2, 8, 8)
    scale = torch.ones((2, 8, 2), device="cuda").to(torch.float8_e4m3fn)
    original_empty_like = torch.empty_like
    allocations, refs = [], []

    def allocate(tensor, **kwargs):
        allocations.append(kwargs)
        if len(allocations) == failed_allocation:
            raise RuntimeError("pinned allocation unavailable")
        output = original_empty_like(tensor, **kwargs)
        refs.append(weakref.ref(output))
        return output

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), lora_merge._ExpertOutputTransfer() as transfer:
        torch.cuda._sleep(10_000_000)  # pylint: disable=protected-access
        prior = transfer.copy(packed, scale)
        with monkeypatch.context() as context:
            context.setattr(torch, "empty_like", allocate)
            fallback = transfer.copy(packed, scale)
        assert len(allocations) == failed_allocation
        assert all(ref() is None for ref in refs)
        assert all(not tensor.is_pinned() for tensor in fallback)
        for actual, expected in zip(fallback, (packed, scale), strict=True):
            assert torch.equal(
                actual.view(torch.uint8), expected.cpu().view(torch.uint8)
            )
        recovered = transfer.copy(packed, scale)
        assert all(tensor.is_pinned() for tensor in recovered)
    assert not transfer.pending
    assert "using pageable copies" in caplog.text
    for pair in (prior, recovered):
        for actual, expected in zip(pair, fallback, strict=True):
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    path = tmp_path / "fallback.safetensors"
    safetensors.torch.save_file(
        dict(zip(("weight", "scale"), fallback, strict=True)), path
    )
    loaded = safetensors.torch.load_file(path)
    for name, expected in zip(("weight", "scale"), fallback, strict=True):
        assert torch.equal(loaded[name].view(torch.uint8), expected.view(torch.uint8))
