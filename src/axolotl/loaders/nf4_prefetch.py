"""Bounded CPU lookahead for Transformers checkpoint conversion groups."""

import math
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import torch


class _DeferredWeight:
    """Retain a lazy materializer and its destination allocation size."""

    def __init__(self, materialize, nbytes):
        self.materialize = materialize
        self.nbytes = nbytes

    def __call__(self):
        return self.materialize()


@contextmanager
def prefetch_nf4_weights(memory_bytes: int):
    """Prefetch at most one upcoming conversion group's CPU inputs within a budget."""
    if not memory_bytes:
        yield
        return

    import transformers.core_model_loading as loading

    spawn = loading.spawn_materialize
    materialize = loading.WeightTransform.materialize_tensors
    progress = loading.tqdm
    ready: dict[int, dict[str, list[torch.Tensor]]] = {}
    pending: dict[int, Future] = {}

    def defer(pool, tensor, device=None, dtype=None, **kwargs):
        if pool is not None or torch.device(device or "cpu").type != "cpu":
            raise ValueError("NF4 prefetch requires deferred CPU checkpoint loading")
        shape = tensor.get_shape() if hasattr(tensor, "get_shape") else tensor.shape
        # Include a floating-point source allocation alongside its cast destination.
        itemsize = 8 + (torch.empty((), dtype=dtype).element_size() if dtype else 8)
        return _DeferredWeight(
            spawn(None, tensor, device, dtype, **kwargs), math.prod(shape) * itemsize
        )

    def consume(mapping):
        if id(mapping) in ready:
            return ready.pop(id(mapping))
        return materialize(mapping)

    def lookahead(items, executor):
        groups = list(items)
        for index, item in enumerate(groups):
            mapping = item[1]
            if id(mapping) in pending:
                ready[id(mapping)] = pending.pop(id(mapping)).result()
            if index + 1 < len(groups):
                upcoming = groups[index + 1][1]
                inputs = [
                    value
                    for values in upcoming.collected_tensors.values()
                    for value in values
                ]
                if inputs and all(
                    isinstance(value, _DeferredWeight) for value in inputs
                ):
                    size = sum(value.nbytes for value in inputs)
                    if size <= memory_bytes:
                        pending[id(upcoming)] = executor.submit(materialize, upcoming)
            try:
                yield item
            finally:
                ready.pop(id(mapping), None)

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="nf4-prefetch"
    ) as executor:

        def bounded_progress(items, *args, **kwargs):
            if kwargs.get("desc") == "Loading weights":
                items = lookahead(items, executor)
            return progress(items, *args, **kwargs)

        try:
            with (
                patch.object(loading, "spawn_materialize", defer),
                patch.object(loading.WeightTransform, "materialize_tensors", consume),
                patch.object(loading, "tqdm", bounded_progress),
            ):
                yield
        finally:
            for future in pending.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            pending.clear()
            ready.clear()
