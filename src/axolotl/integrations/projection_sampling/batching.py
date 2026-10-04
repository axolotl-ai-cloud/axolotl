"""Combine independent chains and their proposal batches into backend requests."""

import hashlib
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .backend import SamplingBackend


def row_seed(seed: int, index: int) -> int:
    """Give each chain an RNG independent of completion and batching order."""
    return int.from_bytes(
        hashlib.sha256(f"{seed}:{index}".encode()).digest()[:8], "big"
    )


@dataclass
class Request:
    method: str
    args: tuple
    future: Future


class QueuedBackend(SamplingBackend):
    """Submit inference to the owner thread while each chain retains its state."""

    def __init__(self, backend, requests, cancelled, futures, lock):
        self.tokenizer = backend.tokenizer
        self.eos_token_ids = backend.eos_token_ids
        self.requests = requests
        self.cancelled = cancelled
        self.futures = futures
        self.lock = lock

    @classmethod
    def from_config(cls, cfg, config):
        raise NotImplementedError

    def call(self, method, *args):
        future: Future = Future()
        with self.lock:
            if self.cancelled.is_set():
                raise RuntimeError("Batch inference was cancelled")
            self.futures.add(future)
            self.requests.put(Request(method, args, future))
        try:
            return future.result()
        finally:
            with self.lock:
                self.futures.discard(future)

    def sample(self, context, max_tokens):
        return self.sample_batch([context], [max_tokens])[0]

    def sample_batch(self, contexts, max_tokens):
        if len(contexts) != len(max_tokens):
            raise ValueError("Sampling batch contexts and budgets must align")
        return self.call("sample", contexts, max_tokens) if contexts else []

    def target_logprob(self, context, tokens):
        return self.target_logprob_batch([context], [tokens])[0]

    def target_logprob_batch(self, contexts, tokens):
        if len(contexts) != len(tokens):
            raise ValueError("Scoring batch contexts and continuations must align")
        return self.call("target_logprob", contexts, tokens) if contexts else []

    def proposal_logprob(self, context, tokens):
        return self.proposal_logprob_batch([context], [tokens])[0]

    def proposal_logprob_batch(self, contexts, tokens):
        if len(contexts) != len(tokens):
            raise ValueError("Scoring batch contexts and continuations must align")
        return self.call("proposal_logprob", contexts, tokens) if contexts else []

    def proposal_kl(self, target_context, proposal_context, tokens, positions):
        return self.call(
            "proposal_kl", target_context, proposal_context, tokens, positions
        )

    def close(self):
        pass


def run_ordered_batch(
    backend: SamplingBackend,
    work: list[Callable],
    *,
    seed: int,
    row_offset: int = 0,
    status: Callable | None = None,
    on_result: Callable | None = None,
) -> list[Any]:
    """Run row jobs concurrently; perform all model calls on the calling thread."""
    if not work:
        return []
    if type(backend).sample_batch_seeded is SamplingBackend.sample_batch_seeded:
        raise ValueError("Concurrent chains require independently seeded batching")
    requests: list[queue.Queue] = [queue.Queue() for _ in work]
    cancelled = threading.Event()
    lock = threading.Lock()
    futures: set[Future] = set()
    results: list[Any] = [None] * len(work)
    calls = [0] * len(work)
    active = list(range(len(work)))

    def worker(index, function):
        proxy = QueuedBackend(backend, requests[index], cancelled, futures, lock)
        try:
            requests[index].put(("done", function(proxy)))
        except BaseException as error:  # pylint: disable=broad-exception-caught
            requests[index].put(("error", error))

    executor = ThreadPoolExecutor(
        max_workers=len(work), thread_name_prefix="projection-sampling"
    )
    try:
        for index, function in enumerate(work):
            executor.submit(worker, index, function)
        while active:
            pending = []
            remaining = []
            for index in active:
                request = requests[index].get()
                if isinstance(request, Request):
                    pending.append((index, request))
                    remaining.append(index)
                elif request[0] == "error":
                    raise request[1]
                else:
                    results[index] = request[1]
                    if on_result is not None:
                        on_result(index, request[1])
            active = remaining
            for method in (
                "sample",
                "target_logprob",
                "proposal_logprob",
                "proposal_kl",
            ):
                selected = [
                    (index, request)
                    for index, request in pending
                    if request.method == method
                ]
                if not selected:
                    continue
                start = time.monotonic()
                values: list[Any]
                if method == "proposal_kl":
                    values = [
                        backend.proposal_kl(*request.args) for _, request in selected
                    ]
                    sizes = [1] * len(selected)
                    count = len(selected)
                else:
                    contexts = [
                        context
                        for _, request in selected
                        for context in request.args[0]
                    ]
                    continuations = [
                        value for _, request in selected for value in request.args[1]
                    ]
                    sizes = [len(request.args[0]) for _, request in selected]
                    count = len(contexts)
                    if method == "sample":
                        seeds = []
                        for (index, _), size in zip(selected, sizes, strict=True):
                            for call in range(calls[index], calls[index] + size):
                                seeds.append(
                                    int.from_bytes(
                                        hashlib.sha256(
                                            f"{seed}:{row_offset + index}:{call}".encode()
                                        ).digest()[:4],
                                        "big",
                                    )
                                )
                            calls[index] += size
                        values = backend.sample_batch_seeded(
                            contexts, continuations, seeds
                        )
                    else:
                        values = getattr(backend, method + "_batch")(
                            contexts, continuations
                        )
                    if len(values) != count:
                        raise ValueError(
                            "Backend returned a misaligned inference batch"
                        )
                offset = 0
                for (_, request), size in zip(selected, sizes, strict=True):
                    response = (
                        values[offset]
                        if method == "proposal_kl"
                        else values[offset : offset + size]
                    )
                    request.future.set_result(response)
                    offset += size
                if status is not None:
                    status(method, count, time.monotonic() - start, calls, results)
        return results
    finally:
        with lock:
            cancelled.set()
            for future in futures:
                if not future.done():
                    future.set_exception(RuntimeError("Batch inference stopped"))
        executor.shutdown(wait=True)
