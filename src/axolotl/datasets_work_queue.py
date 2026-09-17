"""
Load-balanced multiprocess tokenization.

``Dataset.map(num_proc=N)`` hands each worker one contiguous shard, so a shard
that happens to hold the long examples leaves the rest of the pool idle. Here
workers pull small chunks of row indices from a shared pool instead, and the
results are streamed into an Arrow cache file the same way ``map`` does.
"""

import os
import tempfile
from typing import Any

import pyarrow as pa
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from datasets.fingerprint import update_fingerprint
from datasets.utils import tqdm as hf_tqdm
from multiprocess import Pool, TimeoutError

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MAX_CHUNK_SIZE = 1000
CHUNKS_PER_WORKER = 32
WORKER_POLL_SECONDS = 1.0

# Set once per worker by the pool initializer so tasks only ship row indices.
_strategy: PromptTokenizingStrategy | None = None
_dataset: Dataset | None = None


def _init_worker(strategy: PromptTokenizingStrategy, dataset: Dataset) -> None:
    global _strategy, _dataset  # noqa: PLW0603
    _strategy = strategy
    _dataset = dataset


def _tokenize_chunk(bounds: tuple[int, int]) -> dict[str, list]:
    assert _strategy is not None and _dataset is not None
    start, end = bounds
    batch = _dataset[start:end]
    if _strategy.supports_batched:
        return _strategy.tokenize_prompt(batch) or {}

    columns = list(batch)
    rows = [
        _strategy.tokenize_prompt(dict(zip(columns, values, strict=True)))
        for values in zip(*batch.values(), strict=True)
    ]
    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}


def _chunk_bounds(num_rows: int, num_proc: int) -> list[tuple[int, int]]:
    chunk_size = max(1, min(MAX_CHUNK_SIZE, num_rows // (num_proc * CHUNKS_PER_WORKER)))
    return [
        (start, min(start + chunk_size, num_rows))
        for start in range(0, num_rows, chunk_size)
    ]


def _iter_results(pool: Pool, chunks: list[tuple[int, int]]):
    """Yield results in order, failing fast instead of hanging if a worker is killed."""
    initial_pids = {proc.pid for proc in pool._pool}
    results = pool.imap(_tokenize_chunk, chunks)
    for _ in chunks:
        while True:
            try:
                yield results.next(timeout=WORKER_POLL_SECONDS)
                break
            except TimeoutError:
                if {proc.pid for proc in pool._pool} != initial_pids:
                    raise RuntimeError(
                        "A tokenization worker died unexpectedly (possibly OOM-killed). "
                        "Set dataset_num_proc: 1 to debug."
                    ) from None


def tokenize_with_work_queue(
    prompt_tokenizer: PromptTokenizingStrategy,
    dataset: Dataset,
    num_proc: int,
    keep_in_memory: bool | None = False,
) -> Dataset:
    """Tokenize ``dataset`` with ``prompt_tokenizer`` across ``num_proc`` workers.

    Equivalent to ``dataset.map(prompt_tokenizer.tokenize_prompt, num_proc=...,
    remove_columns=<all>)`` but with dynamic scheduling: workers take the next
    chunk as soon as they finish the current one.
    """
    fingerprint = update_fingerprint(
        dataset._fingerprint,
        "tokenize_with_work_queue",
        {
            "function": prompt_tokenizer.tokenize_prompt,
            "batched": prompt_tokenizer.supports_batched,
        },
    )
    cache_file = None if keep_in_memory else dataset._get_cache_file_path(fingerprint)
    if cache_file and os.path.exists(cache_file):
        LOG.info(f"Loading cached tokenized dataset at {cache_file}")
        return Dataset.from_file(cache_file, split=dataset.split)

    writer_kwargs: dict[str, Any] = {"fingerprint": fingerprint}
    buf_writer = None
    tmp_path = None
    if cache_file:
        LOG.info(f"Caching tokenized dataset at {cache_file}")
        cache_dir = os.path.dirname(cache_file)
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=cache_dir, delete=False) as tmp:
            tmp_path = tmp.name
        writer_kwargs["path"] = tmp_path
    else:
        buf_writer = pa.BufferOutputStream()
        writer_kwargs["stream"] = buf_writer

    chunks = _chunk_bounds(len(dataset), num_proc)
    LOG.info(
        f"Tokenizing {len(dataset)} examples with {num_proc} workers "
        f"({len(chunks)} chunks)"
    )

    # Forking after the Rust tokenizer's thread pool has started can deadlock.
    prev_tokenizers_parallelism = os.environ.get("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        writer = ArrowWriter(**writer_kwargs)
        with (
            Pool(
                num_proc,
                initializer=_init_worker,
                initargs=(prompt_tokenizer, dataset),
            ) as pool,
            hf_tqdm(
                total=len(dataset), desc="Tokenizing Prompts", unit=" examples"
            ) as pbar,
        ):
            for (start, end), batch in zip(
                chunks, _iter_results(pool, chunks), strict=True
            ):
                if batch:
                    writer.write_batch(batch)
                pbar.update(end - start)
        writer.finalize()
    except BaseException:
        if tmp_path:
            os.remove(tmp_path)
        raise
    finally:
        if prev_tokenizers_parallelism is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev_tokenizers_parallelism

    if cache_file and tmp_path:
        os.replace(tmp_path, cache_file)
        return Dataset.from_file(cache_file, split=dataset.split)
    assert buf_writer is not None
    return Dataset.from_buffer(buf_writer.getvalue(), split=dataset.split)
