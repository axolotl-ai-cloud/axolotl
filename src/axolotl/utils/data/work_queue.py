"""
Load-balanced multiprocess tokenization.

``Dataset.map(num_proc=N)`` gives each worker one contiguous shard, so a shard
holding the long examples leaves the rest of the pool idle. Workers here pull
chunks of row indices instead.
"""

import math
import multiprocessing
import os
import tempfile
from collections import deque
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor
from typing import Any, Iterator

import pyarrow as pa
from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from datasets.fingerprint import update_fingerprint
from datasets.utils import tqdm as hf_tqdm

from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MAX_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 8
# Enough chunks that a slow worker gets overtaken, few enough to amortize the IPC.
CHUNKS_PER_WORKER = 32
# Bounds submitted-but-uncollected chunks so workers cannot outrun the writer.
MAX_IN_FLIGHT_PER_WORKER = 4
# The first write fixes the Arrow schema, so it must see enough rows to infer
# types for columns that are empty in early rows.
WRITER_BATCH_SIZE = 1000

# Set once per worker by the pool initializer so tasks only ship row indices.
_strategy: PromptTokenizingStrategy | None = None
_dataset: Dataset | None = None


def _mp_context():
    """Prefer fork so the dataset and tokenizer reach workers without pickling."""
    if "fork" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _init_worker(strategy: PromptTokenizingStrategy, dataset: Dataset) -> None:
    global _strategy, _dataset  # noqa: PLW0603
    _strategy = strategy
    _dataset = dataset


def _tokenize_chunk(bounds: tuple[int, int]) -> dict[str, list]:
    if _strategy is None or _dataset is None:
        raise RuntimeError("tokenization worker was not initialized")
    start, end = bounds
    batch = _dataset[start:end]
    if _strategy.supports_batched:
        return _strategy.tokenize_prompt(batch) or {}

    columns = list(batch)
    rows = [
        _strategy.tokenize_prompt(dict(zip(columns, values, strict=True)))
        for values in zip(*batch.values(), strict=True)
    ]
    # A row that tokenizes to nothing must not take its chunk-mates with it.
    rows = [row for row in rows if row]

    if not rows:
        return {}

    keys = rows[0].keys()
    for row in rows[1:]:
        if row.keys() != keys:
            raise ValueError(
                f"tokenize_prompt returned inconsistent keys within rows "
                f"{start}-{end}: {sorted(keys)} then {sorted(row.keys())}. "
                "Every row must produce the same set of columns."
            )
    return {key: [row[key] for row in rows] for key in keys}


def _chunk_bounds(num_rows: int, num_proc: int) -> list[tuple[int, int]]:
    chunk_size = math.ceil(num_rows / (num_proc * CHUNKS_PER_WORKER))
    chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, chunk_size))
    return [
        (start, min(start + chunk_size, num_rows))
        for start in range(0, num_rows, chunk_size)
    ]


def _iter_results(
    executor: ProcessPoolExecutor, chunks: list[tuple[int, int]], num_proc: int
) -> Iterator[dict[str, list]]:
    """Yield chunk results in order, keeping only a bounded number in flight."""
    max_in_flight = max(1, num_proc * MAX_IN_FLIGHT_PER_WORKER)
    remaining = iter(chunks)
    pending: deque = deque()
    try:
        for bounds in remaining:
            pending.append(executor.submit(_tokenize_chunk, bounds))
            if len(pending) >= max_in_flight:
                break
        while pending:
            result = pending.popleft().result()
            next_bounds = next(remaining, None)
            if next_bounds is not None:
                pending.append(executor.submit(_tokenize_chunk, next_bounds))
            yield result
    except BrokenExecutor as exc:
        raise RuntimeError(
            "A tokenization worker died unexpectedly (possibly OOM-killed). "
            "Set dataset_num_proc: 1 to debug."
        ) from exc


def _num_rows(batch: dict[str, list]) -> int:
    return len(next(iter(batch.values()))) if batch else 0


def _extend(into: dict[str, list], batch: dict[str, list]) -> None:
    if not into:
        into.update({key: list(values) for key, values in batch.items()})
        return
    if into.keys() != batch.keys():
        raise ValueError(
            f"tokenize_prompt returned inconsistent keys across chunks: "
            f"{sorted(into)} then {sorted(batch)}. "
            "Every row must produce the same set of columns."
        )
    for key, values in batch.items():
        into[key].extend(values)


def tokenize_with_work_queue(
    prompt_tokenizer: PromptTokenizingStrategy,
    dataset: Dataset,
    num_proc: int,
    keep_in_memory: bool | None = False,
) -> Dataset:
    """Tokenize ``dataset`` across ``num_proc`` workers, dropping input columns.

    Equivalent to ``dataset.map(..., remove_columns=<all>)`` except that workers
    take the next chunk as they finish rather than owning a fixed shard. A hung
    worker is not detected; only a dead one.
    """
    fingerprint = update_fingerprint(
        dataset._fingerprint,
        "tokenize_with_work_queue",
        {
            "function": prompt_tokenizer.tokenize_prompt,
            "batched": prompt_tokenizer.supports_batched,
        },
    )
    # Matching ``map``: an in-memory dataset has no cache dir, so a cache path
    # would land in a per-session temp dir and never be reused.
    cache_file = (
        dataset._get_cache_file_path(fingerprint)
        if not keep_in_memory and dataset.cache_files
        else None
    )
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
        try:
            with (
                ProcessPoolExecutor(
                    num_proc,
                    mp_context=_mp_context(),
                    initializer=_init_worker,
                    initargs=(prompt_tokenizer, dataset),
                ) as executor,
                hf_tqdm(
                    total=len(dataset), desc="Tokenizing Prompts", unit=" examples"
                ) as pbar,
            ):
                buffered: dict[str, list] = {}
                written = 0
                for (start, end), batch in zip(
                    chunks, _iter_results(executor, chunks, num_proc), strict=True
                ):
                    if batch:
                        _extend(buffered, batch)
                    if _num_rows(buffered) >= WRITER_BATCH_SIZE:
                        writer.write_batch(buffered)
                        written += _num_rows(buffered)
                        buffered = {}
                    pbar.update(end - start)
                if buffered:
                    writer.write_batch(buffered)
                    written += _num_rows(buffered)
            if not written:
                raise ValueError(
                    "Tokenization produced no rows. Every example was dropped by "
                    "the prompt strategy; check the dataset and its `type:` config."
                )
            writer.finalize()
        finally:
            writer.close()
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
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_file, 0o666 & ~umask)
        return Dataset.from_file(cache_file, split=dataset.split)
    assert buf_writer is not None
    return Dataset.from_buffer(buf_writer.getvalue(), split=dataset.split)
