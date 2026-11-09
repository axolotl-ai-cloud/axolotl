"""XOR/zstd/Adler-32 delta encoding for tensors too large to encode in one pass."""

import os
import zlib

import numpy as np
import zstandard

LARGE_TENSOR_BYTES = 32 << 20
CHUNK_BYTES = 16 << 20


def default_zstd_threads() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def encode_xor_delta_chunked(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    level: int = 1,
    threads: int | None = None,
    chunk_bytes: int = CHUNK_BYTES,
) -> tuple[np.ndarray, str] | None:
    """One zstd frame of ``current ^ previous`` and the Adler-32 of ``current``.

    Same contract as ``encode_xor_delta``: ``None`` when the bytes are identical,
    and the frame decompresses to the full XOR with its content size recorded.
    XOR runs chunk by chunk into one reused buffer, so the full delta is never
    materialized, and compression uses zstd's own worker threads. Unchanged
    leading bytes are only compressed once a changed byte shows up, so an
    unchanged tensor costs one read and no compression.
    """
    if current.shape != previous.shape or current.dtype != np.uint8:
        raise ValueError("XOR delta needs two flat uint8 arrays of the same size")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    size = current.nbytes
    buffer = np.empty(min(chunk_bytes, max(size, 1)), dtype=np.uint8)
    checksum = zlib.adler32(b"")
    compressor = None
    frames: list[bytes] = []
    unchanged = 0
    for start in range(0, size, chunk_bytes):
        chunk = current[start : start + chunk_bytes]
        delta = np.bitwise_xor(
            chunk, previous[start : start + chunk_bytes], out=buffer[: chunk.nbytes]
        )
        checksum = zlib.adler32(chunk, checksum)
        if compressor is None:
            if not delta.any():
                unchanged += chunk.nbytes
                continue
            # zstd's multithreaded output does not depend on the thread count (>= 1)
            compressor = zstandard.ZstdCompressor(
                level=level, threads=max(1, threads or default_zstd_threads())
            ).compressobj(size=size)
            zeros = np.zeros(min(chunk_bytes, max(unchanged, 1)), dtype=np.uint8)
            while unchanged:
                step = min(unchanged, zeros.nbytes)
                frames.append(compressor.compress(zeros[:step]))
                unchanged -= step
        frames.append(compressor.compress(delta))
    if compressor is None:
        return None
    frames.append(compressor.flush())
    return np.frombuffer(b"".join(frames), dtype=np.uint8), f"{checksum:08x}"
