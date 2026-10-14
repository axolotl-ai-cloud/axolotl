"""Parallel, preallocated stack/cat for transformers' checkpoint weight converters.

transformers fuses per-expert checkpoint tensors into the runtime 3D parameters with
``MergeModulelist`` (``torch.stack``) and ``Concatenate`` (``torch.cat``), serially on the
loading thread. Each call allocates a fresh destination and copies on one core, so a
large MoE (tens of thousands of expert tensors, around a terabyte) loads at the page-fault
rate of a single thread. Copying into a preallocated destination from a thread pool keeps the same result
and runs at memory bandwidth.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# below this many bytes per output the serial op is cheaper than the thread handoff
MIN_PARALLEL_BYTES = 64 << 20


def _workers() -> int:
    env = os.environ.get("AXOLOTL_CONVERT_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(32, (os.cpu_count() or 8) // 2))


@lru_cache(maxsize=1)
def _pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=_workers(), thread_name_prefix="axolotl-convert"
    )


def _parallel_ok(tensors: list[torch.Tensor]) -> bool:
    if len(tensors) < 2 or any(t.device.type != "cpu" for t in tensors):
        return False
    return sum(t.numel() * t.element_size() for t in tensors) >= MIN_PARALLEL_BYTES


def parallel_stack(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not _parallel_ok(tensors):
        return torch.stack(tensors, dim=dim)
    first = tensors[0]
    shape = list(first.shape)
    shape.insert(dim, len(tensors))
    out = torch.empty(shape, dtype=first.dtype)
    futures = [
        _pool().submit(out.select(dim, i).copy_, t) for i, t in enumerate(tensors)
    ]
    for future in futures:
        future.result()
    return out


def parallel_cat(tensors: list[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if not _parallel_ok(tensors):
        return torch.cat(tensors, dim=dim)
    first = tensors[0]
    dim = dim % first.dim()
    shape = list(first.shape)
    shape[dim] = sum(t.shape[dim] for t in tensors)
    out = torch.empty(shape, dtype=first.dtype)
    futures = []
    offset = 0
    for t in tensors:
        n = t.shape[dim]
        futures.append(_pool().submit(out.narrow(dim, offset, n).copy_, t))
        offset += n
    for future in futures:
        future.result()
    return out


def patch_parallel_weight_conversion() -> bool:
    """Route the converters' stack/cat through the thread pool. Idempotent."""
    try:
        from transformers import core_model_loading as cml
    except ImportError:
        return False
    if getattr(cml, "_axolotl_parallel_convert", False):
        return True

    merge = getattr(cml, "MergeModulelist", None)
    concat = getattr(cml, "Concatenate", None)
    if merge is None or concat is None:
        return False

    orig_merge = merge.convert
    orig_concat = concat.convert

    @torch.no_grad()
    def merge_convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        input_size = len(input_dict)
        merged = {}
        for source_pattern in list(input_dict.keys()):
            tensors = input_dict.pop(source_pattern)
            target_pattern = self.get_target_pattern(
                input_size, source_pattern, target_patterns
            )
            if isinstance(tensors, torch.Tensor):
                merged[target_pattern] = tensors
            else:
                merged[target_pattern] = parallel_stack(tensors, dim=self.dim)
        return merged

    @torch.no_grad()
    def concat_convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        target_pattern = self.get_target_pattern(target_patterns)
        all_tensors = []
        for source_pattern in source_patterns:
            if source_pattern not in input_dict:
                continue
            tensors = input_dict.pop(source_pattern)
            if isinstance(tensors, list):
                all_tensors.extend(tensors)
            else:
                all_tensors.append(tensors)
        return {target_pattern: parallel_cat(all_tensors, dim=self.dim)}

    merge.convert = merge_convert
    concat.convert = concat_convert
    merge._axolotl_orig_convert = orig_merge
    concat._axolotl_orig_convert = orig_concat

    workers = os.environ.get("AXOLOTL_LOAD_WORKERS")
    if workers and int(workers) > getattr(cml, "GLOBAL_WORKERS", 4):
        cml.GLOBAL_WORKERS = int(workers)
        LOG.info("Raised transformers materialize workers to %s", workers)

    cml._axolotl_parallel_convert = True
    LOG.info(
        "Parallel weight conversion enabled (%d copy threads for stack/cat)", _workers()
    )
    return True
