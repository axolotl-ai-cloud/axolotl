"""Centralized mutable process-wide ScatterMoE runtime settings.

One place to see (and reset) everything that changes ScatterMoE behavior at runtime. It is applied
once per run from the resolved config by :func:`configure_scattermoe_runtime`
(``KernelsPlugin.pre_model_load``), which resets first so a long-lived multi-run process can't
inherit stale state from a previous config.

The per-module ``set_*`` functions in experts.py / grouped_train.py / chunked_bnb.py remain as thin
compatibility wrappers that delegate to :data:`RUNTIME` (kept for the existing call sites + tests);
the kernel code reads ``RUNTIME`` fields directly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# chunk size for the chunked bnb-4bit grouped path; 32 balances throughput on small-expert MoEs
# against bounding the bf16 transient on large-expert models.
DEFAULT_CHUNK = 32


@dataclass
class ScatterMoERuntime:
    """All mutable knobs that affect ScatterMoE runtime behavior."""

    # grouped fp4 MoE experts (experts.py): None = off (fused/eager paths unchanged) | "nvfp4".
    fp4_grouped_mode: str | None = None
    # base dX backward precision on sm120: True = fp8-read (fast, ~2% grad) | False = bf16-dequant.
    fp4_dx_prefer_fp8: bool = True
    # grouped base-GEMM backend (grouped_train.py): None/"auto" | marlin | cutlass | deepgemm | dequant.
    grouped_backend: str | None = None
    # bnb-4bit experts (chunked_bnb.py): 1-launch parallel_linear path (recompute-in-backward) vs chunked.
    bnb_fast: bool = True
    # chunked-dequant chunk size override; None -> DEFAULT_CHUNK.
    dequant_chunk_size: int | None = None
    # whether layer-level gradient checkpointing is active (skips the redundant per-chunk checkpoint).
    layer_gc_active: bool = False
    # persist the requantized mxfp4 weight in a module-level cache across steps. Safe on a persistent
    # single-device param, but under FSDP2 the gathered param is the FULL weight every step, so
    # caching it holds a full-model mxfp4 copy per rank (OOM). Disabled under FSDP: recompute per
    # forward (freed after each layer), bounding resident mxfp4 to one layer.
    mxfp4_cache_persist: bool = True

    def reset(self) -> None:
        for f in fields(self):
            setattr(self, f.name, f.default)


RUNTIME = ScatterMoERuntime()


def configure_scattermoe_runtime(cfg) -> None:
    """Apply ALL ScatterMoE runtime settings from a run config.

    Resets to defaults first so a long-lived process never inherits stale state, then maps the
    relevant config fields. ``cfg`` is the resolved config (DictDefault).
    """
    RUNTIME.reset()
    RUNTIME.fp4_grouped_mode = cfg.get("dsv4_fp4_grouped_mode")
    backend = cfg.get("moe_grouped_backend")
    RUNTIME.grouped_backend = str(backend).lower() if backend else None
    chunk = cfg.get("moe_dequant_chunk_size")
    RUNTIME.dequant_chunk_size = int(chunk) if chunk else None
    RUNTIME.layer_gc_active = bool(cfg.get("gradient_checkpointing"))
    fast = cfg.get("moe_bnb_fast")
    RUNTIME.bnb_fast = True if fast is None else bool(fast)
    # Cache only in the non-FSDP case; under FSDP a persistent mxfp4 cache holds a full-model copy
    # per rank (OOM).
    RUNTIME.mxfp4_cache_persist = not bool(cfg.get("fsdp_config"))
