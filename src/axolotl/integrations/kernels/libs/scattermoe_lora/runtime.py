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

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# chunk size for the chunked bnb-4bit grouped path; 32 balances throughput on small-expert MoEs
# against bounding the bf16 transient on large-expert models.
DEFAULT_CHUNK = 32


@dataclass
class ScatterMoERuntime:
    """All mutable knobs that affect ScatterMoE runtime behavior."""

    # grouped fp4 MoE experts (experts.py): None = off (fused/eager paths unchanged) | "nvfp4".
    grouped_mode: str | None = None
    # base dX backward precision on sm120: True = fp8-read (fast, ~2% grad) | False = bf16-dequant.
    fp4_dx_prefer_fp8: bool = True
    # transposed-Marlin backward dX (marlin backend): W4A16 grouped GEMM on a requantized W^T pack.
    # OPT-IN; cfg knob = moe_marlin_dx. False = off (dequant dX), True = GPU-resident packs
    # (fastest, ~doubles expert residency).
    marlin_dx: bool = False
    # opt-in full-grid Triton autotune for the s2s kernels (single-process only; winners are
    # reported to telemetry). Off = deterministic table-pruned configs.
    kernel_autotune: bool = False
    # fuse the gated activation into the gate_up LoRA-B GEMM epilogue (kills the separate swiglu
    # pass's [Mt,2I] read). OPT-IN; cfg knob = moe_swiglu_epilogue.
    swiglu_epilogue: bool = False
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
    # expert params are re-gathered every forward (FSDP / DeepSpeed) rather than rank-resident, so
    # the marlin prep must not pin or free them. Only consulted when world_size > 1.
    experts_gathered: bool = True

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
    RUNTIME.grouped_mode = cfg.get("moe_grouped_mode") or cfg.get(
        "dsv4_fp4_grouped_mode"
    )
    backend = cfg.get("moe_grouped_backend")
    RUNTIME.grouped_backend = str(backend).lower() if backend else None
    chunk = cfg.get("moe_dequant_chunk_size")
    RUNTIME.dequant_chunk_size = int(chunk) if chunk else None
    # backward base-dX precision: fp8-read (fast, sm120-only default) vs bf16-dequant. Config knob
    # so users can force the accurate path; env AXOLOTL_FP8_DX=0 as a no-config kill-switch.
    import os

    fp8_dx = cfg.get("moe_fp8_dx")
    if fp8_dx is None and os.environ.get("AXOLOTL_FP8_DX") is not None:
        fp8_dx = os.environ["AXOLOTL_FP8_DX"] != "0"
    if fp8_dx is not None:
        RUNTIME.fp4_dx_prefer_fp8 = bool(fp8_dx)
    marlin_dx = cfg.get("moe_marlin_dx")
    if marlin_dx is None and os.environ.get("AXOLOTL_MARLIN_DX") is not None:
        marlin_dx = os.environ["AXOLOTL_MARLIN_DX"] != "0"
    if marlin_dx is not None:
        RUNTIME.marlin_dx = bool(marlin_dx)
    if RUNTIME.marlin_dx:
        LOG.info(
            "moe_marlin_dx: backward dX runs as a W4A16 marlin GEMM on GPU-resident transposed "
            "expert packs. Per-step transients shrink but a second resident expert copy "
            "(~the forward pack size) is held — VRAM increases with model size. Set "
            "moe_marlin_dx: false to disable."
        )
    ka = cfg.get("moe_kernel_autotune")
    if ka is not None:
        RUNTIME.kernel_autotune = bool(ka)
    swiglu_ep = cfg.get("moe_swiglu_epilogue")
    if swiglu_ep is None and os.environ.get("AXOLOTL_SWIGLU_EPILOGUE") is not None:
        swiglu_ep = os.environ["AXOLOTL_SWIGLU_EPILOGUE"] != "0"
    if swiglu_ep is not None:
        RUNTIME.swiglu_epilogue = bool(swiglu_ep)
    RUNTIME.layer_gc_active = bool(cfg.get("gradient_checkpointing"))
    fast = cfg.get("moe_bnb_fast")
    RUNTIME.bnb_fast = True if fast is None else bool(fast)
    # Cache only in the non-FSDP case; under FSDP a persistent mxfp4 cache holds a full-model copy
    # per rank (OOM).
    RUNTIME.mxfp4_cache_persist = not bool(cfg.get("fsdp_config"))
    RUNTIME.experts_gathered = bool(cfg.get("fsdp_config") or cfg.get("deepspeed"))
