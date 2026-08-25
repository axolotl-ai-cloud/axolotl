"""NVFP4 pre-load patches — torch._dynamo config tuning.

Relocated from core/builders/base.py::_configure_torch_compile. Applied from the
plugin's pre_model_load so the core builder stays unmodified.
"""

import os

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def warn_unfilled_gemms(cfg) -> None:
    """Warn when FP4 will likely regress: no sample_packing/pad_to_sequence_len (under-filled GEMMs) or torch_compile off."""
    if not cfg.sample_packing and not cfg.pad_to_sequence_len:
        LOG.warning(
            "nvfp4_training: neither `sample_packing` nor `pad_to_sequence_len` is "
            "set. FP4 needs filled GEMMs — with ragged/unpadded batches the per-step "
            "quant overhead dominates (often SLOWER than bf16) and torch.compile "
            "recompiles per shape. Enable `sample_packing: true` (best) or "
            "`pad_to_sequence_len: true`."
        )
    if not cfg.torch_compile:
        LOG.warning(
            "nvfp4_training: `torch_compile` is off. The FP4 speedup only "
            "materializes under torch.compile; in eager mode NVFP4 is slower than "
            "bf16. Set `torch_compile: true`."
        )


def configure_dynamo_for_nvfp4(cfg) -> None:
    """Tune torch._dynamo for NVFP4 + multi-GPU DDP under dynamic shapes."""
    if not (cfg.torch_compile and getattr(torch, "_dynamo", None)):
        return
    nvfp4 = getattr(cfg, "nvfp4_training", None)
    if not (nvfp4 and nvfp4.enabled):
        return

    # Dynamo specializes on module.layer_idx and recompiles attention per layer;
    # marking nn.Module ints unspecialized collapses it to one shared graph.
    if str(cfg.attn_implementation).startswith("flash_attention"):
        if hasattr(torch._dynamo.config, "allow_unspec_int_on_nn_module"):
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
        if hasattr(torch._dynamo.config, "capture_scalar_outputs"):
            torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 256
        )

    if (
        not cfg.fsdp_config
        and int(os.environ.get("WORLD_SIZE", "1") or "1") > 1
        and hasattr(torch._dynamo.config, "optimize_ddp")
    ):
        torch._dynamo.config.optimize_ddp = False
        LOG.info(
            "NVFP4 DDP: disabled torch._dynamo DDPOptimizer (optimize_ddp) to avoid "
            "the dynamic-shape graph-split symint codegen error; the model compiles "
            "as a single graph."
        )
