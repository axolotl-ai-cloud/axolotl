"""Telemetry for the fused RMSNorm+RoPE Triton autotune selections.

Mirrors the scattermoe-lora autotune telemetry
(:mod:`axolotl.integrations.kernels.autotune_callback`): after the kernel's
``@triton.autotune`` cache is populated by the first backward pass, report the
selected configs alongside GPU identity so the per-hardware tuning that varies
across architectures can be aggregated.
"""

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Give up looking for autotune data after this many training steps.
_MAX_POLL_STEP = 5

# (human-readable name, attribute on gemma4_fused_rope, autotune key arg names)
_KERNEL_REGISTRY: list[tuple[str, str, list[str]]] = [
    ("fused_rms_norm_rope_bwd", "_rms_norm_rope_backward_kernel", ["n_cols"]),
]


def _get_gpu_info() -> dict:
    """Return basic GPU identification for the current device."""
    if not torch.cuda.is_available():
        return {}
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_memory_bytes": props.total_memory,
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def collect_fused_rope_autotune_configs() -> list[dict]:
    """Read the autotune ``.cache`` from the fused RMSNorm+RoPE backward kernel.

    Each entry is ``{"kernel", "key", "config"}`` — the same shape the
    scattermoe collector emits, so both event types aggregate uniformly.
    Returns ``[]`` if Triton/the kernel isn't loaded or nothing autotuned yet.
    """
    import sys

    # The kernel module is only in sys.modules once the fused path has run —
    # which is exactly when its autotune cache is populated. Read it from there
    # instead of importing (avoids pulling in Triton when the path is unused).
    mod = sys.modules.get("axolotl.kernels.gemma4_fused_rope")
    if mod is None:
        return []

    results: list[dict] = []
    for friendly_name, attr_name, key_names in _KERNEL_REGISTRY:
        kernel_fn = getattr(mod, attr_name, None)
        cache = getattr(kernel_fn, "cache", None)
        if not cache:
            continue
        for key_tuple, config in cache.items():
            config_dict = dict(config.kwargs)
            config_dict["num_warps"] = config.num_warps
            config_dict["num_stages"] = config.num_stages
            if getattr(config, "num_ctas", None) is not None:
                config_dict["num_ctas"] = config.num_ctas

            key: dict = {}
            for i, name in enumerate(key_names):
                if i < len(key_tuple):
                    key[name] = key_tuple[i]
            if len(key_tuple) > len(key_names):
                key["_extra"] = [str(v) for v in key_tuple[len(key_names) :]]

            results.append({"kernel": friendly_name, "key": key, "config": config_dict})
    return results


class FusedRopeAutotuneReportCallback(TrainerCallback):
    """Reports fused RMSNorm+RoPE autotune selections via telemetry.

    Fires once after the autotune cache is populated (the first step whose
    backward has run), retrying up to ``_MAX_POLL_STEP`` then giving up. Every
    later ``on_step_end`` short-circuits on ``_reported`` — zero hot-path cost.
    """

    def __init__(self):
        self._reported = False

    # pylint: disable=unused-argument
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self._reported:
            return

        configs = collect_fused_rope_autotune_configs()
        if not configs:
            if state.global_step >= _MAX_POLL_STEP:
                LOG.debug(
                    "No fused-rope autotune data after %d steps; giving up.",
                    state.global_step,
                )
                self._reported = True
            return

        self._reported = True

        from axolotl.telemetry.manager import TelemetryManager

        telemetry_manager = TelemetryManager.get_instance()
        if not telemetry_manager.enabled:
            return

        properties = {"kernel_count": len(configs), "kernels": configs}
        properties.update(_get_gpu_info())

        telemetry_manager.send_event(
            event_type="fused-rope-autotune",
            properties=properties,
        )
        LOG.info(
            "Reported %d fused-rope kernel autotune config(s) to telemetry.",
            len(configs),
        )
