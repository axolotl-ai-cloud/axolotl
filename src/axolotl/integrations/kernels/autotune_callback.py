"""Trainer callback for reporting Triton autotune results from scattermoe-lora kernels."""

import logging

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

LOG = logging.getLogger(__name__)

# Give up looking for autotune data after this many training steps.
_MAX_POLL_STEP = 5


def _get_gpu_info() -> dict:
    """Return basic GPU identification for the current device."""
    if not torch.cuda.is_available():
        return {}
    try:
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return {
            "gpu_name": props.name,
            "gpu_compute_capability": f"{props.major}.{props.minor}",
            "gpu_memory_bytes": props.total_memory,
        }
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def _get_smem_capacity() -> dict:
    """Return shared memory capacity from the runtime lora_ops module."""
    try:
        from axolotl.integrations.kernels.autotune_collector import (
            _find_lora_ops_module,
        )

        lora_ops = _find_lora_ops_module()
        if lora_ops is None:
            return {}
        fn = getattr(lora_ops, "_get_smem_capacity", None)
        if fn is None:
            return {}
        return {"smem_capacity_bytes": fn()}
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


class AutotuneReportCallback(TrainerCallback):
    """Reports Triton kernel autotune selections via telemetry.

    Fires **once** after the first training step completes (step 1), at
    which point the forward and backward passes have both run and the
    autotuned kernels have populated their caches.  If for some reason
    the caches are still empty (e.g. the kernel was never invoked), the
    callback retries on subsequent steps up to ``_MAX_POLL_STEP`` and
    then stops polling.

    After reporting (or giving up) every subsequent ``on_step_end``
    call short-circuits on the ``_reported`` flag — zero hot-path cost.
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

        # Lazy import — Triton / scattermoe kernels may not be installed.
        from axolotl.integrations.kernels.autotune_collector import (
            collect_autotune_configs,
        )

        configs = collect_autotune_configs()

        if not configs:
            if state.global_step >= _MAX_POLL_STEP:
                LOG.debug(
                    "No autotune data found after %d steps; giving up.",
                    state.global_step,
                )
                self._reported = True
            return

        self._reported = True

        from axolotl.telemetry.manager import TelemetryManager

        telemetry_manager = TelemetryManager.get_instance()
        if not telemetry_manager.enabled:
            return

        properties = {
            "kernel_count": len(configs),
            "kernels": configs,
        }
        properties.update(_get_gpu_info())
        properties.update(_get_smem_capacity())

        telemetry_manager.send_event(
            event_type="scattermoe-autotune",
            properties=properties,
        )

        LOG.info(
            "Reported %d scattermoe kernel autotune config(s) to telemetry.",
            len(configs),
        )
