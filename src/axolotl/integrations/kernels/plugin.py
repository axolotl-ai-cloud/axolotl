import importlib
import os

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.kernels.adapters import get_active_adapters
from axolotl.integrations.kernels.quant_training_guard import (
    relax_quantized_training_guard,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _check_sonicmoe_gpu_compat():
    """Validate GPU compute capability for SonicMoE and configure env.

    Supported: Hopper (sm_90), Blackwell (sm_100 - sm_103).
    B300 (sm_103) additionally requires Triton 3.6.0.
    """
    if not torch.cuda.is_available():
        return

    cc = torch.cuda.get_device_capability()

    if cc < (9, 0):
        raise RuntimeError(
            f"SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+) GPU, "
            f"but detected sm_{cc[0]}{cc[1]}."
        )

    if cc > (10, 3):
        raise RuntimeError(
            f"SonicMoE does not yet support sm_{cc[0]}{cc[1]}. "
            f"Supported: Hopper (sm_90) and Blackwell (sm_100 - sm_103)."
        )

    if cc >= (10, 0):
        os.environ.setdefault("USE_QUACK_GEMM", "1")
        LOG.info(
            f"Blackwell GPU (sm_{cc[0]}{cc[1]}) detected, enabling USE_QUACK_GEMM=1"
        )

    if cc == (10, 3):
        triton_spec = importlib.util.find_spec("triton")
        if triton_spec is None:
            raise RuntimeError(
                "B300 (sm_103) requires Triton 3.6.0, but Triton is not installed."
            )
        import triton

        triton_version = tuple(int(x) for x in triton.__version__.split(".")[:2])
        if triton_version != (3, 6):
            raise RuntimeError(
                f"B300 (sm_103) requires Triton 3.6.x, but found {triton.__version__}."
            )


class KernelsPlugin(BasePlugin):
    """Thin orchestrator: registers the expert-kernel backend and dispatches model-family
    specifics to ``ModelAdapter`` subclasses (see ``adapters/``)."""

    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def _adapters(self, cfg):
        # Cache: matching can be expensive (e.g. AutoConfig.from_pretrained for Gemma-4).
        cached = getattr(self, "_cached_adapters", None)
        if cached is None:
            cached = get_active_adapters(cfg)
            self._cached_adapters = cached
        return cached

    def pre_model_load(self, cfg):
        """Register the expert-kernel backend + generic capabilities, then run adapter hooks.

        Architecture-agnostic: routing stays in each model's SparseMoEBlock; only the experts
        call is dispatched through the registry. When EP is active the ExpertParallelPlugin owns
        ``experts_implementation`` (a ``deep_ep_*`` composite), so we don't overwrite it here.
        """
        ep_active = (getattr(cfg, "expert_parallel_size", 1) or 1) > 1

        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
                register_scattermoe_experts,
            )
            from axolotl.integrations.kernels.libs.scattermoe_lora.runtime import (
                configure_scattermoe_runtime,
            )

            register_scattermoe_experts()
            if not ep_active:
                cfg.experts_implementation = "scattermoe"
            LOG.info("Registered 'scattermoe' in transformers ExpertsInterface")

            # LoRA on a frozen pre-quantized (FP8/NVFP4) base is supported but rejected by the
            # upstream guard; scope-relax it (skips only PEFT/quantized, delegates the rest).
            relax_quantized_training_guard()

            # Apply ALL ScatterMoE runtime settings from this run's config through one entry point
            # (it resets first, so a long-lived multi-run process can't inherit stale state).
            configure_scattermoe_runtime(cfg)
            if cfg.get("dsv4_fp4_grouped_mode"):
                LOG.info(
                    "Enabled grouped fp4 MoE path: dsv4_fp4_grouped_mode=%s",
                    cfg.get("dsv4_fp4_grouped_mode"),
                )
                if cfg.get("moe_grouped_backend"):
                    LOG.info(
                        "Grouped MoE base-GEMM backend override: %s",
                        cfg.get("moe_grouped_backend"),
                    )
        elif cfg.use_sonicmoe:
            _check_sonicmoe_gpu_compat()

            from axolotl.integrations.kernels.libs.sonicmoe.experts import (
                register_sonicmoe_experts,
            )

            register_sonicmoe_experts()
            if not ep_active:
                cfg.experts_implementation = "sonicmoe"
            LOG.info("Registered 'sonicmoe' in transformers ExpertsInterface")

        adapters = self._adapters(cfg)
        self._warn_unclaimed_nonexpert_quantization(cfg, adapters)
        for adapter in adapters:
            adapter.pre_model_load(cfg)

    @staticmethod
    def _warn_unclaimed_nonexpert_quantization(cfg, adapters):
        """Warn if a non-expert quantization policy is set but no active adapter consumes it.

        ``nonexpert_quantization`` is a global intent, but only some model adapters act on it
        (e.g. Gemma-4). Without this, configuring it on an unsupported model silently no-ops.
        ``none``/``bf16`` mean "no quantization", so they're never considered unclaimed.
        """
        policy = cfg.get("nonexpert_quantization")
        if not policy or str(policy).lower() in ("none", "bf16"):
            return
        if any(a.consumes_nonexpert_quantization(cfg) for a in adapters):
            return
        LOG.warning(
            "nonexpert_quantization=%r is set but no active model adapter consumes it "
            "(active adapters: %s); it will have no effect. It is currently implemented only "
            "for Gemma-4 NVFP4 checkpoints.",
            policy,
            [a.name for a in adapters] or "none",
        )

    def pre_lora_load(self, cfg, model):
        for adapter in self._adapters(cfg):
            adapter.pre_lora_load(cfg, model)

    def post_model_load(self, cfg, model):
        for adapter in self._adapters(cfg):
            adapter.post_model_load(cfg, model)

    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.autotune_callback import (
                AutotuneReportCallback,
            )

            callbacks.append(AutotuneReportCallback())
        return callbacks
