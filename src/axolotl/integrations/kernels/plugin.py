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

    Supported: Hopper (sm_90) and Blackwell (sm_100-sm_103 datacenter, sm_120+
    consumer). Consumer Blackwell (sm_120) needs a sonic-moe build bundling
    quack 0.6.1 with nvidia-cutlass-dsl 4.6.0 (earlier prebuilts lack the sm_120
    GEMM). B300 (sm_103) additionally requires Triton 3.6.0.
    """
    if not torch.cuda.is_available():
        return

    cc = torch.cuda.get_device_capability()

    if cc < (9, 0):
        raise RuntimeError(
            f"SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+) GPU, "
            f"but detected sm_{cc[0]}{cc[1]}."
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

    # quack 0.6.1 pins nvidia-cutlass-dsl==4.6.0: 4.5.x lacks the CuTe API it needs
    # (ThrMma dropped) and other majors are unvalidated, so fail fast on a mismatch
    # instead of a cryptic MLIR/import crash.
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        cutlass_dsl_version = _pkg_version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        cutlass_dsl_version = None
    if cutlass_dsl_version is not None:
        try:
            cutlass_mm = tuple(int(x) for x in cutlass_dsl_version.split(".")[:2])
        except ValueError:
            cutlass_mm = None
        if cutlass_mm is not None and cutlass_mm != (4, 6):
            raise RuntimeError(
                f"SonicMoE (quack 0.6.1) requires nvidia-cutlass-dsl==4.6.0, but found "
                f"{cutlass_dsl_version} (uv pip install 'nvidia-cutlass-dsl==4.6.0')."
            )

    # apache-tvm-ffi is an UNDECLARED dep of cutlass-dsl 4.6.0 (absent from its Requires-Dist, so
    # pip won't auto-pull it): <0.1.10 breaks cute.compile, >=0.2 is unvalidated.
    try:
        tvm_ffi_version = _pkg_version("apache-tvm-ffi")
    except PackageNotFoundError:
        tvm_ffi_version = None
    if tvm_ffi_version is not None:
        try:
            tvm_ffi_v = tuple(int(x) for x in tvm_ffi_version.split(".")[:3])
        except ValueError:
            tvm_ffi_v = None
        if tvm_ffi_v is not None and not (0, 1, 10) <= tvm_ffi_v < (0, 2, 0):
            raise RuntimeError(
                f"SonicMoE (quack 0.6.1 on cutlass-dsl 4.6.0) requires "
                f"apache-tvm-ffi>=0.1.10,<0.2, but found {tvm_ffi_version}. cutlass-dsl "
                f"does not pull it transitively; install it explicitly "
                f"(uv pip install 'apache-tvm-ffi>=0.1.10,<0.2')."
            )


def _base_is_nvfp4_modelopt(cfg) -> bool:
    """True iff the base is an NVFP4-modelopt checkpoint the sonicmoe path trains (a specialized
    arch or the generic MoE gate). Reads the base config; any failure = False."""
    from axolotl.integrations.kernels.adapters.gemma4 import is_gemma4_nvfp4_modelopt
    from axolotl.integrations.kernels.adapters.glm_moe_dsa import (
        is_glm_moe_dsa_nvfp4_modelopt,
    )
    from axolotl.integrations.kernels.adapters.nvfp4_moe import is_moe_nvfp4_modelopt
    from axolotl.integrations.kernels.adapters.qwen3_moe import (
        is_qwen3_moe_nvfp4_modelopt,
    )

    return (
        is_qwen3_moe_nvfp4_modelopt(cfg)
        or is_gemma4_nvfp4_modelopt(cfg)
        or is_glm_moe_dsa_nvfp4_modelopt(cfg)
        or is_moe_nvfp4_modelopt(cfg)
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

            # register_sonicmoe_experts() redirects the sonic-moe hub kernel to our build.
            register_sonicmoe_experts()
            if not ep_active:
                cfg.experts_implementation = "sonicmoe"
            LOG.info("Registered 'sonicmoe' in transformers ExpertsInterface")

            # Same frozen-quantized-base + LoRA pattern as the scattermoe path above.
            relax_quantized_training_guard()

            # NVFP4-ness is only in the downloaded base config, not the YAML, so not a validator.
            if (
                cfg.adapter == "lora"
                and not cfg.nvfp4_merge_aware
                and _base_is_nvfp4_modelopt(cfg)
            ):
                LOG.warning(
                    "NVFP4 base + sonicmoe LoRA WITHOUT nvfp4_merge_aware: the "
                    "format-preserving `axolotl merge-lora` snaps dequant(base) + "
                    "scaling*(B@A) back onto the base NVFP4 grid and ERASES the "
                    "sub-grid-step LoRA delta, so the merged checkpoint reverts to the "
                    "base model and this training run is wasted.\n"
                    "Set `nvfp4_merge_aware: true` to fake-quant the effective weight "
                    "during training so the format-preserving merge reproduces the "
                    "trained model, or plan to merge with `--dequant` (bf16 output, "
                    "loses the NVFP4 format)."
                )

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

    def post_lora_load(self, cfg, model):
        if not (cfg.use_sonicmoe and cfg.nvfp4_merge_aware):
            return
        from axolotl.integrations.kernels.merge_aware_linear import (
            install_merge_aware_lora_linears,
        )

        wrapped = install_merge_aware_lora_linears(model)
        if wrapped:
            LOG.info(
                "merge-aware LoRA forward installed on %d non-expert NVFP4 linears",
                wrapped,
            )

    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.autotune_callback import (
                AutotuneReportCallback,
            )

            callbacks.append(AutotuneReportCallback())
        if cfg.use_sonicmoe and cfg.nvfp4_merge_aware:
            from axolotl.integrations.kernels.merge_aware_callback import (
                MergeAwareScheduleCallback,
            )

            callbacks.append(
                MergeAwareScheduleCallback(cfg.nvfp4_merge_aware_start_step)
            )
        return callbacks

    def post_train_unload(self, cfg):
        # runs after the FINAL adapter save (on_save only covers checkpoint-*/)
        if not (cfg.use_sonicmoe and cfg.nvfp4_merge_aware):
            return
        from axolotl.integrations.kernels.libs.sonicmoe import merge_aware_enabled
        from axolotl.integrations.kernels.merge_aware_callback import (
            write_merge_aware_metadata,
        )
        from axolotl.utils.distributed import is_main_process

        # merge_aware_enabled() still False => start_step was never reached, the
        # adapter never trained through the fake-quant: leave it unstamped
        if is_main_process() and merge_aware_enabled():
            if write_merge_aware_metadata(
                cfg.output_dir, start_step=cfg.nvfp4_merge_aware_start_step
            ):
                LOG.info(
                    "stamped nvfp4_merge_aware quantizer identity into %s/adapter_config.json",
                    cfg.output_dir,
                )
