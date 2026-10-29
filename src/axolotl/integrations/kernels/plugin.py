# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

import functools
import importlib
import inspect
import os
import sys

import torch

from axolotl.integrations.base import BasePlugin
from axolotl.integrations.kernels.adapters import get_active_adapters
from axolotl.integrations.kernels.quant_training_guard import (
    relax_quantized_training_guard,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _preload_sonicmoe_kernel() -> bool:
    """Import the hub sonic-moe kernel now so its quack autotuners exist before any forward.

    Otherwise the first MoE forward imports it, and with ``eval_on_start`` that forward is in
    the step-0 eval, which runs after ``on_train_begin`` and before the first ``on_step_begin``:
    neither hook could attach autotune profiles or seed the quack cache for that eval.
    """
    try:
        from transformers.integrations.sonicmoe import load_sonicmoe_kernel

        load_sonicmoe_kernel()
    except Exception as exc:  # pylint: disable=broad-except
        LOG.warning(
            "Could not preload the sonic-moe kernel (%s); its autotune profiles attach "
            "at the first training step instead",
            exc,
        )
        return False
    return True


def _quack_policy(cfg) -> tuple[str, bool] | None:
    """(key buckets, sweep) for the quack autotuners, or None without SonicMoE."""
    if not cfg.use_sonicmoe:
        return None
    return cfg.quack_autotune_key_buckets or "pow2", cfg.quack_autotune is not False


def _seed_ep_padding(cfg, model) -> None:
    from axolotl.integrations.kernels.libs.sonicmoe.experts import seed_ep_padding_marks

    ep = cfg.expert_parallel_size or 1
    cp = getattr(cfg, "context_parallel_size", None) or 1
    tokens = (cfg.sequence_len or 0) * (cfg.micro_batch_size or 1) * ep // cp
    pairs = None
    if cfg.sonicmoe_ep_compact_routing and cfg.sonicmoe_ep_pair_seed_factor:
        config = getattr(model, "config", None)
        config = getattr(config, "text_config", None) or config
        top_k = getattr(config, "num_experts_per_tok", None)
        if top_k:
            pairs = int(tokens * top_k / ep * cfg.sonicmoe_ep_pair_seed_factor)
        else:
            LOG.warning(
                "sonicmoe_ep_pair_seed_factor: no num_experts_per_tok on the model config"
            )
    seed_ep_padding_marks(rows=tokens, pairs=pairs)
    LOG.info("Seeded SonicMoE EP padding marks: %d tokens, %s pairs", tokens, pairs)


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

    _check_sonicmoe_cutlass_dsl()
    _shim_nvvm_mbarrier_arrive()


def _shim_nvvm_mbarrier_arrive():
    """Accept the pre-4.8 ``nvvm.mbarrier_arrive(res, addr, ...)`` call form on cutlass-dsl 4.8+.

    cutlass-dsl 4.8 moved the leading result type to the keyword-only ``results=``; the quack
    0.6.1 vendored in kernels-community/sonic-moe still passes it positionally (``None`` = no
    result) and fails to compile its cluster-scope barrier arrive.
    """
    try:
        from cutlass._mlir.dialects import nvvm
    except ImportError:
        return
    fn = getattr(nvvm, "mbarrier_arrive", None)
    if fn is None or getattr(fn, "_axolotl_legacy_shim", False):
        return
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return
    positional = [
        p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) != 1:
        return

    @functools.wraps(fn)
    def mbarrier_arrive(*args, **kwargs):
        if len(args) == 2:
            result_type, args = args[0], args[1:]
            if result_type is not None:
                kwargs.setdefault("results", [result_type])
        return fn(*args, **kwargs)

    mbarrier_arrive._axolotl_legacy_shim = True  # type: ignore[attr-defined]
    nvvm.mbarrier_arrive = mbarrier_arrive
    LOG.info(
        "cutlass-dsl nvvm.mbarrier_arrive: legacy positional result-type form shimmed"
    )


def _loaded_vendored_quack() -> dict[str, str]:
    """``{module: __version__}`` for quack copies bundled in a loaded hub kernel."""
    found = {}
    for name, module in list(sys.modules.items()):
        if module is None or name == "quack" or not name.endswith(".quack"):
            continue
        if name.startswith("torch."):
            continue
        version = getattr(module, "__version__", None)
        if version is not None:
            found[name] = str(version)
    return found


def _check_sonicmoe_cutlass_dsl():
    """Fail fast when the installed cutlass DSL is outside quack's declared range, instead of a
    cryptic MLIR/import crash (quack 0.6.1 pinned ==4.6.0, 0.6.3+ need >=4.7)."""
    from importlib.metadata import (
        PackageNotFoundError,
        requires,
        version as _pkg_version,
    )

    from packaging.requirements import InvalidRequirement, Requirement

    try:
        cutlass_dsl_version = _pkg_version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        cutlass_dsl_version = None
    try:
        quack_version = _pkg_version("quack-kernels")
        quack_requires = requires("quack-kernels") or []
    except PackageNotFoundError:
        quack_version, quack_requires = None, []
    cutlass_spec = None
    for entry in quack_requires:
        try:
            requirement = Requirement(entry)
        except InvalidRequirement:
            continue
        if requirement.name == "nvidia-cutlass-dsl" and requirement.marker is None:
            cutlass_spec = requirement.specifier
            break
    vendored = _loaded_vendored_quack()
    if vendored:
        # the installed quack-kernels is not the quack that runs: its cutlass pin says nothing
        LOG.info(
            "SonicMoE runs a vendored quack (%s) on nvidia-cutlass-dsl %s; installed "
            "quack-kernels %s (requires nvidia-cutlass-dsl%s) is not the one in use",
            ", ".join(f"{name} {v}" for name, v in vendored.items()),
            cutlass_dsl_version,
            quack_version,
            cutlass_spec or "",
        )
    elif quack_version is not None:
        LOG.info(
            "SonicMoE: quack-kernels %s, nvidia-cutlass-dsl %s (a hub sonic-moe kernel "
            "may load its own vendored quack later)",
            quack_version,
            cutlass_dsl_version,
        )
    if (
        not vendored
        and cutlass_dsl_version is not None
        and cutlass_spec is not None
        and not cutlass_spec.contains(cutlass_dsl_version, prereleases=True)
    ):
        raise RuntimeError(
            f"SonicMoE (quack-kernels {quack_version}) requires "
            f"nvidia-cutlass-dsl{cutlass_spec}, but found {cutlass_dsl_version} "
            f"(uv pip install 'nvidia-cutlass-dsl{cutlass_spec}')."
        )

    # apache-tvm-ffi is an UNDECLARED dep of cutlass-dsl (absent from its Requires-Dist, so
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
                f"SonicMoE (quack on cutlass-dsl {cutlass_dsl_version}) requires "
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
    from axolotl.integrations.kernels.adapters.mistral_large4 import (
        is_mistral_large4_nvfp4_modelopt,
    )
    from axolotl.integrations.kernels.adapters.nemotron_h import (
        is_nemotron_h_nvfp4_modelopt,
    )
    from axolotl.integrations.kernels.adapters.nvfp4_moe import is_moe_nvfp4_modelopt
    from axolotl.integrations.kernels.adapters.qwen3_moe import (
        is_qwen3_moe_nvfp4_modelopt,
    )

    return (
        is_qwen3_moe_nvfp4_modelopt(cfg)
        or is_gemma4_nvfp4_modelopt(cfg)
        or is_glm_moe_dsa_nvfp4_modelopt(cfg)
        or is_nemotron_h_nvfp4_modelopt(cfg)
        or is_mistral_large4_nvfp4_modelopt(cfg)
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
        ``experts_implementation`` (``expert_parallel``, wrapping this kernel), so we don't
        overwrite it here.
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
                configure_ep_compact_routing,
                configure_ep_row_padding,
                register_sonicmoe_experts,
            )
            from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
                configure_lora_split,
            )

            register_sonicmoe_experts()
            configure_ep_row_padding(
                cfg.sonicmoe_ep_row_padding or "pow2",
                granule=cfg.sonicmoe_ep_row_padding_granule or 4096,
                headroom=cfg.sonicmoe_ep_row_padding_headroom or 1.03,
            )
            configure_ep_compact_routing(
                bool(cfg.sonicmoe_ep_compact_routing),
                granule=cfg.sonicmoe_ep_pair_padding_granule or 32768,
            )
            configure_lora_split(bool(cfg.sonicmoe_lora_split))
            if not ep_active:
                cfg.experts_implementation = "sonicmoe"
            LOG.info("Registered 'sonicmoe' in transformers ExpertsInterface")

            # Same frozen-quantized-base + LoRA pattern as the scattermoe path above.
            relax_quantized_training_guard()

        if cfg.adapter in ("lora", "multilora") and _base_is_nvfp4_modelopt(cfg):
            from axolotl.integrations.kernels.merge_aware_setup import (
                configure_modelopt_merge_aware,
            )

            configure_modelopt_merge_aware(cfg)

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
        if cfg.use_sonicmoe and cfg.sonicmoe_ep_padding_seed:
            _seed_ep_padding(cfg, model)
        sonic_loaded = bool(cfg.use_sonicmoe) and _preload_sonicmoe_kernel()
        if cfg.autotune_cache_path:
            from axolotl.integrations.kernels.autotune_cache import load_autotune_cache

            load_autotune_cache(cfg.autotune_cache_path)
            if sonic_loaded:
                from axolotl.integrations.kernels.autotune_cache import (
                    load_quack_cache,
                    quack_cache_path,
                )

                load_quack_cache(quack_cache_path(cfg.autotune_cache_path))
        if cfg.autotune_profiles:
            from axolotl.integrations.kernels.autotune_profiles import (
                apply_autotune_profiles,
            )

            apply_autotune_profiles(cfg.autotune_profiles)
        if cfg.autotune_neighborhood:
            from axolotl.integrations.kernels.autotune_neighborhood import (
                install_neighborhood_search,
            )

            install_neighborhood_search(cfg.autotune_neighborhood)
        policy = _quack_policy(cfg)
        if policy is not None:
            from axolotl.integrations.kernels.quack_autotune_policy import (
                install_quack_key_policy,
            )

            install_quack_key_policy(*policy)
        if cfg.use_sonicmoe:
            from axolotl.integrations.kernels.libs.sonicmoe.epilogue import (
                check_model_epilogues,
            )

            check_model_epilogues(model)
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
        if (
            cfg.autotune_cache_path
            or cfg.autotune_profiles
            or cfg.autotune_neighborhood
            or cfg.use_sonicmoe
        ):
            from axolotl.integrations.kernels.autotune_cache import (
                AutotuneCacheCallback,
            )

            callbacks.append(
                AutotuneCacheCallback(
                    cfg.autotune_cache_path,
                    cfg.autotune_profiles,
                    neighborhood=cfg.autotune_neighborhood,
                    sonicmoe=bool(cfg.use_sonicmoe),
                    quack_policy=_quack_policy(cfg),
                )
            )
        if cfg.use_sonicmoe and (
            cfg.sonicmoe_ep_row_padding == "highwater"
            or cfg.sonicmoe_ep_compact_routing
        ):
            from axolotl.integrations.kernels.libs.sonicmoe.experts import (
                EPPaddingMarksCallback,
            )

            callbacks.append(EPPaddingMarksCallback())
        if cfg.use_sonicmoe and cfg.nvfp4_merge_aware:
            from axolotl.integrations.kernels.merge_aware_callback import (
                MergeAwareScheduleCallback,
            )

            self._merge_aware_schedule_callback = MergeAwareScheduleCallback(
                cfg.nvfp4_merge_aware_start_step
            )
            callbacks.append(self._merge_aware_schedule_callback)
        return callbacks

    def post_train_unload(self, cfg):
        # runs after the FINAL adapter save (on_save only covers checkpoint-*/)
        if not (cfg.use_sonicmoe and cfg.nvfp4_merge_aware):
            return
        from axolotl.integrations.kernels.libs.sonicmoe import merge_aware_enabled
        from axolotl.integrations.kernels.merge_aware_callback import (
            write_merge_aware_metadata,
        )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            is_main = torch.distributed.get_rank() == 0
        else:
            is_main = os.environ.get("RANK", "0") == "0"

        # merge_aware_enabled() still False => start_step was never reached, the
        # adapter never trained through the fake-quant: leave it unstamped
        callback = getattr(self, "_merge_aware_schedule_callback", None)
        if is_main and callback is not None and not callback.merge_aware_valid:
            from axolotl.integrations.kernels.merge_aware_callback import (
                clear_merge_aware_metadata,
            )

            clear_merge_aware_metadata(cfg.output_dir)
            return
        if is_main and merge_aware_enabled():
            if write_merge_aware_metadata(
                cfg.output_dir, start_step=cfg.nvfp4_merge_aware_start_step
            ):
                LOG.info(
                    "stamped nvfp4_merge_aware quantizer identity into %s/adapter_config.json",
                    cfg.output_dir,
                )
