import importlib
import os

import torch

from axolotl.integrations.base import BasePlugin
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

    # Blackwell (sm_100+): enable QuACK GEMM kernels
    if cc >= (10, 0):
        os.environ.setdefault("USE_QUACK_GEMM", "1")
        LOG.info(
            f"Blackwell GPU (sm_{cc[0]}{cc[1]}) detected, enabling USE_QUACK_GEMM=1"
        )

    # B300 (sm_103): requires Triton 3.6.0
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
    def get_input_args(self):
        return "axolotl.integrations.kernels.KernelsArgs"

    def pre_model_load(self, cfg):
        """Register the requested kernel into ``ALL_EXPERTS_FUNCTIONS`` and pin cfg.

        Architecture-agnostic: routing stays in each model's SparseMoEBlock; only
        the experts call is dispatched through the registry.
        """
        # When EP is active, the ExpertParallelPlugin selects a `deep_ep_*`
        # composite for `experts_implementation`. Don't overwrite that here —
        # plugin order is YAML-defined, so we can't rely on EP running last.
        ep_active = (getattr(cfg, "expert_parallel_size", 1) or 1) > 1

        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
                register_scattermoe_experts,
            )

            register_scattermoe_experts()
            if not ep_active:
                cfg.experts_implementation = "scattermoe"
            LOG.info("Registered 'scattermoe' in transformers ExpertsInterface")

            # transformers' `validate_quantization_for_training` rejects pre-quantized
            # FP8/NVFP4 checkpoints even when LoRA adapters are attached (the FP8 `elif`
            # fires regardless of `_is_peft_model`). scattermoe-lora trains a frozen
            # NVFP4/MXFP4 expert base via fused kernels with LoRA adapters — the
            # supported pattern — so neutralize that over-strict guard.
            import transformers.trainer as _hf_trainer

            _hf_trainer.validate_quantization_for_training = lambda *a, **k: None
            LOG.info("Relaxed transformers quantized-training guard for scattermoe-lora")

            # NVFP4 MoE checkpoints (e.g. DeepSeek-V4-Flash-NVFP4) declare `quant_method: fp8`
            # but transformers' finegrained-FP8 quantizer has no NVFP4 expert path. Swap in an
            # NVFP4-aware subclass so the experts load with correct scales (no UNEXPECTED/
            # MISSING warning) and become NVFP4Tensor for the scattermoe fused path.
            if cfg.use_dsv4_kernels:
                from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fp8_quantizer import (
                    install_nvfp4_fp8_quantizer,
                )

                install_nvfp4_fp8_quantizer()
        elif cfg.use_sonicmoe:
            _check_sonicmoe_gpu_compat()

            from axolotl.integrations.kernels.libs.sonicmoe.experts import (
                register_sonicmoe_experts,
            )

            register_sonicmoe_experts()
            if not ep_active:
                cfg.experts_implementation = "sonicmoe"
            LOG.info("Registered 'sonicmoe' in transformers ExpertsInterface")

        if cfg.use_dsv4_kernels:
            # Debug: skip the fused attention/rope/mHC/indexer forward patches (keeps the
            # NVFP4 loading + scattermoe experts) to isolate fused-kernel correctness from
            # loading/FSDP. Eager forward becomes the reference.
            if os.environ.get("DSV4_DISABLE_FUSED_FORWARD") == "1":
                LOG.warning("DSV4_DISABLE_FUSED_FORWARD=1: using EAGER attention/rope/mHC (no fused kernels)")
            else:
                from axolotl.integrations.kernels.libs.dsv4 import (
                    patch_deepseek_v4_kernels,
                )

                patch_deepseek_v4_kernels()

    def pre_lora_load(self, cfg, model):
        """Before PEFT wraps the experts: fix NVFP4 MoE-expert loading. transformers'
        finegrained_fp8 quantizer has no NVFP4 path, so a MIXED_PRECISION checkpoint's
        NVFP4 experts load with dropped/random scales — rebuild them as NVFP4Tensor from
        the checkpoint so scattermoe dequantizes correctly."""
        if not (cfg.use_scattermoe and cfg.use_dsv4_kernels):
            return

        # Debug-only: truncate the decoder stack to the first N layers (before PEFT) so the
        # full model fits a single GPU without FSDP — isolates kernel/scattermoe-lora
        # correctness. The first layers span all 3 types (sliding/CSA/HCA). Done here (not
        # post_model_load) so it runs pre-PEFT (correct trainable count) and on the unwrapped
        # model. Find the decoder ModuleList by content (robust to model nesting).
        trunc = os.environ.get("DSV4_TRUNCATE_LAYERS")
        if trunc:
            n = int(trunc)
            import torch.nn as nn

            for mod in model.modules():
                layers = getattr(mod, "layers", None)
                if isinstance(layers, nn.ModuleList) and len(layers) > n and "DecoderLayer" in type(layers[0]).__name__:
                    orig = len(layers)
                    mod.layers = layers[:n]
                    for cfg_obj in (model.config, getattr(mod, "config", None)):
                        if cfg_obj is None:
                            continue
                        if hasattr(cfg_obj, "num_hidden_layers"):
                            cfg_obj.num_hidden_layers = n
                        # Truncate per-layer config lists (layer_types, compress_ratios, ...)
                        # to keep HF's `len(layer_types) == num_hidden_layers` validators happy.
                        # Lists of length `orig` -> n; length `orig+1` (e.g. compress_ratios) -> n+1.
                        for k, v in list(vars(cfg_obj).items()):
                            if isinstance(v, list) and len(v) in (orig, orig + 1):
                                setattr(cfg_obj, k, v[: n + (len(v) - orig)])
                    import gc

                    gc.collect()
                    torch.cuda.empty_cache()
                    LOG.info("Truncated decoder stack to first %d layers (single-GPU debug)", n)
                    break

        has_packed_experts = any(
            isinstance(getattr(m, "gate_up_proj", None), torch.Tensor)
            and m.gate_up_proj.dtype == torch.uint8
            and m.gate_up_proj.ndim == 3
            for m in model.modules()
        )
        if not has_packed_experts:
            return
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            attach_nvfp4_expert_scales,
        )

        attach_nvfp4_expert_scales(model, cfg.base_model)

    def post_model_load(self, cfg, model):
        """After PEFT wraps the projections, swap V4 shared-expert MLPs for the fused
        clamped-SwiGLU LoRA kernel (the routed experts already go through scattermoe)."""
        # Two sources leave fp32 params in the model: (1) the model's
        # `_keep_in_fp32_modules_strict` (mHC attn_hc/ffn_hc/hc_head, sinks, position_bias,
        # e_score_correction_bias, norms — genuinely kept fp32 for value-storage precision);
        # (2) PEFT's `prepare_model_for_kbit_training` (run just before this hook) upcasting
        # RMSNorm + LoRA to fp32. (The separate non-strict `_keep_in_fp32_modules` list is the
        # compressor/indexer projections, which actually ship BF16 and are only listed so the
        # FP8 quantizer skips them — no fp32 effect.) The fused dsv4 kernels are dtype-robust
        # at their input boundary (each wrapper demotes fp32 activations to the compute dtype),
        # so we keep the strict-fp32 modules in fp32 (precise storage) and only cast the
        # *other* fp32 params (PEFT-upcast LoRA) to the compute dtype for one consistent path.
        # DSV4_BF16_ALL=1 reverts to the old blanket cast (incl. keep_in_fp32) as a fallback.
        if cfg.use_dsv4_kernels:
            dt = cfg.torch_dtype or torch.bfloat16
            keep_all = os.environ.get("DSV4_BF16_ALL") == "1"
            keep_patterns = []
            if not keep_all:
                seen = set()
                for m in model.modules():
                    for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
                        for pat in getattr(m, attr, None) or ():
                            if pat not in seen:
                                seen.add(pat)
                                keep_patterns.append(pat)
            n = kept = 0
            for name, p in model.named_parameters():
                if p.dtype != torch.float32:
                    continue
                if keep_patterns and any(pat in name for pat in keep_patterns):
                    kept += 1
                    continue
                p.data = p.data.to(dt)
                n += 1
            if n:
                LOG.info(
                    "Cast %d residual fp32 params to %s for dsv4 fused kernels (kept %d keep_in_fp32 fp32)",
                    n, dt, kept,
                )

        if cfg.use_dsv4_kernels and cfg.get("lora_mlp_kernel") and cfg.get("adapter") == "lora":
            from axolotl.integrations.kernels.libs.dsv4.lora_mlp import (
                patch_dsv4_shared_mlp_lora,
            )

            n = patch_dsv4_shared_mlp_lora(model)
            LOG.info(f"Patched {n} DeepSeek-V4 shared-expert MLPs with fused clamped-SwiGLU LoRA")

    def add_callbacks_pre_trainer(self, cfg, model):
        callbacks = []
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.autotune_callback import (
                AutotuneReportCallback,
            )

            callbacks.append(AutotuneReportCallback())
        return callbacks
