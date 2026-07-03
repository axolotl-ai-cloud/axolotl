"""Qwen3-MoE / Qwen3-Next NVFP4 kernel adapter.

A ``quant_method: modelopt`` / ``quant_algo: NVFP4`` Qwen3-MoE-family checkpoint (e.g.
nvidia/Qwen3-30B-A3B-NVFP4, nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4) is not a recognized
transformers quantizer, so the model loads as a bf16 skeleton; we then register
``WeightConverter``s so the NVFP4 tensors load correctly. Like the glm_moe_dsa adapter (this is
that adapter minus the DSA attention patching), the checkpoint's safetensors index is inspected
(:func:`inspect_nvfp4_layout`) to discover what is actually quantized: the per-expert
projections (fused into 3D ``NVFP4Tensor`` for the grouped MoE paths) and any non-routed NVFP4
linears (dequantized to bf16), and exactly those converters are registered for THIS checkpoint.
qwen3_next differs only outside the MoE block (Gated DeltaNet linear attention on 3/4 layers,
a shared expert, per-projection ``input_scale`` tensors, an ``mtp.*`` head the causal-LM class
never loads); the layout inspection handles all of that without model-specific code.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import ModelAdapter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def is_qwen3_moe_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``qwen3_moe``/``qwen3_next`` NVFP4-modelopt checkpoint
    (``quant_method=modelopt``, ``quant_algo=NVFP4``). Any failure returns False."""
    try:
        from transformers import AutoConfig

        # Honor the user's opt-in rather than forcing remote code: qwen3_moe is a native
        # transformers model, so config loading needs no remote code by default.
        hf_cfg = AutoConfig.from_pretrained(
            cfg.base_model,
            trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
        )
    except Exception:
        return False
    if str(getattr(hf_cfg, "model_type", "")) not in ("qwen3_moe", "qwen3_next"):
        return False
    qcfg = getattr(hf_cfg, "quantization_config", None)
    if qcfg is None:
        return False
    if isinstance(qcfg, dict):
        return (
            qcfg.get("quant_method") == "modelopt" and qcfg.get("quant_algo") == "NVFP4"
        )
    return (
        getattr(qcfg, "quant_method", None) == "modelopt"
        and getattr(qcfg, "quant_algo", None) == "NVFP4"
    )


class Qwen3MoeAdapter(ModelAdapter):
    name = "qwen3_moe"

    def matches(self, cfg) -> bool:
        return bool(
            cfg.use_scattermoe or cfg.use_sonicmoe
        ) and is_qwen3_moe_nvfp4_modelopt(cfg)

    def pre_model_load(self, cfg) -> None:
        from transformers import AutoConfig

        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            inspect_nvfp4_layout,
            patch_nvfp4_tensor_meta_ops,
            patch_skip_missing_expert_init,
        )

        # the conversion mapping is looked up under the model's actual type
        model_type = str(
            AutoConfig.from_pretrained(
                cfg.base_model,
                trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
            ).model_type
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (
            patch_conversion_loader_rank0_only,
            register_nvfp4_converters_for_layout,
        )

        # FSDP2 cpu_ram_efficient_loading materializes meta receive-buffers via zeros_like/empty_like.
        patch_nvfp4_tensor_meta_ops()
        # transformers' conversion loader ignores cpu_ram_efficient_loading (loads on every rank);
        # gate it to rank0-only so the full model doesn't blow up CPU RAM by the world size. Only
        # safe when the FSDP broadcast will later fill the non-rank-0 meta params — i.e. when
        # cpu_ram_efficient_loading is set — so DDP ranks (which each need real weights) are spared.
        if (cfg.get("fsdp_config") or {}).get("cpu_ram_efficient_loading"):
            patch_conversion_loader_rank0_only()
            # transformers gates the META SKELETON (and its own load path) on is_fsdp_enabled(),
            # which requires the process group to be initialized. axolotl doesn't init it until
            # AFTER model load, so without this non-rank-0 builds a full real-storage skeleton
            # (world-size× CPU blowup) before any weights even load. Init it now.
            from transformers.integrations.fsdp import is_fsdp_enabled

            from axolotl.utils.distributed import init_distributed_state

            init_distributed_state()
            LOG.info(
                "qwen3_moe: initialized distributed state for rank0-only loading "
                "(is_fsdp_enabled=%s)",
                is_fsdp_enabled(),
            )

        import os

        layout = inspect_nvfp4_layout(cfg.base_model)
        LOG.info(
            "qwen3_moe: detected NVFP4 layout — routed experts: %s (projs=%s); "
            "non-routed NVFP4 linears: %s",
            layout["routed_present"],
            layout["routed_projs"],
            layout["nonrouted_suffixes"],
        )
        # FAST routed-expert load (opt-in): skip the routed converters here and read+fuse the experts
        # DIRECTLY in post_model_load — bypasses transformers' per-tensor conversion loop over the
        # per-expert source tensors. Non-routed converters still register.
        self._direct_expert_load = (
            bool(os.environ.get("AXOLOTL_DIRECT_EXPERT_LOAD"))
            and layout["routed_present"]
        )
        self._routed_projs = layout.get("routed_projs", [])
        if self._direct_expert_load:
            reg_layout = dict(layout)
            reg_layout["routed_present"] = (
                False  # don't register the slow routed converters
            )
            register_nvfp4_converters_for_layout(model_type, reg_layout)
            patch_skip_missing_expert_init()
            LOG.info("qwen3_moe: routed experts will be DIRECT-loaded (fast path)")
        else:
            register_nvfp4_converters_for_layout(model_type, layout)

    def post_model_load(self, cfg, model) -> None:
        if getattr(self, "_direct_expert_load", False):
            import time

            from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
                direct_load_nvfp4_experts,
            )

            t0 = time.time()
            n = direct_load_nvfp4_experts(model, cfg.base_model, self._routed_projs)
            LOG.info(
                "qwen3_moe: direct-loaded %d fused expert params in %.1fs (fast path)",
                n,
                time.time() - t0,
            )
