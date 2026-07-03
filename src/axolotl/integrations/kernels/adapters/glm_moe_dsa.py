"""GLM-5.2 (``glm_moe_dsa``) NVFP4 kernel adapter.

GLM-5.2 is the DeepSeek-V3.2 sparse-MLA (DSA / Lightning-Indexer) lineage. A ``quant_method:
modelopt`` / ``quant_algo: NVFP4`` checkpoint is not a recognized transformers quantizer, so the
model loads as a bf16 skeleton; we then register ``WeightConverter``s so the NVFP4 tensors load
correctly. Rather than assume a fixed layout, the adapter inspects the checkpoint's safetensors
index (:func:`inspect_nvfp4_layout`) to discover what is actually quantized — the per-expert
projections (fused into 3D ``NVFP4Tensor`` for the scattermoe path) and any non-routed NVFP4
linears (dequantized to bf16) — and registers exactly those converters for THIS checkpoint.

When ``use_glm_dsa_kernels`` is set, ``post_model_load`` also patches the DSA attention with the
fused absorbed-MLA sparse-gather kernels + fused Lightning-Indexer (``libs/glm_dsa``), keeps the MoE
router fp32, and wires the context-parallel group so the attention shards the sequence on the ``cp``
axis (composing with EP on the orthogonal ``ep`` axis).
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import ModelAdapter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _disable_cudnn_sdp() -> None:
    """Disable the cuDNN SDPA backend for the stock-sdpa GLM path.

    GLM-5.2 develops "massive activations" (residual-stream outliers ~600+ in the top layers). The
    cuDNN flash-attention BACKWARD produces NaN on the resulting extreme q/k (the math/mem-efficient
    backends are robust), which propagates to every gradient → grad_norm=nan. Forward is finite, so
    it only shows in training. Disabling the cuDNN backend makes SDPA fall back to flash/mem-efficient.
    No-op cost on the ``use_glm_dsa_kernels`` path (that replaces SDPA with the fused DSA flash kernel).
    """
    import torch

    try:
        torch.backends.cuda.enable_cudnn_sdp(False)
    except Exception:  # pylint: disable=broad-except
        pass


def is_glm_moe_dsa_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a GLM-5.2 (``glm_moe_dsa``) NVFP4-modelopt checkpoint
    (``quant_method=modelopt``, ``quant_algo=NVFP4``). Any failure returns False."""
    try:
        from transformers import AutoConfig

        # Honor the user's opt-in rather than forcing remote code: glm_moe_dsa is a native
        # transformers model, so config loading needs no remote code by default.
        hf_cfg = AutoConfig.from_pretrained(
            cfg.base_model,
            trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
        )
    except Exception:
        return False
    if str(getattr(hf_cfg, "model_type", "")) != "glm_moe_dsa":
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


def _is_glm_moe_dsa(cfg) -> bool:
    """True iff the base model is a glm_moe_dsa checkpoint (any quant)."""
    try:
        from transformers import AutoConfig

        # Honor the user's opt-in rather than forcing remote code: glm_moe_dsa is a native
        # transformers model, so config loading needs no remote code by default.
        hf_cfg = AutoConfig.from_pretrained(
            cfg.base_model,
            trust_remote_code=bool(getattr(cfg, "trust_remote_code", False)),
        )
    except Exception:
        return False
    return str(getattr(hf_cfg, "model_type", "")) == "glm_moe_dsa"


def _resolve_cp_group(cfg):
    """The context-parallel ProcessGroup (the `cp` axis of accelerate's mesh / the EP mesh), or None
    when ``context_parallel_size <= 1``. The DSA attention shards the sequence on this axis."""
    if (getattr(cfg, "context_parallel_size", None) or 1) <= 1:
        return None
    try:  # the EP plugin owns the mesh when EP is on; else read accelerate's mesh
        from axolotl.integrations.expert_parallel.plugin import ExpertParallelPlugin

        grp = ExpertParallelPlugin._resolve_cp_group(cfg)
        if grp is not None:
            return grp
    except Exception:
        pass
    try:
        from accelerate.state import AcceleratorState

        mesh = getattr(AcceleratorState(), "device_mesh", None)
        if mesh is not None and "cp" in (mesh.mesh_dim_names or ()):
            return mesh["cp"].get_group()
    except Exception:
        pass
    return None


class GlmMoeDsaAdapter(ModelAdapter):
    name = "glm_moe_dsa"

    def matches(self, cfg) -> bool:
        # NVFP4 loading needs scattermoe + a modelopt-NVFP4 checkpoint; the DSA kernels need only
        # use_glm_dsa_kernels on a glm_moe_dsa model. Activate for either.
        if bool(cfg.use_scattermoe) and is_glm_moe_dsa_nvfp4_modelopt(cfg):
            return True
        return bool(cfg.get("use_glm_dsa_kernels")) and _is_glm_moe_dsa(cfg)

    def pre_model_load(self, cfg) -> None:
        # The cuDNN SDPA backward NaNs on GLM's massive activations (see _disable_cudnn_sdp).
        _disable_cudnn_sdp()
        if not (cfg.use_scattermoe and is_glm_moe_dsa_nvfp4_modelopt(cfg)):
            return  # NVFP4 converter registration is only for modelopt-NVFP4 checkpoints
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            inspect_nvfp4_layout,
            patch_nvfp4_tensor_meta_ops,
            patch_skip_missing_expert_init,
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
                "glm_moe_dsa: initialized distributed state for rank0-only loading "
                "(is_fsdp_enabled=%s)",
                is_fsdp_enabled(),
            )

        import os

        layout = inspect_nvfp4_layout(cfg.base_model)
        LOG.info(
            "glm_moe_dsa: detected NVFP4 layout — routed experts: %s (projs=%s); "
            "non-routed NVFP4 linears: %s",
            layout["routed_present"],
            layout["routed_projs"],
            layout["nonrouted_suffixes"],
        )
        # FAST routed-expert load (opt-in): skip the routed converters here and read+fuse the experts
        # DIRECTLY in post_model_load — bypasses transformers' ~7-min per-tensor conversion loop over
        # ~240k expert source tensors (direct path ~25s). Non-routed converters still register.
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
            register_nvfp4_converters_for_layout("glm_moe_dsa", reg_layout)
            patch_skip_missing_expert_init()
            LOG.info("glm_moe_dsa: routed experts will be DIRECT-loaded (fast path)")
        else:
            register_nvfp4_converters_for_layout("glm_moe_dsa", layout)

    def post_model_load(self, cfg, model) -> None:
        """Patch the DSA attention with the fused absorbed-MLA kernels + keep the MoE router fp32.
        Gated on ``use_glm_dsa_kernels``. Wires the context-parallel group so the attention shards
        the sequence on the `cp` axis (composes with EP on the orthogonal `ep` axis)."""
        if getattr(self, "_direct_expert_load", False):
            import time

            from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
                direct_load_nvfp4_experts,
            )

            t0 = time.time()
            n = direct_load_nvfp4_experts(model, cfg.base_model, self._routed_projs)
            LOG.info(
                "glm_moe_dsa: direct-loaded %d fused expert params in %.1fs (fast path)",
                n,
                time.time() - t0,
            )

        if not cfg.get("use_glm_dsa_kernels"):
            return
        from axolotl.integrations.kernels.libs.glm_dsa import (
            keep_router_fp32,
            patch_glm_moe_dsa_attention,
        )

        cp_group = _resolve_cp_group(cfg)
        n = patch_glm_moe_dsa_attention(
            model, use_fused_indexer=True, cp_group=cp_group
        )
        r = keep_router_fp32(model)
        LOG.info(
            "glm_moe_dsa: patched %d attention modules with fused DSA kernels (fused indexer, "
            "context_parallel=%s); kept %d routers fp32",
            n,
            cp_group is not None,
            r,
        )
