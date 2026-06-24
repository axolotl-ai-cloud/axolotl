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


def is_glm_moe_dsa_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a GLM-5.2 (``glm_moe_dsa``) NVFP4-modelopt checkpoint
    (``quant_method=modelopt``, ``quant_algo=NVFP4``). Any failure returns False."""
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
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

        hf_cfg = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
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
        if not (cfg.use_scattermoe and is_glm_moe_dsa_nvfp4_modelopt(cfg)):
            return  # NVFP4 converter registration is only for modelopt-NVFP4 checkpoints
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            inspect_nvfp4_layout,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (
            register_nvfp4_converters_for_layout,
        )

        layout = inspect_nvfp4_layout(cfg.base_model)
        LOG.info(
            "glm_moe_dsa: detected NVFP4 layout — routed experts: %s (projs=%s); "
            "non-routed NVFP4 linears: %s",
            layout["routed_present"],
            layout["routed_projs"],
            layout["nonrouted_suffixes"],
        )
        register_nvfp4_converters_for_layout("glm_moe_dsa", layout)

    def post_model_load(self, cfg, model) -> None:
        """Patch the DSA attention with the fused absorbed-MLA kernels + keep the MoE router fp32.
        Gated on ``use_glm_dsa_kernels``. Wires the context-parallel group so the attention shards
        the sequence on the `cp` axis (composes with EP on the orthogonal `ep` axis)."""
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
