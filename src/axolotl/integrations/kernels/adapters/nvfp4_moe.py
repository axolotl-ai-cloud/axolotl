"""Generic modelopt-NVFP4 MoE kernel adapter.

The routed-expert load path (safetensors-index layout inspection + ``WeightConverter``s that fuse
per-expert ``gate/up/down`` into the model's 3D expert params) is architecture-agnostic: only the
config.json quant metadata (``quant_method=modelopt`` / ``quant_algo=NVFP4``) and the discovered
expert layout matter, not the ``model_type``. :class:`Nvfp4MoeAdapter` holds that shared loader;
the arch-specific adapters (e.g. ``qwen3_moe``) subclass it, and :class:`MoeNvfp4Adapter` is the
generic gate for any modelopt-NVFP4 MoE the specialized adapters do not already own. Expert
layouts the routed converter cannot fuse (Mixtral ``w1/w2/w3``, fused-stacked or transposed
experts) are rejected loudly at load rather than silently loading random bf16 experts.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import (
    ModelAdapter,
    modelopt_nvfp4_model_config,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# proj names the routed NVFP4 WeightConverter fuses into experts.gate_up_proj / experts.down_proj
SUPPORTED_ROUTED_PROJS = frozenset({"gate_proj", "up_proj", "down_proj"})

# model_types a specialized adapter owns (its own extra patches); the generic gate defers to them
_SPECIALIZED_MOE_MODEL_TYPES = frozenset({"glm_moe_dsa", "qwen3_moe", "qwen3_next"})


def is_moe_nvfp4_modelopt(cfg) -> bool:
    """True iff the base is a modelopt-NVFP4 checkpoint NOT owned by a specialized adapter.

    Arch-agnostic: any modelopt/NVFP4 ``model_type`` outside the specialized set (``gemma4*`` /
    ``glm_moe_dsa`` / ``qwen3_moe`` / ``qwen3_next``) is claimed here; whether its expert layout
    is actually loadable is validated at load time (:meth:`Nvfp4MoeAdapter.pre_model_load`). Any
    failure returns False."""
    model_config = modelopt_nvfp4_model_config(cfg)
    if model_config is None:
        return False
    model_type = str(getattr(model_config, "model_type", ""))
    return not (
        model_type.startswith("gemma4") or model_type in _SPECIALIZED_MOE_MODEL_TYPES
    )


class Nvfp4MoeAdapter(ModelAdapter):
    """Shared modelopt-NVFP4 routed-expert loader: layout inspection + WeightConverter registration.

    Base for the arch-specific adapters (``qwen3_moe``) and the generic gate (``moe_nvfp4``). The
    hooks are already layout-driven; the only per-model knob is which ``model_type`` the loader
    looks the conversion mapping up under.
    """

    name = "nvfp4_moe"

    def _validate_supported_layout(self, cfg, model_type: str, layout: dict) -> None:
        """Reject checkpoints whose expert layout the routed NVFP4 converter cannot fuse.

        First consults the arch's ``ModelSupport`` (an arch may declare ``expert_kernels``
        Unsupported/Experimental); an unknown key is a no-op and we fall back to the proj check."""
        from axolotl.model_support import check_capability, get_model_support

        check_capability(
            get_model_support(model_type),
            "expert_kernels",
            model_type,
            feature="sonicmoe/scattermoe NVFP4 experts",
        )
        projs = set(layout.get("routed_projs", []))
        if projs and not projs <= SUPPORTED_ROUTED_PROJS:
            raise ValueError(
                f"{self.name}: unsupported routed NVFP4 expert layout {sorted(projs)} for "
                f"model_type={model_type}. The routed NVFP4 converter currently fuses only "
                f"{sorted(SUPPORTED_ROUTED_PROJS)} into experts.gate_up_proj/experts.down_proj; "
                "other namings (e.g. Mixtral w1/w2/w3) are not yet wired. Remove use_sonicmoe/"
                "use_scattermoe for this checkpoint, or add converter support for this layout."
            )

    def pre_model_load(self, cfg) -> None:
        import os

        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            inspect_nvfp4_layout,
            patch_nvfp4_tensor_meta_ops,
            patch_skip_missing_expert_init,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (
            patch_conversion_loader_rank0_only,
            register_nvfp4_converters_for_layout,
        )

        # transformers looks the conversion mapping up under the model's type; normalize_config
        # already resolved it into cfg.model_config_type, so reuse that instead of re-reading.
        model_type = str(cfg.model_config_type)

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
                "%s: initialized distributed state for rank0-only loading (is_fsdp_enabled=%s)",
                self.name,
                is_fsdp_enabled(),
            )

        layout = inspect_nvfp4_layout(cfg.base_model)
        LOG.info(
            "%s: detected NVFP4 layout — routed experts: %s (projs=%s); "
            "non-routed NVFP4 linears: %s",
            self.name,
            layout["routed_present"],
            layout["routed_projs"],
            layout["nonrouted_suffixes"],
        )
        self._validate_supported_layout(cfg, model_type, layout)

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
            LOG.info("%s: routed experts will be DIRECT-loaded (fast path)", self.name)
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
                "%s: direct-loaded %d fused expert params in %.1fs (fast path)",
                self.name,
                n,
                time.time() - t0,
            )


class MoeNvfp4Adapter(Nvfp4MoeAdapter):
    """Generic gate: any modelopt-NVFP4 MoE not owned by a specialized adapter."""

    name = "moe_nvfp4"

    def matches(self, cfg) -> bool:
        return bool(cfg.use_scattermoe or cfg.use_sonicmoe) and is_moe_nvfp4_modelopt(
            cfg
        )
