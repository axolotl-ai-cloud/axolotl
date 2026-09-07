"""Model adapters for the kernels plugin.

``KernelsPlugin`` orchestrates plugin hooks and generic capabilities (expert-kernel
registration, the quantized-training guard, grouped NVFP4 MoE dispatch). Model-family
specifics (DeepSeek-V4 fused kernels / quantizer / dtype policy, Gemma-4 NVFP4 converters /
non-expert quantization) live in ``ModelAdapter`` subclasses here, so the plugin stays a thin
orchestrator and new models opt in by adding an adapter rather than another inline branch.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ModelAdapter:
    """Base class for model-family kernel adapters. All hooks are no-ops by default.

    An adapter is *active* for a run when :meth:`matches` returns True. The plugin calls the
    hooks of every active adapter at the corresponding lifecycle point.
    """

    #: short name for logging
    name: str = "base"

    def matches(self, cfg) -> bool:  # pragma: no cover - trivial
        return False

    def consumes_nonexpert_quantization(self, cfg) -> bool:
        """Whether this adapter acts on the ``nonexpert_quantization`` intent for ``cfg``.

        The plugin warns when a non-expert quantization policy is configured but no active
        adapter claims it (so a future model can't silently no-op the setting). Default False.
        """
        return False

    def pre_model_load(self, cfg) -> None:
        """Before the model is constructed (register converters/quantizers, patch modules)."""

    def pre_lora_load(self, cfg, model) -> None:
        """After the model loads, before PEFT wraps it (fix expert loading, quantize non-experts)."""

    def post_model_load(self, cfg, model) -> None:
        """After PEFT wraps the model (dtype policy, fused-LoRA kernel swaps)."""


def load_base_model_config(cfg):
    """The base model's HF config via the canonical ``load_model_config``, or None on any failure.

    Adapter matchers must use this rather than a raw ``AutoConfig.from_pretrained``: it reuses the
    HF-cached read ``normalize_config`` already did and honors base_model_config / revision /
    overrides / cfg-gated trust_remote_code, so detection matches how the model actually loads."""
    from axolotl.loaders.utils import load_model_config

    try:
        return load_model_config(cfg)
    except Exception:
        return None


def modelopt_nvfp4_model_config(cfg):
    """The base model's HF config iff it declares a modelopt-NVFP4 quantization
    (``quant_method=modelopt`` / ``quant_algo=NVFP4``), else None. Callers narrow by model_type."""
    model_config = load_base_model_config(cfg)
    if model_config is None:
        return None
    qcfg = getattr(model_config, "quantization_config", None)
    if isinstance(qcfg, dict):
        quant_method, quant_algo = qcfg.get("quant_method"), qcfg.get("quant_algo")
    else:
        quant_method = getattr(qcfg, "quant_method", None)
        quant_algo = getattr(qcfg, "quant_algo", None)
    if quant_method != "modelopt" or quant_algo != "NVFP4":
        return None
    return model_config


def _all_adapters() -> list[ModelAdapter]:
    from axolotl.integrations.kernels.adapters.dsv4 import DSV4Adapter
    from axolotl.integrations.kernels.adapters.gemma4 import Gemma4Adapter
    from axolotl.integrations.kernels.adapters.glm_moe_dsa import GlmMoeDsaAdapter
    from axolotl.integrations.kernels.adapters.nvfp4_moe import MoeNvfp4Adapter
    from axolotl.integrations.kernels.adapters.qwen3_moe import Qwen3MoeAdapter

    # MoeNvfp4Adapter is the generic gate; it excludes model_types the specialized adapters own,
    # so it never double-matches them. Listed last for a stable active-adapter log order.
    return [
        DSV4Adapter(),
        Gemma4Adapter(),
        GlmMoeDsaAdapter(),
        Qwen3MoeAdapter(),
        MoeNvfp4Adapter(),
    ]


def get_active_adapters(cfg) -> list[ModelAdapter]:
    """Return the adapters whose ``matches(cfg)`` is True (order = registration order)."""
    active = []
    for adapter in _all_adapters():
        try:
            if adapter.matches(cfg):
                active.append(adapter)
        except Exception:  # pragma: no cover - matching must never break loading
            LOG.debug("adapter %s match check failed; skipping", adapter.name)
    if active:
        LOG.info("kernels: active model adapters: %s", [a.name for a in active])
    return active
