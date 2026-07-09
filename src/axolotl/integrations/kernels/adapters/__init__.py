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


def _all_adapters() -> list[ModelAdapter]:
    from axolotl.integrations.kernels.adapters.dsv4 import DSV4Adapter
    from axolotl.integrations.kernels.adapters.gemma4 import Gemma4Adapter
    from axolotl.integrations.kernels.adapters.glm_moe_dsa import GlmMoeDsaAdapter
    from axolotl.integrations.kernels.adapters.qwen3_moe import Qwen3MoeAdapter

    return [DSV4Adapter(), Gemma4Adapter(), GlmMoeDsaAdapter(), Qwen3MoeAdapter()]


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
