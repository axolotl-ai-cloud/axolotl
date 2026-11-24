"""Native diffusion rejects integrations without verified objective/mask support."""

import pytest

from axolotl.model_support import get_model_support, resolve_model_support
from axolotl.model_support.base import (
    Experimental,
    Supported,
    Unsupported,
    check_capability,
)


@pytest.mark.parametrize(
    "model_type", ["diffusion_gemma", "Dream", "nemotron_labs_diffusion"]
)
@pytest.mark.parametrize(
    "feature",
    ["cut_cross_entropy", "liger", "context_parallel", "expert_kernels"],
)
def test_native_diffusion_rejects_unverified_integrations(model_type, feature):
    support = get_model_support(model_type)
    assert support is not None
    resolved = resolve_model_support(support)
    if model_type == "nemotron_labs_diffusion" and feature == "cut_cross_entropy":
        assert isinstance(resolved.capabilities[feature], Supported)
        check_capability(support, feature, model_type)
        return
    if model_type == "diffusion_gemma" and feature == "expert_kernels":
        # ScatterMoE is validated against the stock experts (tied expert LoRA).
        assert isinstance(resolved.capabilities[feature], Experimental)
        check_capability(support, feature, model_type)
        return
    assert isinstance(resolved.capabilities[feature], Unsupported)
    with pytest.raises(ValueError, match=f"{feature} is not supported"):
        check_capability(support, feature, model_type)


def test_twotower_allows_expert_kernels():
    support = get_model_support("nemotron_h_twotower")
    assert isinstance(
        resolve_model_support(support).capabilities["expert_kernels"], Experimental
    )
    check_capability(support, "expert_kernels", "nemotron_h_twotower")
