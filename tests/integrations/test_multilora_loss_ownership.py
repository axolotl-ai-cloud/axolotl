"""Multi-LoRA owns segmented CE while core preserves other kernel patches."""

import pytest

from axolotl.utils.dict import DictDefault


@pytest.mark.parametrize("adapter", ["lora", "multilora"])
def test_cce_validates_but_does_not_patch_multilora_forward(monkeypatch, adapter):
    from cut_cross_entropy.transformers import patch

    from axolotl.integrations.cut_cross_entropy import CutCrossEntropyPlugin

    plugin = CutCrossEntropyPlugin()
    calls = []
    monkeypatch.setattr(
        plugin, "_check_requirements", lambda: calls.append("requirements")
    )
    monkeypatch.setattr(plugin, "patch_llama_like", lambda _: calls.append("register"))
    monkeypatch.setattr(patch, "cce_patch", lambda *a, **k: calls.append("patch"))
    plugin.pre_model_load(
        DictDefault(adapter=adapter, cut_cross_entropy=True, model_config_type="llama")
    )
    assert calls == (
        ["requirements"]
        if adapter == "multilora"
        else ["requirements", "register", "patch"]
    )


@pytest.mark.parametrize("adapter", ["lora", "multilora"])
def test_liger_preserves_other_kernels_and_requested_config(monkeypatch, adapter):
    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN

    from axolotl.integrations.liger import LigerPlugin

    calls = []

    def apply(*, fused_linear_cross_entropy, rms_norm, swiglu):
        calls.append((fused_linear_cross_entropy, rms_norm, swiglu))

    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "llama", apply)
    cfg = DictDefault(
        adapter=adapter,
        model_config_type="llama",
        liger_fused_linear_cross_entropy=True,
        liger_rms_norm=True,
        liger_glu_activation=True,
    )
    from liger_kernel.transformers import functional

    original_loss = functional.liger_fused_linear_cross_entropy
    if adapter == "multilora":
        cfg.liger_use_token_scaling = True
    LigerPlugin().pre_model_load(cfg)
    if adapter == "multilora":
        assert functional.liger_fused_linear_cross_entropy is original_loss
    assert calls == [(adapter != "multilora", True, True)]
    assert cfg.liger_fused_linear_cross_entropy is True


def test_core_ce_conflict_validator_still_rejects_multilora():
    from axolotl.utils.schemas.validation import OptimizationValidationMixin

    with pytest.raises(ValueError, match="Only one cross entropy"):
        OptimizationValidationMixin.check_cross_entropy_conflicts(
            dict(
                adapter="multilora",
                cut_cross_entropy=True,
                liger_fused_linear_cross_entropy=True,
            )
        )
