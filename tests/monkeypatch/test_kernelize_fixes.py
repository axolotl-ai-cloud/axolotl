"""Tests for the generic ``kernelize()`` repairs in ``kernelize_fixes``.

Two upstream defects are covered (see the patch module docstring): bare
functions stashed in ``_hidden_kernels`` (gemma4 and ~30 others) and the
gpt-oss rotary ``Func`` whose ``position_ids`` parameter fails the kernels
library's signature check against the hub kernel.
"""

import inspect

import pytest
import transformers
from packaging.version import Version
from transformers.modeling_utils import PreTrainedModel

pytest.importorskip("kernels", reason="kernelize fixes only matter with kernels")

# Canary: if transformers drops set_use_kernels, the patch silently no-ops. Skip dev
# builds, fail a stable release so we re-target.
if not hasattr(PreTrainedModel, "set_use_kernels"):
    if Version(transformers.__version__).is_prerelease:
        pytest.skip(
            "PreTrainedModel.set_use_kernels removed on transformers main; patch no-ops",
            allow_module_level=True,
        )
    pytest.fail(
        "PreTrainedModel.set_use_kernels is gone in a stable release and "
        "patch_kernelize_fixes() now silently no-ops. Re-target it.",
        pytrace=False,
    )


@pytest.fixture
def kernelize_patch():
    """Install the patch, restore everything afterwards."""
    from axolotl.monkeypatch.kernelize_fixes import (
        patch_kernelize_fixes,
        unpatch_kernelize_fixes,
    )

    saved_sig = None
    try:
        from transformers.models.gpt_oss import modeling_gpt_oss

        func = modeling_gpt_oss.apply_rotary_pos_emb
        if hasattr(type(func), "forward"):
            saved_sig = inspect.signature(type(func).forward)
    except ImportError:
        func = None

    assert patch_kernelize_fixes() is True
    yield

    unpatch_kernelize_fixes()
    if func is not None and saved_sig is not None:
        type(func).forward.__signature__ = saved_sig


def _tiny_gpt_oss():
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

    cfg = GptOssConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    return GptOssForCausalLM(cfg)


def test_patch_is_idempotent(kernelize_patch):
    from axolotl.monkeypatch.kernelize_fixes import patch_kernelize_fixes

    assert patch_kernelize_fixes() is True


def test_unpatch_restores_original():
    from transformers.modeling_utils import PreTrainedModel

    from axolotl.monkeypatch.kernelize_fixes import (
        patch_kernelize_fixes,
        unpatch_kernelize_fixes,
    )

    original = PreTrainedModel.set_use_kernels
    patch_kernelize_fixes()
    assert PreTrainedModel.set_use_kernels is not original
    unpatch_kernelize_fixes()
    assert PreTrainedModel.set_use_kernels is original
    # Safe to call again without a prior patch.
    unpatch_kernelize_fixes()


def test_gpt_oss_kernelize_and_rotary_signature(kernelize_patch):
    """gpt-oss: kernelize() succeeds and the rotary signature matches the hub
    kernel (kernels-community/rotary) afterwards."""
    pytest.importorskip("transformers.models.gpt_oss")
    from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

    model = _tiny_gpt_oss()
    model.train()
    model.set_use_kernels(True)

    params = inspect.signature(type(apply_rotary_pos_emb).forward).parameters
    assert list(params) == ["self", "q", "k", "cos", "sin", "unsqueeze_dim"]


def test_bare_function_entries_are_dropped(kernelize_patch):
    """Architectures that stash a bare function (gemma4 and ~30 others) no
    longer crash kernelize(); simulated by planting one on gpt-oss."""
    model = _tiny_gpt_oss()
    attn = model.model.layers[0].self_attn

    def bare(q, k, cos, sin):
        return q, k

    attn.__dict__.setdefault("_hidden_kernels", {})["bare"] = bare
    model.train()
    model.set_use_kernels(True)
    assert "bare" not in attn._hidden_kernels


def _tiny_gemma4():
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig,
        Gemma4Config,
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration,
    )

    text = Gemma4TextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        num_experts=4,
        num_experts_per_tok=2,
    )
    vis = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    aud = Gemma4AudioConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4
    )
    return Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text, vision_config=vis, audio_config=aud)
    )


def test_gemma4_kernelize_succeeds_with_patch():
    """The real gemma4 bare-function case end to end: with the generic patch,
    kernelize() succeeds. The unpatched call raises on transformers releases that
    still carry the bug and succeeds once the upstream fix lands, so that half is
    tolerated rather than required."""
    pytest.importorskip("transformers.models.gemma4")
    from axolotl.monkeypatch.kernelize_fixes import (
        patch_kernelize_fixes,
        unpatch_kernelize_fixes,
    )

    model = _tiny_gemma4()
    model.train()
    try:
        # transformers <= 5.8.x raises TypeError, >= 5.9 ValueError; fixed on main.
        model.set_use_kernels(True)
    except (TypeError, ValueError, AttributeError):
        pass

    patch_kernelize_fixes()
    try:
        model = _tiny_gemma4()
        model.train()
        model.set_use_kernels(True)
    finally:
        unpatch_kernelize_fixes()


def test_patch_does_not_alter_weights(kernelize_patch):
    """The repairs only touch ``_hidden_kernels`` and signature metadata;
    parameters are untouched by kernelize()."""
    import torch

    torch.manual_seed(0)
    model = _tiny_gpt_oss()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.train()
    model.set_use_kernels(True)
    after = model.state_dict()

    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
