"""Tests for the ``kernelize()`` repair in ``kernelize_fixes``."""

import pytest

pytest.importorskip("kernels", reason="kernelize fixes only matter with kernels")


@pytest.fixture
def kernelize_patch():
    """Install the patch, restore everything afterwards."""
    from transformers.utils.import_utils import is_kernels_available

    from axolotl.monkeypatch.kernelize_fixes import (
        patch_kernelize_fixes,
        unpatch_kernelize_fixes,
    )

    if not is_kernels_available():
        pytest.skip("transformers does not accept this kernels release")
    # canary: transformers renaming its kernels.kernelize binding no-ops the patch
    assert patch_kernelize_fixes() is True
    yield
    unpatch_kernelize_fixes()


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


def test_kernelize_gets_a_device_it_can_index():
    """transformers passes kernels' own ``Device``, which kernels >= 0.18 reads an
    ``index`` off; a ``torch.device`` or a string goes through unchanged."""
    import torch
    from kernels import Device

    from axolotl.monkeypatch.kernelize_fixes import _kernelize_with_device_type

    seen = []
    kernelize = _kernelize_with_device_type(
        lambda model, *, device, mode: seen.append(device)
    )
    for device in (Device(type="cuda"), torch.device("cuda", 1), "cpu", None):
        kernelize(None, device=device, mode=None)
    assert seen == ["cuda", torch.device("cuda", 1), "cpu", None]


def test_patch_wraps_and_restores_transformers_kernelize(kernelize_patch):
    import kernels
    from transformers.integrations import hub_kernels

    from axolotl.monkeypatch.kernelize_fixes import (
        patch_kernelize_fixes,
        unpatch_kernelize_fixes,
    )

    assert hub_kernels._kernels_kernelize.__wrapped__ is kernels.kernelize
    assert patch_kernelize_fixes() is True
    unpatch_kernelize_fixes()
    assert hub_kernels._kernels_kernelize is kernels.kernelize
    unpatch_kernelize_fixes()


@pytest.mark.parametrize(
    "build", [_tiny_gpt_oss, _tiny_gemma4], ids=["gpt_oss", "gemma4"]
)
def test_set_use_kernels_leaves_weights_alone(kernelize_patch, build):
    import torch

    torch.manual_seed(0)
    model = build().train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model.set_use_kernels(True)
    after = model.state_dict()

    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)
