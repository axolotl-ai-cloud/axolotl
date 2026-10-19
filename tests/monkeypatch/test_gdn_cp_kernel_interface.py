"""GDN kernel hooks used by Ringmaster context parallelism."""

import pytest
import torch


class _Norm(torch.nn.Module):
    def forward(self, hidden_states, gate):
        return hidden_states


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_next"])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_packed_gdn_prefers_cp_kernels(family, torch_compile, monkeypatch):
    if family == "qwen3_5":
        from transformers.models.qwen3_5 import modeling_qwen3_5 as hf
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
        )

        from axolotl.monkeypatch.models.qwen3_5 import modeling as qm

        config = Qwen3_5TextConfig(
            hidden_size=64,
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
        )
        gated_cls = hf.Qwen3_5GatedDeltaNet
        patch = qm.patch_qwen3_5_modeling_packing
    else:
        from transformers.models.qwen3_next import modeling_qwen3_next as hf
        from transformers.models.qwen3_next.configuration_qwen3_next import (
            Qwen3NextConfig,
        )

        from axolotl.monkeypatch.models.qwen3_next import modeling as qm

        config = Qwen3NextConfig(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=64,
            linear_conv_kernel_dim=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            layer_types=["linear_attention"],
        )
        gated_cls = hf.Qwen3NextGatedDeltaNet
        patch = qm.patch_qwen3_next_modeling_packing

    original_forward = gated_cls.forward
    original_ops = qm._FLA_COMPILED_OPS
    monkeypatch.setattr(qm, "_init_fla_compiled_ops", lambda enabled: enabled)
    monkeypatch.setattr(
        qm,
        "get_cu_seqlens",
        lambda position_ids: pytest.fail("CP should use global document boundaries"),
    )

    try:
        patch(torch_compile=torch_compile)
        mixer = gated_cls(config, 0)
        mixer.norm = _Norm()
        calls = []

        def conv(x, weight, bias, activation):
            calls.append("conv")
            return x

        def chunk(q, k, v, **kwargs):
            calls.append("chunk")
            return v, None

        mixer._axolotl_gdn_cp_conv = conv
        mixer._axolotl_gdn_cp_chunk = chunk
        assert mixer.forward._axolotl_gdn_kernel_interface
        with torch.no_grad():
            output = mixer(
                torch.randn(1, 8, config.hidden_size),
                position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
            )
        assert output.shape == (1, 8, config.hidden_size)
        assert calls == ["conv", "chunk"]
    finally:
        gated_cls.forward = original_forward
        qm._FLA_COMPILED_OPS = original_ops
