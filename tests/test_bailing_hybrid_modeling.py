"""Ling 3.0 (`bailing_hybrid`) modeling: checkpoint layout, packing, and gate math."""

import pytest
import torch

# `import fla` succeeds without triton; the kernels and modules used here do not
pytest.importorskip("fla.ops.kda", reason="Ling 3.0 linear attention needs fla-core")

from safetensors.torch import load_file, save_file  # noqa: E402
from transformers.conversion_mapping import (  # noqa: E402
    register_checkpoint_conversion_mapping,
)

from axolotl.model_support.bailing_hybrid import _weight_conversions  # noqa: E402
from axolotl.model_support.bailing_hybrid.configuration_bailing_moe_v3 import (  # noqa: E402
    BailingMoeV3Config,
)
from axolotl.model_support.bailing_hybrid.modeling_bailing_moe_v3 import (  # noqa: E402
    BailingMoeV3ForCausalLM,
    BailingMoeV3KimiDeltaAttention,
    cu_seqlens_from_position_ids,
)

CONFIG_KWARGS = dict(
    vocab_size=64,
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=8,
    q_lora_rank=16,
    kv_lora_rank=16,
    qk_nope_head_dim=8,
    qk_rope_head_dim=4,
    v_head_dim=8,
    layer_group_size=2,
    num_experts=8,
    num_experts_per_tok=2,
    n_group=2,
    topk_group=1,
    moe_intermediate_size=16,
    moe_shared_expert_intermediate_size=16,
    first_k_dense_replace=1,
    max_position_embeddings=128,
    pad_token_id=0,
    eos_token_id=1,
    attn_implementation="eager",
)


@pytest.fixture(name="config")
def config_fixture():
    return BailingMoeV3Config(**CONFIG_KWARGS)


@pytest.fixture(name="model")
def model_fixture(config):
    torch.manual_seed(0)
    return BailingMoeV3ForCausalLM(config).eval()


def published_layout(model, config):
    """The per-expert modules and `expert_bias` router buffer a Ling checkpoint ships."""
    checkpoint = {}
    for name, tensor in model.state_dict().items():
        prefix = name.rsplit(".", 1)[0]
        if name.endswith("mlp.experts.gate_up_proj"):
            gate, up = tensor.chunk(2, dim=1)
            for expert in range(config.num_experts):
                checkpoint[f"{prefix}.{expert}.gate_proj.weight"] = gate[expert].clone()
                checkpoint[f"{prefix}.{expert}.up_proj.weight"] = up[expert].clone()
        elif name.endswith("mlp.experts.down_proj"):
            for expert in range(config.num_experts):
                checkpoint[f"{prefix}.{expert}.down_proj.weight"] = tensor[
                    expert
                ].clone()
        elif name.endswith("mlp.gate.e_score_correction_bias"):
            checkpoint[f"{prefix}.expert_bias"] = tensor.clone()
        else:
            checkpoint[name] = tensor.clone()
    return checkpoint


class TestBailingHybridModeling:
    """The published checkpoint must load and save without weight surgery."""

    @pytest.fixture(autouse=True)
    def register_conversions(self):
        for key, entries in _weight_conversions().items():
            register_checkpoint_conversion_mapping(key, list(entries), overwrite=True)

    def test_layer_types_alternate_kda_and_mla(self, config, model):
        assert config.layer_types == [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
        assert [layer.is_linear_attention for layer in model.model.layers] == [
            True,
            False,
            True,
            False,
        ]

    def test_published_checkpoint_loads_unchanged(self, config, model, tmp_path):
        save_file(published_layout(model, config), tmp_path / "model.safetensors")
        config.save_pretrained(tmp_path)

        reloaded = BailingMoeV3ForCausalLM.from_pretrained(
            tmp_path, dtype=torch.float32
        )

        expected = model.state_dict()
        assert reloaded.state_dict().keys() == expected.keys()
        for name, tensor in reloaded.state_dict().items():
            assert torch.equal(tensor, expected[name]), name

    def test_save_restores_published_layout(self, config, model, tmp_path):
        model.save_pretrained(tmp_path)

        saved = load_file(tmp_path / "model.safetensors")
        assert saved.keys() == published_layout(model, config).keys()

    def test_layer_types_ignore_a_stale_serialized_value(self, config):
        """`layer_types` sizes the hybrid cache while the decoder layers build from
        `is_linear_attention_layer`; a serialized value must not desynchronise them."""
        stale = ["full_attention"] * CONFIG_KWARGS["num_hidden_layers"]

        reloaded = BailingMoeV3Config(**{**CONFIG_KWARGS, "layer_types": stale})

        assert reloaded.layer_types == config.layer_types
        assert reloaded.layer_types != stale

    def test_layer_types_survive_a_round_trip(self, config, tmp_path):
        config.save_pretrained(tmp_path)

        assert BailingMoeV3Config.from_pretrained(tmp_path).layer_types == (
            config.layer_types
        )

    def test_multi_token_prediction_weights_are_not_reported_missing(self):
        config = BailingMoeV3Config(**{**CONFIG_KWARGS, "num_nextn_predict_layers": 1})
        model = BailingMoeV3ForCausalLM(config)

        patterns = model._keys_to_ignore_on_load_unexpected  # pylint: disable=protected-access
        assert patterns
        import re

        matcher = re.compile("|".join(f"({p})" for p in patterns))
        assert matcher.search("model.layers.4.mlp.gate.expert_bias")
        assert matcher.search("model.layers.4.eh_proj.weight")
        # the real layers must still be reported
        assert matcher.search("model.layers.3.mlp.gate.expert_bias") is None


class TestKimiDeltaAttentionGate:
    """`kda_safe_gate` swaps the activation; it does not clamp the softplus one."""

    @pytest.fixture(name="attention")
    def attention_fixture(self):
        torch.manual_seed(0)
        config = BailingMoeV3Config(**{**CONFIG_KWARGS, "no_kda_lora": True})
        assert config.kda_safe_gate and config.kda_lower_bound == -5.0
        attention = BailingMoeV3KimiDeltaAttention(config, layer_idx=0)
        with torch.no_grad():
            attention.A_log.uniform_(1, 16).log_()
            attention.dt_bias.normal_()
        return attention

    def test_matches_the_lower_bound_activation(self, attention):
        """`lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`, per fla's
        `naive_kda_lowerbound_gate` / the `safe_gate` kwarg docs."""
        hidden = torch.randn(2, 6, attention.config.hidden_size)

        gate = attention._forget_gate(hidden)  # pylint: disable=protected-access

        raw = attention.f_proj(hidden).view(2, 6, attention.num_heads, -1)
        expected = attention.lower_bound * torch.sigmoid(
            attention.A_log.view(-1, 1).exp()
            * (raw.float() + attention.dt_bias.view(attention.num_heads, -1))
        )
        torch.testing.assert_close(gate, expected)

    def test_zero_pre_activation_is_half_the_bound(self, attention):
        """`lower_bound * sigmoid(0)` whatever `A_log` holds. A clamped softplus
        would land on the `A_log`-dependent `-exp(A_log) * ln(2)` instead."""
        with torch.no_grad():
            attention.f_proj.weight.zero_()
            attention.dt_bias.zero_()
        hidden = torch.randn(2, 5, attention.config.hidden_size)

        gate = attention._forget_gate(hidden)  # pylint: disable=protected-access

        torch.testing.assert_close(
            gate, torch.full_like(gate, attention.lower_bound / 2)
        )

    def test_stays_within_the_bound(self, attention):
        hidden = torch.randn(4, 16, attention.config.hidden_size) * 20

        gate = attention._forget_gate(hidden)  # pylint: disable=protected-access

        assert (gate >= attention.lower_bound).all()
        assert (gate <= 0).all()


class TestCuSeqlensFromPositionIds:
    """Packing is signalled by position_ids: axolotl drops the packed mask."""

    def test_splits_on_every_restart(self):
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0, 1, 2]])

        assert cu_seqlens_from_position_ids(position_ids).tolist() == [0, 4, 6, 9]

    def test_unpacked_batch_needs_no_varlen(self):
        assert cu_seqlens_from_position_ids(torch.tensor([[0, 1, 2, 3]])) is None
        assert cu_seqlens_from_position_ids(None) is None
        # fla's varlen API only accepts a flattened batch
        assert cu_seqlens_from_position_ids(torch.tensor([[0, 1], [0, 1]])) is None

    def test_chunk_starting_mid_document(self):
        """Under context parallelism a rank's chunk may not begin at position 0."""
        position_ids = torch.tensor([[3, 4, 5, 0, 1, 2]])

        assert cu_seqlens_from_position_ids(position_ids).tolist() == [0, 3, 6]

    def test_batched_packing_splits_rows_and_restarts(self):
        """micro_batch_size > 1 with packing: the rows become one varlen stream, so
        a row boundary is a document boundary alongside every restart."""
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 0, 1, 2, 3]])

        assert cu_seqlens_from_position_ids(position_ids).tolist() == [0, 3, 6, 8, 12]


class TestPositionIdsReachAttention:
    """Flash attention infers packed sequences from position_ids alone: a 2D mask
    that survives `create_causal_mask` is bool-cast, so the segment ids are gone by
    the time `_flash_attention_forward` runs. A layer loop that keeps position_ids
    to itself silently trains with cross-document attention."""

    def test_both_attention_branches_receive_position_ids(self, model):
        seen = []

        class Recorder(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):  # noqa: D102
                seen.append(kwargs.get("position_ids"))
                return torch.zeros_like(hidden_states)

        branches = [layer.is_linear_attention for layer in model.model.layers]
        assert True in branches and False in branches, "need both layer types"
        for layer in model.model.layers:
            layer.attention = Recorder()

        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        model(input_ids=torch.tensor([[5, 9, 13, 7, 2]]), position_ids=position_ids)

        assert len(seen) == len(branches)
        for recorded in seen:
            assert recorded is not None
            torch.testing.assert_close(recorded, position_ids)


class TestPackedDocumentIsolation:
    """`layer_group_size=1` makes every layer multi-latent, so this runs on CPU."""

    @pytest.fixture(name="model")
    def mla_only_model(self):
        torch.manual_seed(0)
        config = BailingMoeV3Config(**{**CONFIG_KWARGS, "layer_group_size": 1})
        assert config.layer_types == ["full_attention"] * config.num_hidden_layers
        return BailingMoeV3ForCausalLM(config).eval()

    def test_packed_documents_match_separate_forwards(self, model):
        first = torch.tensor([[5, 9, 13, 21]])
        second = torch.tensor([[7, 33, 2]])
        packed = torch.cat([first, second], dim=1)
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])

        logits = model(input_ids=packed, position_ids=position_ids).logits

        torch.testing.assert_close(logits[:, :4], model(input_ids=first).logits)
        torch.testing.assert_close(logits[:, 4:], model(input_ids=second).logits)

    def test_second_document_is_unaffected_by_the_first(self, model):
        """Guards the failure mode directly: without per-document masking, editing
        the first document moves the second document's logits."""
        position_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
        packed = torch.tensor([[5, 9, 13, 21, 7, 33, 2]])
        edited = packed.clone()
        edited[0, 0] = 42

        baseline = model(input_ids=packed, position_ids=position_ids).logits
        changed = model(input_ids=edited, position_ids=position_ids).logits

        torch.testing.assert_close(changed[:, 4:], baseline[:, 4:])
        assert not torch.allclose(changed[:, :4], baseline[:, :4])


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for fla kernels"
)
class TestBailingHybridForward:
    """Mask handling through the real linear-attention kernels."""

    @pytest.fixture(name="model")
    def cuda_model_fixture(self, model):
        return model.cuda()

    def test_packed_batch_splits_documents(self, model):
        """The hybrid model as axolotl feeds it: no attention mask, position_ids
        restarting per document."""
        first = torch.randint(2, 64, (1, 5), device=model.device)
        second = torch.randint(2, 64, (1, 3), device=model.device)
        packed = torch.cat([first, second], dim=1)
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], device=model.device)

        logits = model(input_ids=packed, position_ids=position_ids).logits

        torch.testing.assert_close(logits[:, :5], model(input_ids=first).logits)
        torch.testing.assert_close(logits[:, 5:], model(input_ids=second).logits)

    def test_padding_is_ignored(self, model):
        input_ids = torch.randint(2, 64, (1, 6), device=model.device)
        padded = torch.cat(
            [input_ids, torch.zeros(1, 2, dtype=torch.long, device=model.device)], dim=1
        )
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device=model.device)

        unpadded_logits = model(input_ids=input_ids).logits
        padded_logits = model(input_ids=padded, attention_mask=attention_mask).logits

        torch.testing.assert_close(padded_logits[:, :6], unpadded_logits)

    def test_segment_id_mask_still_splits_documents(self, model):
        """The mask-based path stays supported for callers that keep the mask."""
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]], device=model.device)
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2]], device=model.device)
        input_ids = torch.randint(2, 64, (1, 8), device=model.device)

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).logits

        assert logits.shape == (1, 8, 64)
        assert torch.isfinite(logits).all()

    def test_packed_documents_are_isolated_at_micro_batch_size_2(self, model):
        """Without a flattened varlen stream the KDA recurrence runs straight through
        the batch, so editing document 1 moves document 2's logits."""
        position_ids = torch.arange(4, device=model.device).repeat(2).expand(2, 8)
        input_ids = torch.randint(2, 64, (2, 8), device=model.device)
        edited = input_ids.clone()
        edited[0, :4] = torch.randint(2, 64, (4,), device=model.device)

        baseline = model(input_ids=input_ids, position_ids=position_ids).logits
        changed = model(input_ids=edited, position_ids=position_ids).logits

        torch.testing.assert_close(changed[0, 4:], baseline[0, 4:])
        torch.testing.assert_close(changed[1], baseline[1])
        assert not torch.allclose(changed[0, :4], baseline[0, :4])

    def test_batched_packing_matches_micro_batch_size_1(self, model):
        """A packed row must produce the same logits whether it is batched or not."""
        position_ids = torch.arange(4, device=model.device).repeat(2).expand(2, 8)
        input_ids = torch.randint(2, 64, (2, 8), device=model.device)

        batched = model(input_ids=input_ids, position_ids=position_ids).logits

        for row in range(2):
            single = model(
                input_ids=input_ids[row : row + 1],
                position_ids=position_ids[row : row + 1],
            ).logits
            torch.testing.assert_close(batched[row : row + 1], single)

    def test_causal_masking(self, model):
        """A hardcoded eager interface is how the published code loses causality."""
        input_ids = torch.randint(2, 64, (1, 8), device=model.device)
        baseline = model(input_ids=input_ids).logits

        edited = input_ids.clone()
        edited[0, -1] = (edited[0, -1] + 1) % 64
        torch.testing.assert_close(
            model(input_ids=edited).logits[:, :-1], baseline[:, :-1]
        )
