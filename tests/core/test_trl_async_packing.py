"""Async GRPO packing for models with recurrent layers."""

import types
from collections import defaultdict

import numpy as np
import pytest
import torch
from trl.experimental.async_grpo.async_grpo_trainer import (
    FixedCountBatcher,
    TokenBudgetBatcher,
)
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample
from trl.experimental.async_grpo.packing import TreeRow
from trl_async_test_utils import StubTransfer, StubWorker, reference_tree_attention

from axolotl.core.trainers.grpo.trl_async_packing import (
    ChainFixedCountBatcher,
    ChainRow,
    ChainTokenBudgetBatcher,
    sample_boundaries,
    shares_prefixes,
)
from axolotl.core.trainers.grpo.trl_async_scoring import pack_sequences
from axolotl.core.trainers.grpo.trl_async_trainer import (
    AxolotlAsyncGRPOConfig,
    AxolotlAsyncGRPOTrainer,
)
from axolotl.monkeypatch.models import gated_delta_net_tree  # noqa: F401  registers GDN


def _sample(ids, prompt_len, group_id=0):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.asarray(
            [0] * prompt_len + [1] * (len(ids) - prompt_len), dtype=np.int64
        ),
        "old_log_probs": np.zeros(len(ids), np.float32),
        "advantage": 1.0,
        "group_id": group_id,
        "metrics": {},
    }


# Two rollouts of one prompt, which a TreeRow would share, and another prompt.
SAMPLES = [
    _sample([1, 2, 3, 4, 5], 3),
    _sample([1, 2, 3, 6], 3),
    _sample([7, 8, 9], 1, group_id=1),
]


class TestChainRow:
    def test_packs_each_sample_as_its_own_chain(self):
        tree, chain = TreeRow(), ChainRow()
        for sample in SAMPLES:
            tree.add(sample)
            chain.add(sample)
        assert len(tree.tokens) == 10 and len(chain.tokens) == 12

        packed = chain.pack()
        assert packed["input_ids"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3, 6, 7, 8, 9]
        assert packed["position_ids"].tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2]
        # Tree attention degenerates to varlen attention over the samples.
        assert packed["tree_kv_index"].tolist() == list(range(12))
        assert packed["tree_cu_seqlens_q"].tolist() == [0, 5, 9, 12]
        assert packed["tree_cu_seqlens_k"].tolist() == [0, 5, 9, 12]
        assert packed["tree_max_seqlens"].tolist() == [5, 5]

        # The same loss terms as the trie, scored off unshared positions.
        shared = tree.pack()
        for key in ("old_log_probs", "advantages", "segment_id"):
            assert torch.equal(packed[key], shared[key])
        assert torch.equal(
            packed["shift_labels"][packed["token_index"]],
            shared["shift_labels"][shared["token_index"]],
        )

    def test_budget_counts_every_token(self):
        tree, chain = TreeRow(), ChainRow()
        tree.add(SAMPLES[0])
        chain.add(SAMPLES[0])
        # The trie forks one token early, so [1, 2, 3, 6] shares [1, 2] with the row.
        assert tree.fits(SAMPLES[1], budget=7)
        assert not chain.fits(SAMPLES[1], budget=8)
        assert chain.fits(SAMPLES[1], budget=9)

    def test_pack_sequences_without_sharing(self):
        sequences = [np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])]
        shared = pack_sequences(sequences)
        chains = pack_sequences(sequences, share_prefixes=False)
        assert len(shared["input_ids"]) == 6
        assert chains["input_ids"].tolist() == [1, 2, 3, 4, 1, 2, 3, 5]
        assert chains["position_ids"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert chains["tree_cu_seqlens_q"].tolist() == [0, 4, 8]
        assert chains["positions"].tolist() == list(range(8))


def _rows(batcher):
    return [[[s["group_id"] for s in row.samples] for row in rows] for rows in batcher]


class TestChainBatchers:
    # Distinct first tokens: no prefix to share, so TRL's batchers place them the same.
    SAMPLES = [
        _sample([10 * i + 1, *range(2, 3 + i % 5)], 1, group_id=i // 2)
        for i in range(12)
    ]

    def test_token_budget_batcher_places_like_trl(self):
        for budget in (6, 9, 20):
            ours = ChainTokenBudgetBatcher(self.SAMPLES, 2, budget, defaultdict(list))
            trl = TokenBudgetBatcher(self.SAMPLES, 2, budget, defaultdict(list))
            assert _rows(ours) == _rows(trl)
            assert all(isinstance(row, ChainRow) for rows in ours for row in rows)

    def test_fixed_count_batcher_places_like_trl(self):
        ours = ChainFixedCountBatcher(self.SAMPLES, 2, 4)
        assert _rows(ours) == _rows(FixedCountBatcher(self.SAMPLES, 2, 4))
        assert all(isinstance(row, ChainRow) for rows in ours for row in rows)


def _model(model_type, layer_types=None, mixers=()):
    config = types.SimpleNamespace(model_type=model_type, layer_types=layer_types)
    config.get_text_config = lambda: config
    return types.SimpleNamespace(config=config, modules=lambda: list(mixers))


def _gated_delta_net(**marks):
    def forward(self, *args, **kwargs):
        pass

    for name, value in marks.items():
        setattr(forward, name, value)
    return type("Qwen3_5GatedDeltaNet", (), {"forward": forward})()


HYBRID = ["linear_attention", "full_attention"]
TREE_ROUTED = {"_axolotl_gdn_kernel_interface": True, "_axolotl_recurrent_tree": True}


class TestSharesPrefixes:
    def test_attention_models_share(self):
        assert shares_prefixes(_model("llama"))
        assert shares_prefixes(
            _model("gemma3", ["sliding_attention", "full_attention"])
        )

    def test_trie_routed_gated_delta_nets_share(self):
        mixers = [_gated_delta_net(**TREE_ROUTED)] * 2
        assert shares_prefixes(_model("qwen3_5_text", HYBRID, mixers))

    def test_other_patched_recurrent_models_pack_chains(self):
        assert not shares_prefixes(_model("mamba2"))
        varlen_only = [_gated_delta_net(_axolotl_gdn_kernel_interface=True)]
        assert not shares_prefixes(_model("qwen3_5_text", HYBRID, varlen_only))

    def test_gated_delta_nets_need_flash_linear_attention(self, monkeypatch):
        from axolotl.core.trainers.grpo import trl_async_packing

        monkeypatch.setattr(
            trl_async_packing.importlib.util, "find_spec", lambda name: None
        )
        mixers = [_gated_delta_net(**TREE_ROUTED)]
        with pytest.raises(ValueError, match="flash-linear-attention"):
            shares_prefixes(_model("qwen3_5_text", HYBRID, mixers))

    def test_stock_gated_delta_net_raises(self):
        with pytest.raises(ValueError, match="stock forward"):
            shares_prefixes(_model("qwen3_5_text", HYBRID, [_gated_delta_net()]))

    @pytest.mark.parametrize("model", [_model("jamba"), _model("my_hybrid", HYBRID)])
    def test_unsupported_recurrent_models_raise(self, model):
        with pytest.raises(ValueError, match="state would leak"):
            shares_prefixes(model)


def _reference_causal_conv1d(x, weight, bias, activation, cu_seqlens):
    """Pure-torch stand-in for FLA's varlen causal conv1d on ``[1, T, D]``."""
    width = weight.shape[1]
    outs = []
    for start, end in zip(
        cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True
    ):
        seq = torch.nn.functional.pad(x[:, start:end].transpose(1, 2), (width - 1, 0))
        out = torch.nn.functional.conv1d(seq, weight[:, None], bias, groups=x.shape[-1])
        outs.append(torch.nn.functional.silu(out).transpose(1, 2))
    return torch.cat(outs, dim=1), None


def _gated_delta_rule(query, key, value, g, beta, state=None):
    """The gated delta rule token by token on ``[1, T, H, D]``; returns the final state."""
    query = torch.nn.functional.normalize(query.float(), dim=-1, eps=1e-6)
    key = torch.nn.functional.normalize(key.float(), dim=-1, eps=1e-6)
    query = query * query.shape[-1] ** -0.5
    value, g, beta = value.float(), g.float(), beta.float()
    if state is None:
        state = query.new_zeros(query.shape[2], key.shape[-1], value.shape[-1])
    outs = []
    for t in range(query.shape[1]):
        state = state * g[0, t].exp()[:, None, None]
        delta = (value[0, t] - (state * key[0, t, :, :, None]).sum(-2)) * beta[
            0, t, :, None
        ]
        state = state + key[0, t, :, :, None] * delta[:, None, :]
        outs.append((state * query[0, t, :, :, None]).sum(-2))
    return torch.stack(outs)[None].to(value.dtype), state


def _reference_chunk_gated_delta_rule(
    query, key, value, *, g, beta, cu_seqlens, initial_state=None, **kwargs
):
    """Pure-torch stand-in for FLA's varlen chunk_gated_delta_rule."""
    bounds = cu_seqlens.tolist()
    outs, finals = [], []
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        out, final = _gated_delta_rule(
            *(x[:, a:b] for x in (query, key, value, g, beta)),
            state=None if initial_state is None else initial_state[i],
        )
        outs.append(out)
        finals.append(final)
    return torch.cat(outs, dim=1), torch.stack(finals)


@pytest.fixture
def qwen3_5_packing_patch(monkeypatch):
    """axolotl's Qwen3.5 packing patch, with FLA swapped for reference kernels."""
    from transformers.models.qwen3_5 import modeling_qwen3_5
    from trl.experimental.async_grpo import packing

    from axolotl.monkeypatch.models.qwen3_5 import modeling as qwen3_5_packing

    monkeypatch.setattr(packing, "tree_attention", reference_tree_attention)
    for cls in (
        modeling_qwen3_5.Qwen3_5DecoderLayer,
        modeling_qwen3_5.Qwen3_5GatedDeltaNet,
    ):
        monkeypatch.setattr(cls, "forward", cls.forward)
    qwen3_5_packing.patch_qwen3_5_modeling_packing()
    monkeypatch.setattr(qwen3_5_packing, "_FLA_COMPILED_OPS", False)
    calls = []

    def causal_conv1d(x, weight, bias, activation, cu_seqlens):
        calls.append(cu_seqlens.tolist())
        return _reference_causal_conv1d(x, weight, bias, activation, cu_seqlens)

    monkeypatch.setattr(qwen3_5_packing, "fla_causal_conv1d", causal_conv1d)
    monkeypatch.setattr(
        qwen3_5_packing,
        "fla_chunk_gated_delta_rule",
        _reference_chunk_gated_delta_rule,
    )
    return types.SimpleNamespace(
        conv_calls=calls, gated_delta_net=modeling_qwen3_5.Qwen3_5GatedDeltaNet
    )


def test_packed_hybrid_rows_match_each_sample_alone(qwen3_5_packing_patch):
    """Packed rows, prompts shared or not, score every sample as it scores alone.

    State leaking across samples or branches would show up as drift. Every path
    runs axolotl's patched GatedDeltaNet forward: the rows through the trie
    kwargs, the samples alone through the position_ids boundary detection.
    """
    from trl.experimental.async_grpo import packing

    packing.register_tree_attention()
    model = _tiny_qwen3_5(64)
    linear_attn = model.model.language_model.layers[0].linear_attn
    assert type(linear_attn).forward._axolotl_recurrent_tree
    rng = np.random.default_rng(0)
    samples = []
    for group_id in range(2):
        prompt = rng.integers(1, 64, size=6 + group_id).tolist()
        for g in range(3):
            completion = rng.integers(1, 64, size=3 + g).tolist()
            samples.append(_sample(prompt + completion, len(prompt), group_id))

    def grads():
        found = {
            n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None
        }
        model.zero_grad()
        return found

    def score_row(row):
        model.set_attn_implementation(packing.TREE_ATTENTION)
        for sample in samples:
            row.add(sample)
        packed = row.pack()
        tree = {
            "tree_kv_index": packed["tree_kv_index"],
            "tree_cu_seqlens_q": packed["tree_cu_seqlens_q"],
            "tree_cu_seqlens_k": packed["tree_cu_seqlens_k"],
            "tree_max_seqlen_q": int(packed["tree_max_seqlens"][0]),
            "tree_max_seqlen_k": int(packed["tree_max_seqlens"][1]),
        }
        if isinstance(row, ChainRow):
            tree.update(sample_boundaries(tree))
        logits = model(
            input_ids=packed["input_ids"][None],
            position_ids=packed["position_ids"][None],
            use_cache=False,
            **tree,
        ).logits[0]
        index = packed["token_index"]
        log_probs = torch.log_softmax(logits[index], -1).gather(
            1, packed["shift_labels"][index][:, None]
        )[:, 0]
        log_probs.sum().backward()
        return len(packed["input_ids"]), log_probs.detach(), grads()

    def score_alone():
        model.set_attn_implementation("sdpa")
        log_probs = []
        for sample in samples:
            ids = torch.as_tensor(sample["input_ids"])[None]
            logits = model(
                input_ids=ids,
                position_ids=torch.arange(ids.shape[1])[None],
                use_cache=False,
            ).logits[0, :-1]
            trained = torch.as_tensor(sample["completion_mask"][1:]).bool()
            log_probs.append(
                torch.log_softmax(logits[trained], -1).gather(
                    1, ids[0, 1:][trained][:, None]
                )[:, 0]
            )
        torch.cat(log_probs).sum().backward()
        return torch.cat(log_probs).detach(), grads()

    shared_len, shared, shared_grads = score_row(TreeRow())
    chain_len, chains, chain_grads = score_row(ChainRow())
    alone, alone_grads = score_alone()
    assert shared_len < chain_len
    for packed_log_probs, packed_grads in (
        (shared, shared_grads),
        (chains, chain_grads),
    ):
        torch.testing.assert_close(packed_log_probs, alone, rtol=1e-5, atol=1e-5)
        assert packed_grads.keys() == alone_grads.keys()
        for name, grad in packed_grads.items():
            torch.testing.assert_close(grad, alone_grads[name], rtol=1e-4, atol=1e-5)
    # The trie's conv ran over root-to-segment paths, not over the packed row.
    assert any(calls[-1] > shared_len for calls in qwen3_5_packing_patch.conv_calls)


def _tiny_qwen3_5(vocab_size):
    """A tiny Qwen3.5 checkpoint as axolotl loads it: the multimodal class."""
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    config = Qwen3_5Config(
        text_config={
            "vocab_size": vocab_size,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
        },
    )
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(config)


@pytest.mark.parametrize("routes_trie", [True, False])
def test_trains_hybrid_model_on_cpu(
    download_smollm2_135m_instruct_model,
    tmp_path,
    monkeypatch,
    qwen3_5_packing_patch,
    routes_trie,
):
    """Rollouts share prompts when the GatedDeltaNet patch follows the trie, else
    each is its own chain; either way only the adapter trains."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer
    from trl.experimental.async_grpo import async_grpo_trainer

    monkeypatch.setattr(
        async_grpo_trainer, "get_peak_flops_per_device", lambda *a, **k: None
    )
    if not routes_trie:
        forward = qwen3_5_packing_patch.gated_delta_net.forward
        monkeypatch.setattr(forward, "_axolotl_recurrent_tree", False)

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
    model = _tiny_qwen3_5(len(tokenizer))
    layer_types = list(model.config.get_text_config().layer_types)
    model = get_peft_model(
        model,
        LoraConfig(r=4, target_modules=r".*language_model.*\.(q_proj|v_proj|out_proj)"),
    )
    trainable_before = {n: p.requires_grad for n, p in model.named_parameters()}

    prompt = [{"role": "user", "content": "What is 2+2?"}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, return_dict=False
    )
    rng = np.random.default_rng(0)
    samples, lengths = [], set()
    for group_id in range(16):
        for g in range(4):
            completion = [*rng.integers(100, 1000, size=2 + g), tokenizer.eos_token_id]
            ids = np.asarray(prompt_ids + completion, dtype=np.int64)
            lengths.add(len(ids))
            samples.append(
                RolloutSample(
                    prompt=prompt,
                    completion=[{"role": "assistant", "content": "x"}],
                    input_ids=ids,
                    completion_mask=np.asarray(
                        [0] * len(prompt_ids) + [1] * len(completion), dtype=np.int64
                    ),
                    old_log_probs=np.zeros(len(ids), np.float32),
                    advantage=float(g) - 1.5,
                    model_version=0,
                    group_id=group_id,
                    metrics={"reward": float(g), "reward_std": 1.0},
                )
            )

    args = AxolotlAsyncGRPOConfig(
        output_dir=str(tmp_path),
        max_steps=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        bf16=False,
        use_cpu=True,
        token_budget=0,
        report_to=[],
        save_strategy="no",
        gradient_checkpointing=False,
    )
    trainer = AxolotlAsyncGRPOTrainer(
        model,
        reward_funcs=[],
        args=args,
        train_dataset=Dataset.from_dict({"prompt": [prompt] * 16}),
        processing_class=tokenizer,
        rollout_worker=StubWorker(samples),
        weight_transfer=StubTransfer(),
    )
    assert trainer._share_prefixes is routes_trie
    text_config = trainer.model.get_base_model().config.get_text_config()
    assert text_config.layer_types == layer_types
    # TRL's VLM branch would have made the whole text model trainable.
    params = dict(trainer.model.named_parameters())
    assert {n: p.requires_grad for n, p in params.items()} == trainable_before
    assert all("lora_" in n for n, p in params.items() if p.requires_grad)

    trained = {n: p.detach().clone() for n, p in params.items() if p.requires_grad}
    frozen = {n: p.detach().clone() for n, p in params.items() if not p.requires_grad}
    trainer.train()

    ratios = [
        log["batch/packing_ratio"]
        for log in trainer.state.log_history
        if "batch/packing_ratio" in log
    ]
    calls = qwen3_5_packing_patch.conv_calls
    assert ratios and calls
    if routes_trie:
        assert min(ratios) > 1
    else:
        assert set(ratios) == {1.0}
        # One sequence per rollout reaches the GatedDeltaNet kernels.
        for cu_seqlens in calls:
            assert set(np.diff(cu_seqlens).tolist()) <= lengths
    params = dict(trainer.model.named_parameters())
    assert all(not torch.equal(trained[n], params[n].detach()) for n in trained)
    assert all(torch.equal(frozen[n], params[n].detach()) for n in frozen)
